"""Parse QQ chat exports (desktop format or QQChatExporter V5) into ingestable
windows, then POST each window to the Kairos ingest API.

Desktop format (per message):
    YYYY-MM-DD HH:MM:SS <qq>(<display name>)
    body line(s), until a blank line

QQChatExporter V5 format (nickname only, no QQ id — resolved via nickmap):
    [nickname]:
    时间: YYYY-MM-DD HH:MM:SS
    内容: body line(s)
    <blank line>

Usage:
    python tools/import_chat_txt.py stats  <file> [channel_id]
    python tools/import_chat_txt.py import <file> <channel_id> [--workers=4]
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HEADER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*(\S*)\((.*?)\)\s*$")
MARKER_RE = re.compile(r"^\[(图片|表情|语音|视频|文件|动画表情|链接|卡片|闪照|红包|转账|位置|分享)\]")
MARKER_BLOCK_RE = re.compile(r"^\[(图片|表情|语音|视频|文件|动画表情|闪照|视频)\b")
GAP_MINUTES = 30
WINDOW_CAP = 240
API = "http://192.168.10.13:8095/api/knowledge/ingest"
CHECKPOINT = Path("data/import_checkpoint.json")
NICKMAP_PATH = Path("data/nickmap.json")


def read_text(path: str) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def load_nickmap() -> dict[str, str]:
    if NICKMAP_PATH.exists():
        return json.loads(NICKMAP_PATH.read_text(encoding="utf-8"))
    return {}


# --- format detection --------------------------------------------------

def detect_format(text: str) -> str:
    return "qqce" if "[QQChatExporter" in text[:500] else "desktop"


# --- desktop format parser ----------------------------------------------

def parse_desktop(text: str) -> list[dict]:
    messages: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            if cur:
                messages.append(cur)
            stamp, uid, nick = m.group(1), m.group(2).strip(), m.group(3).strip()
            cur = {"time": stamp, "user_id": uid, "nickname": nick, "_lines": []}
            continue
        if cur is None:
            continue
        stripped = line.strip()
        if not stripped:
            if cur["_lines"]:
                messages.append(cur)
                cur = None
            continue
        cur["_lines"].append(stripped)
    if cur and cur["_lines"]:
        messages.append(cur)
    return _finalize(messages)


# --- QQChatExporter V5 parser -------------------------------------------

def parse_qqce(text: str, nickmap: dict[str, str]) -> list[dict]:
    lines = text.splitlines()
    messages: list[dict] = []
    i = 0
    n = len(lines)
    state = "IDLE"
    cur_nick = ""
    cur_time = ""
    cur_body: list[str] = []
    while i < n:
        line = lines[i].rstrip()
        if state == "IDLE":
            if line.endswith(":") and not line.startswith("时间") and not line.startswith("内容") and not line.startswith("聊天") and not line.startswith("导出") and not line.startswith("消息") and not line.startswith("时间范围"):
                raw = line[:-1].strip().lstrip("[").strip()
                # strip role tags like "管理员] " / "网管] " left after lstrip("[")
                cur_nick = re.sub(r"^[^\]]{1,6}\]\s*", "", raw) if "]" in raw[:8] else raw
                state = "HEADER"
        elif state == "HEADER":
            m = re.match(r"^时间:\s*(.+)$", line)
            if m:
                cur_time = m.group(1).strip()
                state = "TIME"
            else:
                state = "IDLE"
        elif state == "TIME":
            m = re.match(r"^内容:\s*(.*)$", line)
            if m:
                if m.group(1).strip():
                    cur_body.append(m.group(1).strip())
                state = "CONTENT"
            else:
                state = "IDLE"
        elif state == "CONTENT":
            if not line.strip():
                messages.append({"time": cur_time, "user_id": "", "nickname": cur_nick, "_lines": cur_body})
                cur_body = []
                state = "IDLE"
            else:
                cur_body.append(line.strip())
        i += 1
    if state == "CONTENT" and cur_body:
        messages.append({"time": cur_time, "user_id": "", "nickname": cur_nick, "_lines": cur_body})
    return _finalize(messages, nickmap)


def _finalize(messages: list[dict], nickmap: dict[str, str] | None = None) -> list[dict]:
    nickmap = nickmap or {}
    out = []
    dropped = 0
    for msg in messages:
        lines = [ln for ln in msg["_lines"] if not MARKER_RE.match(ln) and not MARKER_BLOCK_RE.match(ln)]
        # drop resource-list residue lines ("资源: N 个文件", "- image: ...")
        lines = [ln for ln in lines if not ln.startswith(("资源:", "- image", "- video", "- file"))]
        body = "\n".join(lines).strip()
        if not body:
            dropped += 1
            continue
        uid = msg["user_id"]
        if not uid and nickmap:
            uid = nickmap.get(msg["nickname"], "")
        if not uid:
            uid = msg["nickname"]  # fallback: nickname itself as stable identity
        out.append({"time": msg["time"], "user_id": uid, "nickname": msg["nickname"], "text": body[:800]})
    return out


def parse_messages(text: str, nickmap: dict[str, str] | None = None) -> list[dict]:
    fmt = detect_format(text)
    if fmt == "qqce":
        return parse_qqce(text, nickmap or {})
    return parse_desktop(text)


def split_windows(messages: list[dict]) -> list[list[dict]]:
    def ts(m: dict) -> float:
        return time.mktime(time.strptime(m["time"], "%Y-%m-%d %H:%M:%S"))

    windows: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        if current:
            gap = (ts(msg) - ts(current[-1])) / 60.0
            if gap > GAP_MINUTES or len(current) >= WINDOW_CAP:
                if len(current) >= 2:
                    windows.append(current)
                current = []
        current.append(msg)
    if len(current) >= 2:
        windows.append(current)
    # merge any trailing tiny window into the previous one to avoid losing it
    if windows and len(windows[-1]) < 2 and len(windows) >= 2:
        windows[-2].extend(windows.pop())
    return windows


def post_window(channel_id: str, window: list[dict]) -> dict:
    payload = json.dumps(
        {"source": "qq", "channel_id": channel_id, "title": "", "messages": window},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_checkpoint(channel_id: str) -> set:
    p = Path(f"data/import_checkpoint_{channel_id}.json")
    if p.exists():
        return set(json.loads(p.read_text()))
    # legacy shared file (from the first import)
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text()))
    return set()


def save_checkpoint(channel_id: str, done: set) -> None:
    p = Path(f"data/import_checkpoint_{channel_id}.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(sorted(done)))


def main() -> None:
    mode, path = sys.argv[1], sys.argv[2]
    text = read_text(path)
    nickmap = load_nickmap()
    fmt = detect_format(text)
    print(f"format: {fmt} | nickmap: {len(nickmap)} entries")
    messages = parse_messages(text, nickmap)
    print(f"parsed {len(messages)} messages from {path}")
    if mode == "stats":
        windows = split_windows(messages)
        spans = [(w[0]["time"], w[-1]["time"]) for w in windows[:3]]
        print(f"windows: {len(windows)} (cap={WINDOW_CAP}, gap>{GAP_MINUTES}min)")
        print(f"first windows span: {spans}")
        uids = {m['user_id'] for m in messages}
        print(f"unique senders: {len(uids)}")
        return
    channel_id = sys.argv[3]
    workers = next((int(a.split('=')[1]) for a in sys.argv[4:] if a.startswith('--workers=')), 4)
    windows = split_windows(messages)
    done = load_checkpoint(channel_id)
    todo = [(i, w) for i, w in enumerate(windows) if i not in done]
    print(f"importing {len(todo)}/{len(windows)} windows with {workers} workers -> {API}")

    ok = err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(post_window, channel_id, w): (i, w) for i, w in todo}
        for fut in as_completed(futures):
            i, w = futures[fut]
            try:
                fut.result()
                ok += 1
                done.add(i)
                save_checkpoint(channel_id, done) if ok % 5 == 0 else None
            except Exception as exc:  # noqa: BLE001
                err += 1
                print(f"[ERR] window {i} ({w[0]['time']}): {exc}", flush=True)
            total_done = ok + err
            if total_done % 25 == 0:
                rate = total_done / max(1, time.time() - t0) * 60
                eta = (len(todo) - total_done) / max(rate / 60, 1e-9) / 60
                print(f"progress {total_done}/{len(todo)} ok={ok} err={err} "
                      f"{rate:.1f} win/min eta={eta:.0f}min", flush=True)
    save_checkpoint(channel_id, done)
    print(f"DONE ok={ok} err={err} elapsed={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
