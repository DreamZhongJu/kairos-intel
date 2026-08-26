#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured reimport of the ersi group from chunked JSONL export.

Every message carries sender uin + time, so identities anchor deterministically:
  - render lines as 『群名片(uin)』 text so the extractor sees speakers
  - @ mentions and replies become direct uid-anchored relations (zero LLM cost)
  - extracted entity names matching a window speaker's card/nick are pinned
    to qq:<uin> via upsert_entity(canonical=...)
  - relations inherit real window time bounds when the extractor gives none
Sharding + flock checkpoint mirror import_chat_txt.py conventions.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import glob
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")
from kairos.knowledge import engine  # noqa: E402
from kairos.knowledge import extract as kg_extract  # noqa: E402

CHUNK_GLOB = "/app/data/ersi_jsonl/chunks/*.jsonl"
CKPT_GLOB = "/app/data/import_checkpoint_ersi_jsonl_v1.s*.json"


def ckpt_path(shard: int) -> str:
    return f"/app/data/import_checkpoint_ersi_jsonl_v1.s{shard}.json"


GROUP_NAME = "二色恋花蝶·豫康网吧版"
GROUP_CANON = "group:830070676"
MAX_MSGS = 120
MAX_CHARS = 2800

NOISE_RE = re.compile(r"\s+")


def load_messages():
    """Yield parsed messages in chronological order."""
    files = sorted(glob.glob(CHUNK_GLOB))
    if not files:
        raise SystemExit(f"no chunks under {CHUNK_GLOB}")
    buf = []
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if m.get("system") or m.get("recalled"):
                    continue
                ts = m.get("timestamp")
                if not ts:
                    continue
                buf.append(m)
    buf.sort(key=lambda x: int(x["timestamp"]))
    return buf


def display_name(sender: dict) -> str:
    return (sender.get("groupCard") or sender.get("nickname")
            or sender.get("name") or "").strip() or f"qq{sender.get('uin', '')}"


def build_windows(msgs):
    """Group by Shanghai calendar day, then split by size caps."""
    days = {}
    for m in msgs:
        d = datetime.datetime.fromtimestamp(
            int(m["timestamp"]) / 1000, tz=datetime.timezone(datetime.timedelta(hours=8))
        ).date().isoformat()
        days.setdefault(d, []).append(m)
    windows = []
    for day in sorted(days):
        cur = []
        chars = 0
        for m in days[day]:
            s = m.get("sender") or {}
            text = ((m.get("content") or {}).get("text") or "").strip()
            rsrc = (m.get("content") or {}).get("resources") or []
            if not text and rsrc:
                kinds = ",".join(sorted({r.get("type", "file") for r in rsrc}))
                text = f"[{kinds}]"
            if not text:
                continue
            line = f"『{display_name(s)}({s.get('uin', '?')})』 {NOISE_RE.sub(' ', text)[:400]}"
            cur.append({"m": m, "line": line})
            chars += len(line)
            if len(cur) >= MAX_MSGS or chars >= MAX_CHARS:
                windows.append((day, cur))
                cur, chars = [], 0
        if cur:
            windows.append((day, cur))
    return windows


def win_sig(day, items):
    h = hashlib.sha1()
    h.update(day.encode())
    h.update(items[0]["m"]["id"].encode())
    h.update(items[-1]["m"]["id"].encode())
    return h.hexdigest()[:16]


def load_ckpt(shard):
    p = ckpt_path(shard)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_ckpt(shard, data):
    tmp = ckpt_path(shard) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, ckpt_path(shard))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6, help="concurrent windows per process")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first N windows (dry-run)")
    args = ap.parse_args()

    out = open(f"/tmp/_imp_{args.shard}.out", "a", buffering=1)

    msgs = load_messages()
    windows = build_windows(msgs)
    todo = [(i, w) for i, w in enumerate(windows) if i % args.shards == args.shard]
    if args.limit:
        todo = todo[: args.limit]
    out.write(f"total_msgs={len(msgs)} total_windows={len(windows)} "
              f"mine={len(todo)} shard={args.shard}/{args.shards} jobs={args.jobs}\n")

    ckpt = load_ckpt(args.shard)
    done = ckpt.setdefault("done", {})

    if args.dry_run:
        for i, (day, items) in todo[:3]:
            out.write(f"--- window#{i} {day} msgs={len(items)} ---\n")
            for it in items[:6]:
                out.write("  " + it["line"][:150] + "\n")
        out.write("DRY-RUN-END\n")
        out.close()
        return

    engine.init()
    gid = engine.upsert_entity(GROUP_NAME, "群组", canonical=GROUP_CANON)

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    state = {"n": 0, "ok_llm": 0, "skip": 0, "errs": 0}
    t0 = time.time()

    def _process_window(task):
        i, (day, items) = task
        sig = win_sig(day, items)
        with lock:
            if done.get(sig):
                state["skip"] += 1
                return None
        try:
            text = "\n".join(it["line"] for it in items)
            title = f"{GROUP_NAME} {day} 第{i}窗"
            con0 = engine._connect()
            row0 = con0.execute("SELECT id FROM documents WHERE title=?", (title,)).fetchone()
            con0.close()
            if row0:
                doc_id = int(row0["id"])
            else:
                doc_id = engine.add_document(title, text, kind="qqjson",
                                             source=f"qqjson:830070676")
            con = engine._connect()
            rows = con.execute(
                "SELECT id FROM chunks WHERE doc_id=? ORDER BY seq", (doc_id,)
            ).fetchall()
            chunk_ids = [r["id"] for r in rows]
            con.close()

            # roster for post-anchoring
            roster = {}
            for it in items:
                s = it["m"].get("sender") or {}
                uin = str(s.get("uin") or "")
                if not uin.isdigit():
                    continue
                variants = {display_name(s), s.get("nickname") or "", s.get("name") or ""}
                roster.setdefault(uin, set()).update(v for v in variants if v)

            def ensure_person(uin):
                cards = roster.get(str(uin))
                disp = sorted(cards)[0] if cards else f"qq用户{uin}"
                return engine.upsert_entity(disp, "人名", canonical=f"qq:{uin}")

            # deterministic mention/reply edges
            seen_det = set()
            for it in items:
                m = it["m"]
                suin = str((m.get("sender") or {}).get("uin") or "")
                if not suin.isdigit():
                    continue
                sid = ensure_person(suin)
                els = (m.get("content") or {}).get("elements") or []
                for el in els:
                    d = el.get("data") or {}
                    muin = str(d.get("uin") or "")
                    if el.get("type") == "at" and muin.isdigit() and muin != suin:
                        key = (sid, "艾特", muin)
                        if key not in seen_det:
                            seen_det.add(key)
                            engine.add_relation_by_ids(
                                sid, "艾特", ensure_person(muin),
                                confidence=7, valid_from=day)
                    elif el.get("type") == "reply":
                        ru = str(d.get("senderUin") or "")
                        if ru.isdigit() and ru != suin:
                            key = (sid, "回复", ru)
                            if key not in seen_det:
                                seen_det.add(key)
                                engine.add_relation_by_ids(
                                    sid, "回复", ensure_person(ru),
                                    confidence=6, valid_from=day)
                # membership
                engine.add_relation_by_ids(sid, "活跃于", gid,
                                           confidence=9, valid_from=day)

            had_llm = False
            n_rels = 0
            if len(text) >= 40:
                extracted = _extract_with_deadline(text, args.provider)
                ents = extracted.get("entities", [])
                rels = extracted.get("relations", [])
                had_llm = bool(rels) or bool(ents)
                n_rels = len(rels)
                for ent in ents:
                    nm = (ent.get("name") or "").strip()
                    if not nm:
                        continue
                    canon = None
                    nl = nm.lower()
                    for uin, cards in roster.items():
                        for cv in cards:
                            cvl = cv.lower()
                            if nl == cvl or strip_key(nl) == strip_key(cvl):
                                canon = f"qq:{uin}"
                                break
                        if canon:
                            break
                    try:
                        engine.upsert_entity(nm, ent.get("type") or "entity",
                                             canonical=canon)
                    except Exception:
                        pass
                for rel in rels:
                    try:
                        ci = int(rel.pop("_ci", 0) or 0)
                        cid = chunk_ids[ci] if 0 <= ci < len(chunk_ids) else None
                        engine.add_relation(
                            rel["subject"], rel["predicate"], rel["object"],
                            chunk_id=cid,
                            confidence=int(rel.get("confidence", 1)),
                            valid_from=rel.get("time_start") or day,
                            valid_to=rel.get("time_end"),
                            playful=bool(rel.get("playful")),
                        )
                    except Exception:
                        pass

            with lock:
                done[sig] = {"w": i, "day": day, "msgs": len(items)}
                state["n"] += 1
                state["ok_llm"] += int(had_llm)
                n = state["n"]
                out.write(f"OK w{i} {day} rels={n_rels}\n")
                if n % 10 == 0:
                    save_ckpt(args.shard, ckpt)
                    rate = n / max(1e-9, time.time() - t0) * 60
                    out.write(f"progress {n}/{len(todo)} llm={state['ok_llm']} "
                              f"skip={state['skip']} rate={rate:.0f}/min\n")
            return True
        except Exception as e:
            with lock:
                state["errs"] += 1
            out.write(f"WERR w{i}: {type(e).__name__}: {str(e)[:200]}\n")
            return False

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(_process_window, todo))
    save_ckpt(args.shard, ckpt)
    out.write(f"DONE shard={args.shard} processed={len(todo)} llm={state['ok_llm']} "
              f"skip={state['skip']} err={state['errs']} mins={(time.time()-t0)/60:.1f}\n")
    out.close()


_KEY_RE = re.compile(r"[\uff08(].*?[)\uff09]|\s+")


def strip_key(n: str) -> str:
    return _KEY_RE.sub("", n or "").lower()


_HATCH = None


def _extract_with_deadline(text: str, provider: str, timeout: int = 150):
    """Run one extraction under a hard wall-clock deadline.

    Slow-drip responses evade httpx read timeouts indefinitely; an abandoned
    future's thread dies on its own transport timeout shortly after we give up.
    """
    global _HATCH
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    if _HATCH is None:
        _HATCH = ThreadPoolExecutor(max_workers=64)
    fut = _HATCH.submit(lambda: kg_extract.extract(text, provider=provider))
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        return {"entities": [], "relations": []}


if __name__ == "__main__":
    main()
