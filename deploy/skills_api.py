#!/usr/bin/env python3
"""Open skill reader: serves SKILL.md catalogs over HTTP without dependencies.

The skill contents themselves stay outside the assistant repository; this
service is the independent, open reading interface for them.

Endpoints:
  GET /              service info (JSON)
  GET /skills        list of {name, description} for every skill folder
  GET /skills/<name> raw SKILL.md content (text/markdown)
  GET /health        ok

Configuration via environment:
  SKILLS_ROOT   directory containing skill folders (default: ./skills)
  SKILLS_PORT   listen port (default 8777)
  SKILLS_TOKEN  optional bearer token; when set, every request must send
                "Authorization: Bearer <token>"
"""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.getenv("SKILLS_ROOT", str(Path(__file__).resolve().parent / "skills")))
PORT = int(os.getenv("SKILLS_PORT", "8777"))
TOKEN = os.getenv("SKILLS_TOKEN", "").strip()
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_frontmatter(text: str) -> dict[str, str]:
    """Minimal YAML-frontmatter reader: name and description only."""
    info: dict[str, str] = {}
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return info
    current = ""
    for line in match.group(1).splitlines():
        if line.startswith((" ", "\t")):
            if current:
                info[current] = (info.get(current, "") + " " + line.strip()).strip()
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            current = key.strip()
            info[current] = value.strip().strip("'\"")
    return info


def skill_dirs() -> list[Path]:
    if not ROOT.is_dir():
        return []
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def skill_meta(path: Path) -> dict[str, str]:
    text = (path / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    info = parse_frontmatter(text)
    return {
        "name": info.get("name") or path.name,
        "description": info.get("description") or "",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quieter access log
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _authorized(self) -> bool:
        if not TOKEN:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("", "/", "/health"):
            self._json(200, {"service": "kairos-skill-reader", "skills": len(skill_dirs()), "root": str(ROOT)})
            return
        if path == "/skills":
            self._json(200, {"skills": [skill_meta(p) for p in skill_dirs()]})
            return
        prefix = "/skills/"
        if path.startswith(prefix):
            name = path[len(prefix):]
            if not SAFE_NAME.match(name):
                self._json(400, {"error": "invalid skill name"})
                return
            target = ROOT / name / "SKILL.md"
            if not target.is_file():
                self._json(404, {"error": f"skill '{name}' not found"})
                return
            body = target.read_text(encoding="utf-8", errors="replace").encode("utf-8")
            self._send(200, body, "text/markdown; charset=utf-8")
            return
        self._json(404, {"error": "not found"})


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"skill reader listening on :{PORT} root={ROOT} skills={len(skill_dirs())}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()