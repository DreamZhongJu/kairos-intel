#!/usr/bin/env python3
"""Interactively import a user-owned Reddit browser session into rdt-cli.

The cookie header is read with getpass, so it is not printed or put in shell
history.  This program intentionally offers no non-interactive flag.
"""

from __future__ import annotations

import getpass
import json
import os
import tempfile
from http.cookies import SimpleCookie
from pathlib import Path


TARGET = Path("/root/.config/rdt-cli/credential.json")


def main() -> None:
    raw = getpass.getpass("Paste Reddit Cookie-Editor Header String (input hidden): ").strip()
    parsed = SimpleCookie()
    parsed.load(raw)
    cookies = {name: morsel.value for name, morsel in parsed.items()}
    if not cookies or "reddit_session" not in cookies:
        raise SystemExit("Import cancelled: a reddit_session cookie is required.")

    TARGET.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps({"cookies": cookies, "source": "manual:cookie-editor"}, ensure_ascii=False)
    fd, temporary_name = tempfile.mkstemp(prefix=".credential.", dir=TARGET.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, TARGET)
        print("Reddit session imported. Run: rdt status")
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
