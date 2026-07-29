"""OAuth callback and automatic token renewal for the trusted Feishu user."""
from __future__ import annotations

import ast
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from cryptography.fernet import Fernet
from flask import Flask, request

from assistant.infrastructure.settings import APP_ID, APP_SECRET, DATA_DIR, DB_PATH

PUBLIC_URL = os.getenv("OAUTH_PUBLIC_URL", "").rstrip("/")
FERNET = Fernet(os.environ["TOKEN_ENCRYPTION_KEY"].encode())
SCOPES = " ".join([
    "offline_access", "wiki:wiki:readonly", "wiki:wiki", "docx:document:readonly", "docx:document", "search:docs:read", "drive:drive",
    "calendar:calendar:readonly", "calendar:calendar",
])

app = Flask(__name__)
_refresh_lock = threading.Lock()


def init_oauth() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("CREATE TABLE IF NOT EXISTS oauth_states (state TEXT PRIMARY KEY, expires_at INTEGER NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS user_tokens (id INTEGER PRIMARY KEY CHECK(id=1), token BLOB NOT NULL, updated_at INTEGER NOT NULL)")


def authorization_link() -> str | None:
    if not PUBLIC_URL:
        return None
    state = secrets.token_urlsafe(32)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM oauth_states WHERE expires_at < ?", (int(time.time()),))
        con.execute("INSERT INTO oauth_states VALUES (?, ?)", (state, int(time.time()) + 900))
    params = {
        "client_id": APP_ID, "response_type": "code", "redirect_uri": f"{PUBLIC_URL}/oauth/callback",
        "scope": SCOPES, "state": state,
    }
    return "https://accounts.feishu.cn/open-apis/authen/v1/authorize?" + urlencode(params)


def _token_data(payload: dict) -> dict:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def _save_token_payload(payload: dict) -> None:
    encrypted = FERNET.encrypt(repr(payload).encode())
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR REPLACE INTO user_tokens(id, token, updated_at) VALUES(1, ?, ?)", (encrypted, int(time.time())))


def _load_token_payload() -> tuple[dict, int]:
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT token, updated_at FROM user_tokens WHERE id=1").fetchone()
    if not row:
        raise RuntimeError("尚未完成个人飞书授权，请先发送“授权飞书”。")
    return ast.literal_eval(FERNET.decrypt(row[0]).decode()), int(row[1])


@app.get("/healthz")
def healthz():
    return "ok", 200


@app.get("/oauth/callback")
def callback():
    state, code = request.args.get("state", ""), request.args.get("code", "")
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT expires_at FROM oauth_states WHERE state=?", (state,)).fetchone()
        con.execute("DELETE FROM oauth_states WHERE state=?", (state,))
    if not row or row[0] < int(time.time()) or not code:
        return "授权链接无效或已过期，请回到飞书重新发起授权。", 400
    response = requests.post("https://open.feishu.cn/open-apis/authen/v2/oauth/token", json={
        "grant_type": "authorization_code", "code": code, "client_id": APP_ID,
        "client_secret": APP_SECRET, "redirect_uri": f"{PUBLIC_URL}/oauth/callback",
    }, timeout=30).json()
    if response.get("code") != 0:
        return "飞书授权失败，请回到飞书重新尝试。", 400
    _save_token_payload(response)
    return "授权成功。现在可回到飞书继续使用凯伊。", 200


def start_oauth_server() -> None:
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False), daemon=True).start()


def user_access_token() -> str:
    """Return a valid user token, refreshing it before expiry when possible."""
    with _refresh_lock:
        payload, updated_at = _load_token_payload()
        data = _token_data(payload)
        token = data.get("access_token")
        expires_in = int(data.get("expires_in") or 0)
        if token and expires_in and time.time() < updated_at + expires_in - 120:
            return token
        refresh_token = data.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("个人飞书授权已过期且没有刷新令牌，请发送“授权飞书”重新授权。")
        response = requests.post("https://open.feishu.cn/open-apis/authen/v2/oauth/token", json={
            "grant_type": "refresh_token", "client_id": APP_ID,
            "client_secret": APP_SECRET, "refresh_token": refresh_token,
        }, timeout=30).json()
        if response.get("code") != 0:
            raise RuntimeError("个人飞书授权已过期，请发送“授权飞书”重新授权。")
        _save_token_payload(response)
        token = _token_data(response).get("access_token")
    if not token:
        raise RuntimeError("授权令牌刷新失败，请发送“授权飞书”重新授权。")
    return token
