# -*- coding: utf-8 -*-
"""Server entry point untuk package routes.

Jalankan:
    python -m src.routes

Semua route didaftarkan lewat register_all(app) dari src/routes/__init__.py.
"""

from __future__ import annotations

import hmac
import os

from flask import Flask, abort, request
from flask_cors import CORS

from .context import *  # noqa: F401,F403 - runtime bersama
from . import register_all

app = Flask(__name__)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("AI_CORE_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
# An empty allowlist means same-service/server-to-server use only. Never use
# wildcard CORS with credentials or user data.
if _allowed_origins:
    CORS(app, origins=_allowed_origins, supports_credentials=False)


@app.before_request
def require_internal_api_key():
    """Protect AI Core when deployed behind a network boundary.

    Local development remains convenient when no key is configured. Production
    must set AI_CORE_API_KEY; the key is never shipped to mobile clients.
    """
    if request.method == "OPTIONS" or request.path in {"/", "/health"}:
        return None

    expected = os.environ.get("AI_CORE_API_KEY", "")
    production = os.environ.get("FLASK_ENV", "").lower() == "production" or os.environ.get("APP_ENV", "").lower() == "production"
    if production and not expected:
        abort(503)
    if expected and not hmac.compare_digest(
        expected, request.headers.get("X-AI-Core-Key", "")
    ):
        abort(401)

    return None


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

register_all(app)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print("=" * 55)
    print("  Wedding CBIR API Server v2.0 (routes)")
    print("=" * 55)
    print(f"  Host      : {host}:{port}")
    print("  Ready!")

    get_extractor_instance()
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
