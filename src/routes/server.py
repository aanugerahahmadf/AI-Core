# -*- coding: utf-8 -*-
"""Server entry point untuk package routes.

Jalankan:
    python -m src.routes

Semua route didaftarkan lewat register_all(app) dari src/routes/__init__.py.
"""

from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from .context import *  # noqa: F401,F403 - runtime bersama
from . import register_all

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

register_all(app)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
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