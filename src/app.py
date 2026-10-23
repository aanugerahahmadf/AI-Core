# -*- coding: utf-8 -*-
"""
src/app.py �?" kompatibilitas entry point.

Server utama kini ada di package routes (src/routes/server.py dan
src/routes/__main__.py). Modul ini hanya re-export namespace agar pemakai
lama (`import src.app`, tests) tetap berfungsi.
"""

from __future__ import annotations

from .routes.context import *  # noqa: F401,F403 - kompatibilitas namespace
from .routes.server import app  # noqa: F401

from .routes.context import (
    _extract_ktp_number,  # noqa: F401
    _extract_amounts,     # noqa: F401
    _normalize_number,    # noqa: F401
)


if __name__ == "__main__":
    from .routes.server import main as _main
    _main()