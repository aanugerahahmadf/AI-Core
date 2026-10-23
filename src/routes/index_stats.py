# -*- coding: utf-8 -*-
"""GET /api/index/stats — alias statistik database (CBIRController::getStats()).

Saling terhubung: delegasi ke handler `status` di routes/status.py."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .status import status_stats as _status_stats  # hub antar-file route


def register(app):
    @app.route("/api/index/stats", methods=["GET"])
    def get_index_stats():
        return _status_stats()
