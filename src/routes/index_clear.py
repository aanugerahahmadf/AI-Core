# -*- coding: utf-8 -*-
"""POST /api/index/clear - kosongkan seluruh index (admin panel)."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _reload_engine,
)  # helper privat: tidak ikut star-import


def register(app):
    @app.route("/api/index/clear", methods=["POST"])
    def clear_index():
        try:
            save_feature_database({"images": []}, FEATURE_DB)
            _reload_engine()
            return jsonify({"success": True, "message": "Index cleared"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
