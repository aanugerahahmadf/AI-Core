# -*- coding: utf-8 -*-
"""GET /status - statistik database fitur (CBIRController::getStats())."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


def status_stats():
    # Implementasi module-level agar saling terhubung:
    # routes/index_stats.py mendelegasikan ke fungsi ini (endpoint alias).
    try:
        db     = load_feature_database(FEATURE_DB)
        images = db.get("images", [])
        cats   = {}
        types  = {}
        for img in images:
            m = img.get("metadata", {})
            cats[m.get("category", "unknown")]  = cats.get(m.get("category", "unknown"), 0) + 1
            types[m.get("type", "unknown")]     = types.get(m.get("type", "unknown"), 0) + 1
        return jsonify({
            "status"        : "healthy",
            "total_catalogs": len(images),
            "categories"    : cats,
            "types"         : types,
            "database_path" : FEATURE_DB,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


def register(app):
    @app.route("/status", methods=["GET"])
    def status():
        return status_stats()

