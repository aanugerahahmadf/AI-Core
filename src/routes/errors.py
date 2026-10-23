# -*- coding: utf-8 -*-
"""Handler error global (413 / 404 / 500)."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
def register(app):
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Max 16MB"}), 413


    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404


    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500
