# -*- coding: utf-8 -*-
"""GET /api/arithmetic/ops - daftar operasi aritmetika yang tersedia."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


def register(app):
    @app.route("/api/arithmetic/ops", methods=["GET"])
    def arithmetic_ops():
        """Daftar operasi aritmetika yang tersedia."""
        return jsonify({
            "operations": {
                name: {"symbol": sym, "description": desc}
                for name, (_, sym, desc) in OPS.items()
            }
        })
