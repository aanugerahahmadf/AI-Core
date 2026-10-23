# -*- coding: utf-8 -*-
"""GET /health - health check AI Core."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


def register(app):
    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status" : "healthy",
            "service": "Wedding CBIR API",
            "version": "2.0.0",
            "method" : EXTRACT_METHOD,
            "metric" : FIND_METRIC,
            "capabilities": {
                "cbir"           : True,
                "face_verify"    : True,
                "ktp_verify"     : True,
                "ocr"            : True,
                "proof_verify"   : True,
                "video_frames"   : True,
                "face_threshold" : FACE_VERIFY_THRESHOLD,
            },
        })
