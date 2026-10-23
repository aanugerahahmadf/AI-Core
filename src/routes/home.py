# -*- coding: utf-8 -*-
"""GET / - informasi layanan & daftar endpoint."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


# ---------------------------------------------------------------------------
# Routes — Health & Status
# ---------------------------------------------------------------------------
def register(app):
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "service": "Wedding CBIR API",
            "version": "2.0.0",
            "status" : "running",
            "endpoints": {
                "GET  /health"                        : "Health check",
                "GET  /status"                        : "Database statistics",
                "GET  /api/index/stats"               : "Database statistics (alias)",
                "POST /api/search"                    : "Search similar images",
                "POST /api/index/add"                 : "Add image to index",
                "POST /api/index/remove"              : "Remove image from index",
                "POST /api/index/rebuild-from-dataset": "Rebuild index dari dataset.csv",
                "POST /api/index/clear"               : "Clear index",
                "POST /api/features/extract"          : "Extract features from image",
                "POST /api/sync"                      : "Trigger ai_sync.py (php artisan ai:sync)",
                "POST /api/arithmetic"                : "Aritmetika Citra: +, -, ×, ÷ antar gambar (AI fusion)",
                "GET  /api/arithmetic/ops"            : "Daftar operasi aritmetika yang tersedia",
                "POST /api/face/verify"               : "Verifikasi wajah (selfie vs KTP) — Computer Vision",
                "POST /api/ktp/verify"                : "Validasi dokumen KTP — Computer Vision",
                "POST /api/proof/verify"               : "Validasi bukti pembayaran — OCR (nominal)",
            },
        })
