# -*- coding: utf-8 -*-
"""POST /api/sync - trigger ai_sync.py dari Laravel (php artisan ai:sync)."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _reload_engine,
)  # helper privat: tidak ikut star-import


# ---------------------------------------------------------------------------
# Routes — AI Sync  (dipanggil php artisan ai:sync)
# ---------------------------------------------------------------------------
def register(app):
    @app.route("/api/sync", methods=["POST"])
    def trigger_sync():
        """
        Trigger ai_sync.py dari Laravel (php artisan ai:sync).
        Sama dengan rebuild-from-dataset tapi juga sync ke CSV.

        Request JSON (opsional): {"csv_path": "...", "app_url": "..."}
        """
        t0 = time.perf_counter()
        try:
            from ai_sync import main as ai_sync_main

            data     = request.json or {}
            csv_path = data.get("csv_path") or os.path.join(DATA_DIR, "dataset.csv")
            app_url  = data.get("app_url", LARAVEL_URL)

            exit_code = ai_sync_main(csv_path=csv_path, app_url=app_url)
            _reload_engine()

            if exit_code == 0:
                db    = load_feature_database(FEATURE_DB)
                total = len(db.get("images", []))
                return jsonify({
                    "success"        : True,
                    "message"        : f"AI Sync selesai: {total} gambar terindeks.",
                    "total"          : total,
                    "elapsed_seconds": round(time.perf_counter() - t0, 2),
                })

            return jsonify({"success": False, "message": "AI Sync gagal. Cek log server."}), 500

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
