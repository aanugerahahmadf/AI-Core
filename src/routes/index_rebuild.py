# -*- coding: utf-8 -*-
"""POST /api/index/rebuild-from-dataset - rebuild index dari dataset.csv."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _reload_engine,
)  # helper privat: tidak ikut star-import


def register(app):
    @app.route("/api/index/rebuild-from-dataset", methods=["POST"])
    def rebuild_from_dataset():
        """
        Rebuild index dari dataset.csv Laravel.
        Dipanggil oleh SyncCbirCsv.php dan SyncAICoreCommand.php.

        Request JSON (opsional): {"csv_path": "...", "app_url": "..."}
        """
        t0 = time.perf_counter()
        try:
            data     = request.json or {}
            csv_path = data.get("csv_path") or os.path.join(DATA_DIR, "dataset.csv")
            app_url  = data.get("app_url", LARAVEL_URL)

            if not os.path.exists(csv_path):
                return jsonify({
                    "success": False,
                    "message": f"Dataset CSV tidak ditemukan: {csv_path}. Jalankan php artisan cbir:sync.",
                }), 404

            from src.features.build_features import build_features
            try:
                from ai_sync import sync_features_to_csv
            except Exception:
                sync_features_to_csv = None

            # Build features → update metadata.json
            result = build_features(method="combined", csv_path=csv_path, app_url=app_url)

            # Sync fitur ke dataset.csv (angka muncul di CSV/Excel)
            if sync_features_to_csv is not None:
                try:
                    sync_features_to_csv(csv_path=csv_path, metadata_path=FEATURE_DB)
                except Exception:
                    pass

            # Reload engine
            _reload_engine()

            combined = result.get("combined", {})
            cats     = {}
            types    = {}
            db       = load_feature_database(FEATURE_DB)
            for img in db.get("images", []):
                m = img.get("metadata", {})
                cats[m.get("category", "unknown")]  = cats.get(m.get("category", "unknown"), 0) + 1
                types[m.get("type", "unknown")]     = types.get(m.get("type", "unknown"), 0) + 1

            return jsonify({
                "success"        : True,
                "message"        : f"Rebuild selesai: {combined.get('total', 0)} gambar terindeks.",
                "total"          : combined.get("total", 0),
                "skipped"        : combined.get("skipped", 0),
                "errors"         : 0,
                "categories"     : cats,
                "types"          : types,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
