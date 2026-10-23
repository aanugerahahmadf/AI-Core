# -*- coding: utf-8 -*-
"""POST /api/index/remove - hapus gambar dari index (CBIRService::removeFromIndex())."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _reload_engine,
)  # helper privat: tidak ikut star-import


def register(app):
    @app.route("/api/index/remove", methods=["POST"])
    def remove_from_index():
        """
        Hapus gambar dari index berdasarkan metadata_id atau owner_id+type.
        Dipanggil oleh Laravel CBIRService::removeFromIndex().

        Request JSON: {"metadata_id": 5}
                      ATAU {"owner_id": 5, "type": "catalog"}
        """
        try:
            data        = request.json or {}
            metadata_id = data.get("metadata_id")
            owner_id    = data.get("owner_id")
            item_type   = normalize_type(data.get("type"))

            db     = load_feature_database(FEATURE_DB)
            before = len(db["images"])

            if metadata_id is not None:
                # Hapus berdasarkan metadata.id (owner_id di metadata)
                db["images"] = [
                    img for img in db["images"]
                    if img.get("metadata", {}).get("owner_id") != int(metadata_id)
                ]
            elif owner_id is not None:
                # Hapus berdasarkan owner_id + type
                db["images"] = [
                    img for img in db["images"]
                    if not (
                        img.get("metadata", {}).get("owner_id") == int(owner_id)
                        and img.get("metadata", {}).get("type") == item_type
                    )
                ]
            else:
                return jsonify({"success": False, "message": "Provide metadata_id or owner_id"}), 400

            # Re-number IDs
            for i, img in enumerate(db["images"], start=1):
                img["id"] = i

            removed = before - len(db["images"])
            save_feature_database(db, FEATURE_DB)
            _reload_engine()

            return jsonify({
                "success": True,
                "message": f"{removed} image(s) removed from index",
                "removed": removed,
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
