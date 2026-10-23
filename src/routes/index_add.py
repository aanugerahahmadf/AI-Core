# -*- coding: utf-8 -*-
"""POST /api/index/add - tambah satu gambar ke index (CBIRService::indexMedia())."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _reload_engine,
)  # helper privat: tidak ikut star-import


# ---------------------------------------------------------------------------
# Routes — Index Management  (dipanggil CBIRService.php)
# ---------------------------------------------------------------------------
def register(app):
    @app.route("/api/index/add", methods=["POST"])
    def add_to_index():
        """
        Tambah satu gambar ke index.
        Dipanggil oleh Laravel CBIRService::indexMedia().

        Request JSON: {"image_path": "...", "metadata": {...}}
        """
        try:
            import numpy as np

            data       = request.json or {}
            image_path = data.get("image_path", "").strip()
            metadata   = data.get("metadata", {})

            if not image_path:
                return jsonify({"success": False, "message": "Missing image_path"}), 400

            image_path = resolve_portable_path(image_path)
            if not os.path.exists(image_path):
                return jsonify({"success": False, "message": f"File not found: {image_path}"}), 404

            # Validasi type
            metadata["type"] = normalize_type(metadata.get("type"))

            if "image_path" in metadata:
                metadata["image_path"] = image_path

            extractor = get_extractor_instance()
            feat      = extractor.extract(image_path)
            phash_hex = compute_phash(image_path)

            db = load_feature_database(FEATURE_DB)

            # Dedup per FILE, bukan per (type, owner_id).
            #
            # Sebuah produk/paket bisa punya banyak gambar (galeri). Kalau entri lain
            # milik (type, owner_id) yang sama ikut dihapus, setiap panggilan
            # `php artisan ai:sync` (yang looping SEMUA media per produk) akan
            # menyisakan hanya gambar terakhir per produk dan membuang sisanya —
            # indeks 400 entri bisa menyusut jadi 100 tanpa ada error sama sekali.
            #
            # Menampilkan "1 hasil per produk" memang tugas layer search
            # (`search_similar` dedup by type_owner_id, ambil skor terbaik), jadi di
            # sini cukup.replace entri ketika file yang sama di-index ulang.
            def _norm(p):
                return os.path.normcase(os.path.abspath(resolve_portable_path(p)))

            target_key = _norm(image_path)
            db["images"] = [img for img in db["images"] if _norm(img.get("path", "")) != target_key]

            existing_ids = [int(img.get("id") or 0) for img in db["images"]]
            new_id = (max(existing_ids) + 1) if existing_ids else 1

            # Simpan hanya key kanonik — alias fitur di-expand otomatis saat load,
            # sehingga tidak perlu diduplikasi di disk.
            if hasattr(extractor, "_deep"):
                deep_feat    = extractor._deep.extract(image_path)
                color_feat   = extractor._color.extract(image_path)
                texture_feat = extractor._texture.extract(image_path)
            else:
                deep_feat    = feat if EXTRACT_METHOD in ("deep", "resnet50") else np.array([])
                color_feat   = feat if EXTRACT_METHOD in ("color", "color_histogram") else np.array([])
                texture_feat = feat if EXTRACT_METHOD == "lbp" else np.array([])

            def _vec(arr):
                # WAJIB float64: float32 tidak bisa mewakili hasil round 6 desimal.
                return np.round(np.asarray(arr, dtype=np.float64), FLOAT_PRECISION).tolist()

            entry = {
                "id"      : new_id,
                "path"    : image_path,
                "phash"   : phash_hex,
                "metadata": metadata,
                "features": {
                    "combined" : _vec(feat),
                    "deep"     : _vec(deep_feat),
                    "color"    : _vec(color_feat),
                    "lbp"      : _vec(texture_feat),
                    "method"   : EXTRACT_METHOD,
                },
            }
            db["images"].append(entry)
            save_feature_database(db, FEATURE_DB)
            _reload_engine()

            return jsonify({
                "success" : True,
                "message" : "Image added to index",
                "entry_id": new_id,
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
