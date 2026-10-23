# -*- coding: utf-8 -*-
"""POST /api/search - pencarian citra CBIR (CBIRService::searchByImage())."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


# ---------------------------------------------------------------------------
# Routes — Search  (dipanggil CBIRService::searchByImage)
# ---------------------------------------------------------------------------
def register(app):
    @app.route("/api/search", methods=["POST"])
    def search_similar():
        """
        Cari gambar mirip.
        Dipanggil oleh Laravel CBIRService::searchByImage().

        Request : multipart/form-data  key='file'  (+ optional top_k)
                  ATAU JSON {"image": "<base64>", "top_k": 10}
        Response: {"success": true, "results": [...], "query_time_seconds": 0.5}
        """
        t0 = time.perf_counter()

        try:
            # --- top_k ---
            top_k_raw = request.form.get("top_k")
            if top_k_raw is None and request.is_json:
                top_k_raw = (request.json or {}).get("top_k")
            top_k = int(top_k_raw) if top_k_raw is not None else 10
            top_k = max(1, min(top_k, 50))

            # --- Terima gambar ---
            if "file" in request.files:
                f = request.files["file"]
                if not f or not f.filename:
                    return jsonify({"error": "No file selected"}), 400
                if not allowed_file(f.filename) and not is_video_file(f.filename):
                    return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, bmp, webp, mp4, mov, m4v, webm, 3gp"}), 400
                filepath = save_upload(f, f.filename)

            elif request.is_json and request.json and "image" in request.json:
                raw = request.json["image"]
                if "," in raw:
                    raw = raw.split(",", 1)[1]
                img_bytes = base64.b64decode(raw)
                img       = Image.open(io.BytesIO(img_bytes))
                fname     = f"cbir-temp-{int(time.time())}.jpg"
                filepath  = save_upload(img_bytes, fname)
                img.save(filepath)

            else:
                return jsonify({"error": "No image provided"}), 400

            # --- Video: ekstrak frame terbaik agar bisa dianalisis CBIR ---
            filepath, is_frame = ensure_image_frame(filepath)

            # --- Ekstrak fitur & hitung skor ---
            import numpy as np
            extractor  = get_extractor_instance()
            finder     = get_finder_instance()
            query_feat = extractor.extract(filepath)
            # pHash query — dipakai untuk short-circuit "gambar sama persis" agar = 100%
            query_phash = compute_phash(filepath)

            db     = load_feature_database(FEATURE_DB)
            images = db.get("images", [])
            is_sim = finder.is_similarity()

            scores = []
            exact_hits = 0
            for entry in images:
                feat_dict = entry.get("features", {})
                feat_list = (
                    feat_dict.get(EXTRACT_METHOD)
                    or feat_dict.get("combined")
                    or feat_dict.get("combined_features")
                    or feat_dict.get("deep_features")
                )
                if feat_list is None:
                    continue
                candidate = np.array(feat_list, dtype=np.float32)
                if candidate.shape != query_feat.shape:
                    continue
                raw = finder.compute(query_feat, candidate)
                # Duplikat persis: pHash identik → skor maksimal (100%)
                stored_phash = entry.get("phash", "")
                if query_phash and stored_phash and query_phash == stored_phash:
                    raw = 1.0 if is_sim else 0.0
                    exact_hits += 1
                scores.append((raw, entry))

            if exact_hits:
                print(f"[CBIR] Exact duplicate(s) detected via pHash: {exact_hits}")

            scores.sort(key=lambda x: x[0], reverse=is_sim)

            # --- Dedup per (type, owner) + urutkan ulang (post-processing) ---
            # Skor asli (incl. short-circuit pHash = 100%) dipertahankan, hanya
            # memastikan satu hasil per item dan urutan menurun untuk similarity.
            should_rerank = (request.form.get("rerank", "true").lower() == "true" or
                             (request.is_json and (request.json or {}).get("rerank", True)))
            if should_rerank and len(scores) > 1:
                try:
                    deduped: dict[str, tuple[float, dict]] = {}
                    for s, entry in scores:
                        key = "{}_{}".format(
                            normalize_type(entry.get("metadata", {}).get("type")),
                            entry.get("metadata", {}).get("owner_id"),
                        )
                        if key not in deduped:
                            deduped[key] = (s, entry)
                    scores = list(deduped.values())
                    scores.sort(key=lambda x: x[0], reverse=is_sim)
                except Exception:
                    pass

            # --- Format hasil (kompatibel dengan CBIRService.php) ---
            results = []
            for raw_score, entry in scores[:top_k]:
                meta      = entry.get("metadata", {})
                raw_score = float(raw_score)

                if is_sim:
                    similarity = round(max(0.0, raw_score * 100.0), 2)
                    score_01   = round(raw_score, 6)
                else:
                    max_dist   = 25.0
                    linear     = max(0.0, 100.0 - (raw_score / max_dist * 100.0))
                    similarity = round((linear / 100.0) ** 2 * 100.0 if linear >= 15.0 else 0.0, 2)
                    score_01   = round(similarity / 100.0, 6)

                item_type = normalize_type(meta.get("type"))

                results.append({
                    # Fields yang dibaca CBIRService.php
                    "owner_id"      : meta.get("owner_id"),
                    "type"          : item_type,
                    "score"         : score_01,
                    "similarity"    : similarity,
                    "image_url"     : meta.get("image_url", ""),
                    # Fields tambahan
                    "id"            : entry.get("id"),
                    "name"          : meta.get("name", ""),
                    "category"      : meta.get("category", ""),
                    "distance"      : round(raw_score, 4),
                    "image_path"    : meta.get("image_path", ""),
                    "price"         : meta.get("price", 0),
                    "discount_price": meta.get("discount_price", 0),
                    "description"   : meta.get("description", ""),
                })

            elapsed = round(time.perf_counter() - t0, 3)

            return jsonify({
                "success"             : True,
                "results"             : results,
                "total_results"       : len(results),
                # Laravel CBIRService.php membaca "query_time_seconds"
                "query_time_seconds"  : elapsed,
                # Alias untuk kompatibilitas
                "query_time_s"        : elapsed,
                "video_frame_extracted": is_frame,
                "method"              : EXTRACT_METHOD,
                "metric"              : FIND_METRIC,
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
