# -*- coding: utf-8 -*-
"""POST /api/arithmetic - aritmetika citra +, -, x, / lalu pencarian."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def register(app):
    @app.route("/api/arithmetic", methods=["POST"])
    def arithmetic_search_endpoint():
        """
        Aritmetika Citra: gabungkan 2+ gambar dengan operasi +, -, ×, ÷
        lalu cari hasilnya di database.

        Request:
          JSON: { "images": ["/path/gbr1.jpg", "/path/gbr2.jpg"], "operation": "...", ... }
          ATAU multipart: files image_1, image_2 + form fields operation, ...

        Response: { "success": true, "results": [...], "operation": {...}, ... }
        """
        t0 = time.perf_counter()
        try:
            images = []

            # --- Coba baca dari upload files ---
            f1 = request.files.get("image_1")
            f2 = request.files.get("image_2")
            if f1 and f1.filename:
                images.append(save_upload(f1, f1.filename))
            if f2 and f2.filename:
                images.append(save_upload(f2, f2.filename))

            # --- Kalau tidak ada file, baca dari JSON ---
            if not images:
                data = request.get_json(force=True) if request.is_json else {}
                raw = data.get("images", []) if isinstance(data, dict) else []
                if not isinstance(raw, list) or len(raw) < 1:
                    return jsonify({"error": "Upload 2 file (image_1, image_2) atau kirim JSON 'images'"}), 400
                # Resolve portable path bila perlu
                for p in raw:
                    resolved = resolve_portable_path(p) if not os.path.exists(p) else p
                    images.append(resolved)

            if len(images) < 1:
                return jsonify({"error": "Minimal 1 gambar diperlukan"}), 400

            # --- Baca parameter ---
            json_data = request.get_json(silent=True) if request.is_json else {}
            if not isinstance(json_data, dict):
                json_data = {}
            operation = request.form.get("operation") or json_data.get("operation", "average")
            method = request.form.get("method") or json_data.get("method", EXTRACT_METHOD)
            metric = request.form.get("metric") or json_data.get("metric", FIND_METRIC)
            weights = json_data.get("weights")

            result = arithmetic_search(
                image_paths=images,
                operation=operation,
                method=method,
                metric=metric,
                db_path=FEATURE_DB,
                weights=weights,
            )
            result["server_time_seconds"] = round(time.perf_counter() - t0, 4)
            return jsonify(result)

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": f"Arithmetic search failed: {str(e)}"}), 500
