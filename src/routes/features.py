# -*- coding: utf-8 -*-
"""POST /api/features/extract - ekstrak fitur satu gambar (debugging)."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


# ---------------------------------------------------------------------------
# Routes — Feature Extraction (debugging)
# ---------------------------------------------------------------------------
def register(app):
    @app.route("/api/features/extract", methods=["POST"])
    def extract_features():
        """Extract fitur dari satu gambar (untuk debugging)."""
        try:
            if "file" not in request.files:
                return jsonify({"error": "No file provided"}), 400
            f = request.files["file"]
            if not f or not f.filename or not allowed_file(f.filename):
                return jsonify({"error": "Invalid file type"}), 400

            filepath  = save_upload(f, f.filename)
            extractor = get_extractor_instance()
            feat      = extractor.extract(filepath)

            return jsonify({
                "success"        : True,
                "method"         : EXTRACT_METHOD,
                "feature_dim"    : int(feat.shape[0]),
                "feature_preview": feat[:8].tolist(),
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
