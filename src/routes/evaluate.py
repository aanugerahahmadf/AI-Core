# -*- coding: utf-8 -*-
"""GET /api/evaluate - evaluasi kualitatif: MAP, MRR, Precision@K."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama


# ---------------------------------------------------------------------------
# Routes — Evaluation  (kualitatif: MAP, MRR, Precision@K, First Rank)
# ---------------------------------------------------------------------------
def register(app):
    @app.route("/api/evaluate", methods=["GET"])
    def evaluate():
        """
        Evaluasi kualitatif sistem CBIR.
        Menggunakan setiap gambar sebagai query dan membandingkan dengan ground truth
        (kategori yang sama = relevan).
        Response: { "success": true, "metrics": { "map": ..., "mrr": ..., ... } }
        """
        from src.evaluation.run_evaluation import run_and_save
        t0 = time.perf_counter()

        try:
            from src.addons.data import load_feature_database
            db     = load_feature_database(FEATURE_DB)
            images = db.get("images", [])

            if len(images) < 2:
                return jsonify({
                    "success": False,
                    "message": "Minimal 2 gambar diperlukan untuk evaluasi",
                    "metrics": {},
                })

            # Evaluasi + persisten ke data/evaluation/
            payload   = run_and_save(db_path=FEATURE_DB, method=EXTRACT_METHOD, metric=FIND_METRIC)
            metrics   = payload["metrics"]
            n_queries = payload["n_queries"]
            elapsed   = round(time.perf_counter() - t0, 4)

            return jsonify({
                "success"                : True,
                "metrics"                : metrics,
                "precision_at_3"         : metrics.get("precision_at_3", 0.0),
                "n_images"               : len(images),
                "n_queries"              : n_queries,
                "evaluation_time_seconds": elapsed,
                "saved_paths"            : payload.get("saved_paths", []),
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
