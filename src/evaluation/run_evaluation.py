# -*- coding: utf-8 -*-
"""
run_evaluation.py — Jalankan evaluasi kualitatif CBIR dan simpan ke data/evaluation/.

Berfungsi untuk endpoint /api/evaluate (Flask) maupun CLI `python run_evaluation.py`,
sehingga data/evaluation/ selalu terisi hasil evaluasi (summary JSON + per-query CSV).

Output:
  data/evaluation/evaluation_summary.json
  data/evaluation/evaluation_results.csv
  data/evaluation/evaluation_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.addons.metrics import (
    evaluation_report,
    average_precision,
    mean_reciprocal_rank,
)
from src.addons.data import load_feature_database

EVAL_DIR = os.path.join(ROOT, "data", "evaluation")


def _candidate_features(entry: dict) -> Any:
    """Ambil vektor fitur kandidat (combined dulu, lalu fallback)."""
    feat_dict = entry.get("features", {}) or {}
    for key in ("combined", "combined_features", "deep_features"):
        if feat_dict.get(key):
            return feat_dict.get(key)
    return feat_dict.get("features") or None


def evaluate_database(
    db_path: str,
    method: str = "combined",
    metric: str = "cosine",
    top_k: int = 3,
) -> dict:
    """
    Evaluasi semua gambar: gunakan tiap gambar sebagai query, ground truth =
    kategori yang sama. Mengembalikan metrics + per-query rows.
    """
    from src.addons.finder import get_finder
    from src.addons.extraction.extractor import get_extractor
    from src.addons.metrics import first_rank_accuracy

    db        = load_feature_database(db_path)
    finder    = get_finder(metric)
    is_sim    = finder.is_similarity()
    images    = db.get("images", [])
    queries   = []
    rows      = []

    t0 = time.perf_counter()
    extractor = get_extractor(method)

    for entry in images:
        q_path = os.path.normpath(entry.get("path", ""))
        if not q_path or not os.path.exists(q_path):
            continue
        q_meta = entry.get("metadata", {})
        q_cat  = (q_meta.get("category") or "").strip().lower()
        q_id   = entry.get("id")
        q_name = q_meta.get("name", "")

        q_feat = np.asarray(extractor.extract(q_path), dtype=np.float32)

        scores = []
        for cand in images:
            if cand.get("id") == q_id:
                continue
            feat_list = _candidate_features(cand)
            if feat_list is None:
                continue
            cand_vec = np.asarray(feat_list, dtype=np.float32)
            if cand_vec.shape != q_feat.shape:
                continue
            scores.append((finder.compute(q_feat, cand_vec), cand))

        scores.sort(key=lambda x: x[0], reverse=is_sim)

        retrieved = []
        relevant  = set()
        for s, c in scores:
            c_meta = c.get("metadata", {})
            retrieved.append(c.get("id"))
            if q_cat and (c_meta.get("category") or "").strip().lower() == q_cat:
                relevant.add(c.get("id"))

        if not relevant:
            continue

        ap = average_precision(retrieved, relevant)
        rr = mean_reciprocal_rank([(retrieved, relevant)])
        top = scores[:top_k]
        top1_correct = bool(retrieved and retrieved[0] in relevant)

        rows.append({
            "query_id"     : q_id,
            "query_name"   : q_name,
            "category"     : q_cat,
            "top1_correct" : top1_correct,
            "ap"           : round(ap, 6),
            "rr"           : round(rr, 6),
            "top1_name"    : top[0][1].get("metadata", {}).get("name", "") if top else "",
            "top1_category": top[0][1].get("metadata", {}).get("category", "") if top else "",
            "top1_score"   : round(float(top[0][0]), 6) if top else "",
            "top2_name"    : top[1][1].get("metadata", {}).get("name", "") if len(top) > 1 else "",
            "top2_category": top[1][1].get("metadata", {}).get("category", "") if len(top) > 1 else "",
            "top2_score"   : round(float(top[1][0]), 6) if len(top) > 1 else "",
            "top3_name"    : top[2][1].get("metadata", {}).get("name", "") if len(top) > 2 else "",
            "top3_category": top[2][1].get("metadata", {}).get("category", "") if len(top) > 2 else "",
            "top3_score"   : round(float(top[2][0]), 6) if len(top) > 2 else "",
        })

        queries.append((retrieved, relevant))

    elapsed = round(time.perf_counter() - t0, 4)
    metrics = evaluation_report(queries, avg_query_time=elapsed)
    metrics["metric"] = metric
    metrics["method"] = method

    # Precision@3
    p3_sum = 0.0
    for ret, rel in queries:
        top3 = ret[:3]
        if top3:
            hits = sum(1 for rid in top3 if rid in rel)
            p3_sum += hits / min(3, len(top3))
    metrics["precision_at_3"] = round(p3_sum / len(queries), 4) if queries else 0.0

    return {"metrics": metrics, "rows": rows, "n_queries": len(queries)}


def save_evaluation(payload: dict, method: str, metric: str):
    """Simpan hasil evaluasi ke data/evaluation/ (summary JSON + CSV + rows JSON)."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    summary_path = os.path.join(EVAL_DIR, "evaluation_summary.json")
    rows_json    = os.path.join(EVAL_DIR, "evaluation_results.json")
    csv_path     = os.path.join(EVAL_DIR, "evaluation_results.csv")

    summary = {
        "metric"      : metric,
        "method"      : method,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics"     : payload["metrics"],
        "n_queries"   : payload["n_queries"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(rows_json, "w", encoding="utf-8") as f:
        json.dump(payload["rows"], f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload["rows"][0].keys()) if payload["rows"] else [])
        if payload["rows"]:
            writer.writeheader()
            writer.writerows(payload["rows"])

    return summary_path, csv_path


def run_and_save(db_path: str, method: str = "combined", metric: str = "cosine") -> dict:
    """Jalankan evaluasi, simpan ke data/evaluation/, kembalikan payload lengkap."""
    payload = evaluate_database(db_path=db_path, method=method, metric=metric)
    save_evaluation(payload, method, metric)
    payload["saved_paths"] = [
        os.path.join(EVAL_DIR, "evaluation_summary.json"),
        os.path.join(EVAL_DIR, "evaluation_results.csv"),
    ]
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi CBIR dan simpan ke data/evaluation/")
    parser.add_argument(
        "--db", default=os.path.join(ROOT, "data", "metadata.json"),
        help="Path metadata.json (default: data/metadata.json)",
    )
    parser.add_argument("--method", default="combined", help="Metode ekstraksi (default: combined)")
    parser.add_argument("--metric", default="cosine", help="Metrik jarak (default: cosine)")
    args = parser.parse_args()

    result = run_and_save(db_path=args.db, method=args.method, metric=args.metric)

    m = result["metrics"]
    print("=" * 50)
    print("EVALUASI SELESAI")
    print(f"  MAP : {m.get('map')} | MRR : {m.get('mrr')} | FirstRank: {m.get('first_rank_accuracy')} | P@3: {m.get('precision_at_3')}")
    print(f"  N queries : {result['n_queries']}")
    print(f"  Hasil tersimpan di: {EVAL_DIR}")
    print("=" * 50)