# -*- coding: utf-8 -*-
"""
src/models/make_prediction.py — CBIR Inference.

Usage:
    python -m src.models.make_prediction --query path/to/image.jpg
    python -m src.models.make_prediction --query path/to/image.jpg --top-k 5 --metric euclidean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.addons.data import load_feature_database
from src.addons.extraction.extractor import extractors, get_extractor
from src.addons.finder import finders, get_finder

DATA_DIR = os.path.join(ROOT, "data")
METADATA = os.path.join(DATA_DIR, "metadata.json")


def search(
    query_path: str,
    top_k: int = 10,
    metric: str = "euclidean",
    feature_db_path: str | None = None,
    method: str = "combined",
) -> tuple[list[dict], float]:
    """
    Cari gambar paling mirip dengan query dari feature database.

    Returns: (results, elapsed_seconds)
    """
    db_path = feature_db_path or METADATA

    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Query image tidak ditemukan: {query_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"metadata.json tidak ditemukan: {db_path}\n"
            "Jalankan: python rebuild_index.py"
        )

    db   = load_feature_database(db_path)
    imgs = db.get("images", [])
    if not imgs:
        raise ValueError("Index kosong. Jalankan: python rebuild_index.py")

    extractor = get_extractor(method)
    finder    = get_finder(metric)

    t0        = time.perf_counter()
    query_vec = extractor.extract(query_path)

    scores = []
    for img in imgs:
        feat_dict = img.get("features", {})
        feat_list = (
            feat_dict.get(method)
            or feat_dict.get("combined")
            or feat_dict.get("combined_features")
            or feat_dict.get("deep_features")
        )
        if feat_list is None:
            continue

        db_vec = np.array(feat_list, dtype=np.float32)
        if db_vec.shape != query_vec.shape:
            continue

        score = finder.compute(query_vec, db_vec)
        scores.append((score, img))

    scores.sort(key=lambda x: x[0], reverse=finder.is_similarity())
    elapsed = round(time.perf_counter() - t0, 3)

    results = []
    for score, img in scores[:top_k]:
        meta = img.get("metadata", {})
        if finder.is_similarity():
            similarity = round(max(0.0, float(score) * 100.0), 2)
        else:
            max_dist   = 25.0
            linear     = max(0.0, 100.0 - (float(score) / max_dist * 100.0))
            similarity = round((linear / 100.0) ** 2 * 100.0 if linear >= 15.0 else 0.0, 2)

        results.append({
            "id"            : img.get("id"),
            "type"          : meta.get("type", "product"),
            "owner_id"      : meta.get("owner_id"),
            "name"          : meta.get("name", ""),
            "category"      : meta.get("category", ""),
            "price"         : meta.get("price", 0),
            "discount_price": meta.get("discount_price", 0),
            "organizer"     : meta.get("organizer", ""),
            "similarity"    : similarity,
            "score"         : round(float(score), 6),
            "image_url"     : meta.get("image_url", ""),
            "image_path"    : img.get("path", ""),
        })

    return results, elapsed


def inference(
    input_path: str,
    feature_path: str,
    output_path: str,
    method: str = "combined",
    metric: str = "euclidean",
    top_k: int = 5,
) -> None:
    """Batch inference: cari kemiripan untuk semua gambar di input_path."""
    feature_db = os.path.join(feature_path, "metadata.json")
    os.makedirs(output_path, exist_ok=True)

    image_exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    query_images = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    if not query_images:
        print(f"[WARN] Tidak ada gambar di: {input_path}")
        return

    print(f"[INFO] {len(query_images)} gambar query | method={method} metric={metric} top_k={top_k}")

    all_results = []
    for i, qpath in enumerate(query_images, 1):
        try:
            results, elapsed = search(
                query_path=qpath,
                top_k=top_k,
                metric=metric,
                feature_db_path=feature_db,
                method=method,
            )
            all_results.append({"query": qpath, "results": results, "elapsed": elapsed})
            print(f"  [{i:3d}/{len(query_images)}] {os.path.basename(qpath)} → {len(results)} results ({elapsed}s)")
        except Exception as e:
            print(f"  [{i:3d}/{len(query_images)}] ERROR {os.path.basename(qpath)}: {e}")

    out_file = os.path.join(output_path, f"{method}_{metric}_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Hasil disimpan ke: {out_file}")


def predict(query_path: str, top_k: int = 10, metric: str = "euclidean"):
    """Alias untuk search() — backward compatibility."""
    return search(query_path=query_path, top_k=top_k, metric=metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBIR Inference — cari gambar mirip")
    parser.add_argument("--query",  required=True)
    parser.add_argument("--top-k",  type=int, default=10)
    parser.add_argument("--metric", default="euclidean", choices=list(finders.keys()))
    parser.add_argument("--method", default="combined",  choices=list(extractors.keys()))
    parser.add_argument("--db",     default=METADATA)
    args = parser.parse_args()

    results, elapsed = search(
        query_path=args.query,
        top_k=args.top_k,
        metric=args.metric,
        feature_db_path=args.db,
        method=args.method,
    )

    print(f"\nTop-{args.top_k} | method={args.method} metric={args.metric} waktu={elapsed}s")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['type'].upper():8s}] {r['name']} — {r['similarity']}%")
        print(f"       {r['image_path']}")
