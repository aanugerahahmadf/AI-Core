# -*- coding: utf-8 -*-
"""
ai_sync.py — Unified Synchronization & Reindexing Component for Wedding CBIR

This script consolidates all reindexing and feature sync functionality.
It is called by Laravel: php artisan ai:sync

Alur:
    1. Laravel generate data/dataset.csv
    2. Laravel panggil: python ai_sync.py
    3. Script ini:
       a. Ekstrak fitur semua gambar dari dataset.csv (mendukung combined, ultra, all)
       b. Update data/metadata.json (dipakai app.py untuk search)
       c. Sync 15 kolom fitur lengkap ke dataset.csv

Usage:
    python ai_sync.py
    python ai_sync.py --method ultra
    python ai_sync.py --csv data/dataset.csv --app-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

# Semua kolom fitur yang akan disync ke dataset.csv
# Format: (nama_kolom_csv, key_di_features_dict)
FEATURE_COLUMNS: list[tuple[str, str]] = [
    # Utama
    ("features",              "combined"),
    ("features_combined",     "combined"),
    # Deep Learning
    ("features_deep",         "deep"),
    ("features_resnet50",     "resnet50"),
    ("features_efficientnet", "efficientnet"),
    ("features_vgg16",        "vgg16"),
    # Color
    ("features_color",        "color"),
    ("features_color_hist",   "color_histogram"),
    ("features_rgb",          "rgb_histogram"),
    ("features_dominant",     "dominant_color"),
    # Texture & Shape
    ("features_lbp",          "lbp"),
    ("features_gabor",        "gabor"),
    ("features_hog",          "hog"),
    # Keypoint
    ("features_sift",         "sift"),
    ("features_akaze",        "akaze"),
]


def sync_features_to_csv(csv_path: str, metadata_path: str) -> None:
    """
    Sync SEMUA kolom fitur dari metadata.json ke dataset.csv.

    Kolom yang ditambahkan/diperbarui di CSV:
      features, features_combined, features_deep, features_resnet50,
      features_efficientnet, features_vgg16, features_color,
      features_color_hist, features_rgb, features_dominant,
      features_lbp, features_gabor, features_hog,
      features_sift, features_akaze
    """
    print(f"\n[SYNC] Membaca metadata: {metadata_path}")
    if not os.path.exists(metadata_path):
        print(f"[WARN] metadata.json tidak ditemukan di: {metadata_path}")
        return

    with open(metadata_path, encoding="utf-8") as f:
        db = json.load(f)

    # Bangun lookup: "owner_id_type" → dict semua fitur
    lookup: dict[str, dict] = {}
    for img in db.get("images", []):
        m    = img.get("metadata", {})
        feat = img.get("features", {})
        key  = str(m.get("owner_id")) + "_" + str(m.get("type", "")).lower()

        method = feat.get("method", "")

        # Pastikan combined terisi meskipun method individual
        if not feat.get("combined"):
            feat["combined"] = feat.get("combined_features", [])

        # Fallback deep untuk method resnet50/deep
        if not feat.get("deep") and method in ("deep", "resnet50"):
            feat["deep"] = feat.get("combined", [])
            feat["resnet50"] = feat.get("combined", [])

        # Fallback color untuk method color/color_histogram
        if not feat.get("color") and method in ("color", "color_histogram"):
            feat["color"] = feat.get("combined", [])
            feat["color_histogram"] = feat.get("combined", [])

        lookup[key] = feat

    # Baca dataset.csv
    print(f"[SYNC] Membaca CSV: {csv_path}")
    csv_module.field_size_limit(10000000)  # Increase field size limit for large vectors
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))

    if not rows:
        print("[SYNC] CSV kosong, dibatalkan.")
        return

    synced   = 0
    missing  = 0

    for row in rows:
        key = str(row.get("ID", "")) + "_" + str(row.get("Type", "")).lower()
        feat = lookup.get(key, {})

        if feat:
            for col_name, feat_key in FEATURE_COLUMNS:
                vec = feat.get(feat_key, [])
                row[col_name] = json.dumps(vec) if vec else ""
            synced += 1
        else:
            # Biarkan kosong jika tidak ada di metadata
            for col_name, _ in FEATURE_COLUMNS:
                row.setdefault(col_name, "")
            missing += 1

    # Pastikan semua kolom baru ada di fieldnames
    fieldnames = list(rows[0].keys())
    for col_name, _ in FEATURE_COLUMNS:
        if col_name not in fieldnames:
            fieldnames.append(col_name)

    # Tulis ulang CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SYNC] OK: {synced} baris diperbarui, {missing} baris tidak ditemukan di metadata.")
    print(f"[SYNC] Kolom yang disync ({len(FEATURE_COLUMNS)}):")
    for col_name, feat_key in FEATURE_COLUMNS:
        # Tampilkan dimensi sample dari baris pertama
        sample_row = next((r for r in rows if r.get(col_name)), None)
        if sample_row and sample_row.get(col_name):
            try:
                dim = len(json.loads(sample_row[col_name]))
                print(f"         {col_name:<25} -> {dim}-dim")
            except Exception:
                print(f"         {col_name:<25} -> (error baca dimensi)")
        else:
            print(f"         {col_name:<25} -> (kosong)")
    print(f"[SYNC] CSV disimpan: {csv_path}")


def main(csv_path: str, app_url: str, method: str = "combined") -> int:
    """
    Main entry point.
    Returns 0 on success, 1 on error.
    """
    t0 = time.perf_counter()

    print("=" * 60)
    print("AI SYNC — Wedding CBIR (Unified)")
    print("=" * 60)
    print(f"CSV     : {csv_path}")
    print(f"Method  : {method}")
    print(f"App URL : {app_url}")
    print()

    # Validasi CSV ada
    if not os.path.exists(csv_path):
        print(f"[ERROR] dataset.csv tidak ditemukan: {csv_path}")
        print("Pastikan Laravel sudah generate CSV terlebih dahulu.")
        return 1

    # Step 1: Build features → update metadata.json
    print("[STEP 1] Ekstrak fitur dari gambar...")
    from src.features.build_features import build_features
    result = build_features(
        method   = method,
        csv_path = csv_path,
        app_url  = app_url,
    )

    # Ambil metadata key pertama untuk total info
    total_indexed = 0
    if method in result:
        total_indexed = result[method].get("total", 0)
    elif result:
        # fallback jika method all atau key lain
        first_key = list(result.keys())[0]
        total_indexed = result[first_key].get("total", 0)

    if total_indexed == 0:
        print("[WARN] Tidak ada gambar yang berhasil diindeks.")

    # Step 2: Sync fitur ke dataset.csv
    print()
    print("[STEP 2] Sync fitur ke dataset.csv...")
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    sync_features_to_csv(csv_path=csv_path, metadata_path=metadata_path)

    elapsed = round(time.perf_counter() - t0, 2)

    print()
    print("=" * 60)
    print(f"AI SYNC SELESAI — {elapsed}s")
    print(f"  Terindeks : {total_indexed}")
    print(f"  metadata.json : {metadata_path}")
    print(f"  dataset.csv   : {csv_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Sync — Unified Synchronization & Reindexing"
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(DATA_DIR, "dataset.csv"),
        help="Path ke dataset.csv (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--app-url",
        default=os.environ.get("LARAVEL_APP_URL", "http://127.0.0.1:8000"),
        help="Laravel APP_URL",
    )
    parser.add_argument(
        "--method",
        default="combined",
        help="Metode ekstraksi: combined (cepat), ultra (lengkap), all (semua)",
    )
    args = parser.parse_args()

    exit_code = main(csv_path=args.csv, app_url=args.app_url, method=args.method)
    sys.exit(exit_code)
