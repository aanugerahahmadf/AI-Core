# -*- coding: utf-8 -*-
"""
rebuild_index.py — Wrapper minimal untuk ai_sync.py (backward compatibility)

File ini telah digabungkan ke dalam ai_sync.py.
Wrapper ini dipertahankan untuk menjaga kompatibilitas dengan Laravel & CLI eksternal.
"""

from __future__ import annotations

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_sync import main as ai_sync_main, DATA_DIR


def rebuild(
    csv_path: str | None = None,
    app_url:  str        = "http://127.0.0.1:8000",
    method:   str        = "combined",
) -> int:
    """Rebuild CBIR index dengan memanggil modul utama ai_sync.py."""
    csv_file = csv_path or os.path.join(DATA_DIR, "dataset.csv")
    print("[INFO] rebuild_index.py memanggil modul terpadu ai_sync.py...")
    return ai_sync_main(csv_path=csv_file, app_url=app_url, method=method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild CBIR index dari dataset.csv Laravel (Legacy Wrapper)"
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(DATA_DIR, "dataset.csv"),
        help="Path ke dataset.csv (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--app-url",
        default=os.environ.get("LARAVEL_APP_URL", "http://127.0.0.1:8000"),
        help="Laravel APP_URL untuk image_url",
    )
    parser.add_argument(
        "--method",
        default="combined",
        help="Metode ekstraksi: combined (cepat), ultra (lengkap), all (semua)",
    )
    args = parser.parse_args()

    exit_code = rebuild(csv_path=args.csv, app_url=args.app_url, method=args.method)
    sys.exit(exit_code)
