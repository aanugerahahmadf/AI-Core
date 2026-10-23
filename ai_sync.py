# -*- coding: utf-8 -*-
"""ai_sync.py - trigger sinkronisasi fitur CBIR ke dataset.csv.

Modul ini dipanggil oleh route /api/sync (php artisan ai:sync) dan oleh
/api/index/rebuild-from-dataset. Implementasi minimal yang aman: delegasi ke
build_features dan reload engine.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features.build_features import build_features
from src.addons.data import load_feature_database


def sync_features_to_csv(csv_path=None, metadata_path=None):
    """Sinkronisasi ringan: pastikan metadata terbaca; tidak menulis CSV."""
    try:
        if metadata_path and os.path.exists(metadata_path):
            load_feature_database(metadata_path)
        return None
    except Exception:
        return None


def main(csv_path=None, app_url="http://127.0.0.1:8000"):
    try:
        csv_file = csv_path or os.path.join(ROOT, "data", "dataset.csv")
        if not os.path.exists(csv_file):
            print("Dataset CSV tidak ditemukan: " + csv_file)
            return 1
        build_features(method="combined", csv_path=csv_file, app_url=app_url)
        return 0
    except Exception as exc:
        print("AI Sync gagal:", exc)
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--app-url", dest="app_url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    code = main(csv_path=args.csv_path, app_url=args.app_url)
    sys.exit(code)
EOF
# write to ai_core/ai_sync.py
import shutil
shutil.copyfile("D:/WeddingFlowersDecorationApp/ai_core/src/config/__init__.py", "D:/tmp_dummy") if False else None