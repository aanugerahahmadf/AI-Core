# -*- coding: utf-8 -*-
"""src/rebuild_index.py - entry point untuk `make sync`."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features.build_features import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", dest="csv", default=None)
    parser.add_argument("--app-url", dest="app_url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    csv_path = args.csv
    build_features(method="combined", csv_path=csv_path, app_url=args.app_url)


if __name__ == "__main__":
    main()