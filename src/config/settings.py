# -*- coding: utf-8 -*-
"""Konfigurasi dasar AI Core CBIR."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
METADATA_PATH = DATA_DIR / "metadata.json"
DATASET_CSV = DATA_DIR / "dataset.csv"
UPLOAD_DIR = DATA_DIR / "uploads"
