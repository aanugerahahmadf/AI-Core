#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_ci_fixture.py — Bangun dataset.csv + metadata.json yang valid untuk CI.

Kenapa perlu
------------
`data/dataset.csv` yang ikut ter-commit berisi path ABSOLUTU Windows
(`D:\\Weeding-Organizer-CBIR\\...`). Di runner Linux path itu tidak ada, sehingga
`resolve_portable_path()` gagal untuk seluruh baris dan
`build_features` menulis indeks KOSONG. Akibatnya `tests/test_cbir.py` gagal
dengan pesan "Tidak ada product di metadata.json" — bukan karena kodenya salah,
tapi karena tidak ada data.

Skrip ini membuat gambar placeholder lalu menulis `dataset.csv` dengan path
yang dihitung saat runtime, sehingga selalu valid di OS mana pun. Hasilnya test
CBIR benar-benar menguji ekstraksi fitur, bukan sekadar melewati argumen kosong.

Pakai:
    python make_ci_fixture.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CSV_PATH = DATA_DIR / "dataset.csv"

HEADER = [
    "ID", "Type", "Name", "Category", "Price", "Discount_Price",
    "Organizer", "Image_Path", "Description",
]


def _make_image(path: Path, color: tuple, text: str) -> None:
    """Gambar RGB polos 256x256 dengan label di tengah."""
    img = Image.new("RGB", (256, 256), color=color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
    except AttributeError:  # Pillow sangat lama
        w, h = font.getsize(text)

    draw.text(((256 - w) // 2, (256 - h) // 2), text, fill=(255, 255, 255), font=font)
    img.save(path)


def build() -> int:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 8 gambar, 2 per item. Dua gambar per produk itu penting: itulah kondisi
    # nyata (produk punya galeri), dan itulah yang rusak ketika
    # `/api/index/add` salah dedup per (type, owner_id) sehingga tiap produk
    # hanya menyisakan 1 gambar. Fixture dengan 1 gambar/produk tidak akan
    # menangkap bug itu.
    images = [
        ("ci-product-1a.jpg", (220, 220, 220), "Product 1 A"),
        ("ci-product-1b.jpg", (200, 210, 225), "Product 1 B"),
        ("ci-product-2a.jpg", (218, 165, 32), "Product 2 A"),
        ("ci-product-2b.jpg", (240, 200, 120), "Product 2 B"),
        ("ci-package-1a.jpg", (255, 192, 203), "Package 1 A"),
        ("ci-package-1b.jpg", (245, 175, 190), "Package 1 B"),
        ("ci-package-2a.jpg", (173, 216, 230), "Package 2 A"),
        ("ci-package-2b.jpg", (150, 200, 220), "Package 2 B"),
    ]

    # (file, type, owner_id, category)
    layout = [
        ("ci-product-1a.jpg", "product", 1, "backdrop pelaminan"),
        ("ci-product-1b.jpg", "product", 1, "backdrop pelaminan"),
        ("ci-product-2a.jpg", "product", 2, "backdrop pelaminan"),
        ("ci-product-2b.jpg", "product", 2, "backdrop pelaminan"),
        ("ci-package-1a.jpg", "package", 1, "dekorasi pelaminan"),
        ("ci-package-1b.jpg", "package", 1, "dekorasi pelaminan"),
        ("ci-package-2a.jpg", "package", 2, "dekorasi pelaminan"),
        ("ci-package-2b.jpg", "package", 2, "dekorasi pelaminan"),
    ]

    for filename, color, label in images:
        _make_image(UPLOAD_DIR / filename, color, label)

    label_of = {filename: label for filename, _, label in images}

    rows = []
    for filename, row_type, owner_id, category in layout:
        rows.append([
            owner_id,
            row_type,
            f"{row_type.capitalize()} {owner_id} Fixture",
            category,
            15000000,
            "",
            "Vendor Fixture",
            str(UPLOAD_DIR / filename),   # path absolut milik mesin ini
            f"Deskripsi {label_of[filename]} untuk fixture pengujian otomatis.",
        ])

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"  images : {len(images)} file di {UPLOAD_DIR}")
    print(f"  dataset: {CSV_PATH} ({len(rows)} baris, 2 gambar per item)")
    return 0


if __name__ == "__main__":
    sys.exit(build())
