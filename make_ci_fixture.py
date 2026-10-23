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
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CSV_PATH = DATA_DIR / "dataset.csv"
BACKEND_CATALOG_DIR = (
    ROOT.parent / "Mobile-And-Backend" / "public" / "images" / "catalog"
)

HEADER = [
    "ID", "Type", "Name", "Category", "Price", "Discount_Price",
    "Organizer", "Image_Path", "Description",
]


def _make_image(path: Path, color: tuple, text: str) -> None:
    """Gambar RGB polos 256x256 dengan label di tengah."""
    source = BACKEND_CATALOG_DIR / path.name
    if source.is_file():
        shutil.copy2(source, path)
        return

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

    # 8 gambar, 2 per item. Dua gambar per katalog itu penting: itulah kondisi
    # nyata (produk punya galeri), dan itulah yang rusak ketika
    # `/api/index/add` salah dedup per (type, owner_id) sehingga tiap produk
    # hanya menyisakan 1 gambar. Fixture dengan 1 gambar/produk tidak akan
    # menangkap bug itu.
    images = [
        ("catalog-1.png", (220, 220, 220), "Catalog 1 A"),
        ("catalog-2.png", (200, 210, 225), "Catalog 1 B"),
        ("catalog-3.png", (218, 165, 32), "Catalog 2 A"),
        ("catalog-4.png", (240, 200, 120), "Catalog 2 B"),
        ("catalog-5.png", (255, 192, 203), "Package 1 A"),
        ("catalog-6.png", (245, 175, 190), "Package 1 B"),
        ("catalog-7.png", (173, 216, 230), "Package 2 A"),
        ("catalog-8.png", (150, 200, 220), "Package 2 B"),
    ]

    # (file, type, owner_id, category)
    layout = [
        ("catalog-1.png", "catalog", 1, "backdrop pelaminan"),
        ("catalog-2.png", "catalog", 1, "backdrop pelaminan"),
        ("catalog-3.png", "catalog", 2, "backdrop pelaminan"),
        ("catalog-4.png", "catalog", 2, "backdrop pelaminan"),
        ("catalog-5.png", "package", 1, "dekorasi pelaminan"),
        ("catalog-6.png", "package", 1, "dekorasi pelaminan"),
        ("catalog-7.png", "package", 2, "dekorasi pelaminan"),
        ("catalog-8.png", "package", 2, "dekorasi pelaminan"),
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
            # Nilai bervariasi agar visualisasi notebook tidak gagal saat KDE
            # menghitung kovarians pada fixture kecil.
            15000000 + owner_id * 1000000 + (500000 if row_type == "package" else 0),
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
