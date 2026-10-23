# -*- coding: utf-8 -*-
"""Tipe media CBIR (catalog / package) dan normalisasi nilainya.

The whole index (dataset.csv, inputs/dataset_clean.csv and data/metadata.json)
has `type` column/field yang hanya boleh salah satu dari dua nilai kanonik
di bawah. `normalize_type()` memetakan berbagai alias (Indonesia/Inggris) ke
dua nilai itu agar indeks tetap konsisten.
"""

from __future__ import annotations

# Nilai kanonik tipe media yang disimpan di metadata.json / CSV.
CATALOG = "catalog"
PACKAGE = "package"

# Alias -> nilai kanonik. String yang tidak dikenal dipetakan ke CATALOG (default
# aman, sesuai validasi "hanya catalog atau package yang sah").
_PACKAGE_ALIASES = {
    "package", "packages", "paket", "paketan", "package decoration",
    "decoration package", "decor package",
}
_CATALOG_ALIASES = {
    "catalog", "catalogs", "katalog", "katalog produk", "catalog product",
    "product", "produk", "item", "catalogue",
}


def normalize_type(value) -> str:
    """Normalisasi nilai `type` media ke salah satu dari CATALOG / PACKAGE."""
    raw = "" if value is None else str(value)
    t = raw.strip().lower()
    if t in _PACKAGE_ALIASES:
        return PACKAGE
    if t in _CATALOG_ALIASES or t == "":
        return CATALOG
    # Nilai tidak dikenal: default ke CATALOG (validasi "hanya catalog/package").
    return CATALOG
