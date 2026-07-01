from __future__ import annotations

import csv
import json
import os
from typing import Any


def resolve_portable_path(path: str) -> str:
    """
    Mengubah path absolut dari sistem lain menjadi path valid di sistem saat ini.
    Prioritas utama: D:\\Weeding-Organizer-CBIR\\Mobile-App\\...
    """
    if not path:
        return ""

    # Standarkan separator path (Windows/Linux)
    path = os.path.normpath(path)

    # Jika path asli langsung ada, kembalikan
    if os.path.exists(path):
        return os.path.abspath(path)

    # Dapatkan root project: ai_core/src/addons/data.py → naik 3 level = Weeding-Organizer-CBIR
    addon_dir  = os.path.dirname(os.path.abspath(__file__))
    src_dir    = os.path.dirname(addon_dir)
    ai_core    = os.path.dirname(src_dir)
    parent_dir = os.path.dirname(ai_core)  # D:\Weeding-Organizer-CBIR

    # ----------------------------------------------------------------
    # PRIORITAS 1: Langsung cek Mobile-App terlebih dahulu
    # ----------------------------------------------------------------
    mobile_app_dir = os.path.join(parent_dir, "Mobile-App")

    if "storage" in path and os.path.exists(mobile_app_dir):
        idx      = path.find("storage")
        rel_part = path[idx:]
        candidate = os.path.join(mobile_app_dir, rel_part)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # ----------------------------------------------------------------
    # PRIORITAS 2: Cari di semua folder sibling lain
    # ----------------------------------------------------------------
    if "storage" in path:
        idx      = path.find("storage")
        rel_part = path[idx:]

        if os.path.exists(parent_dir):
            for folder in os.listdir(parent_dir):
                if folder == "Mobile-App":
                    continue  # sudah dicek di atas
                folder_path = os.path.join(parent_dir, folder)
                if os.path.isdir(folder_path):
                    candidate = os.path.join(folder_path, rel_part)
                    if os.path.exists(candidate):
                        return os.path.abspath(candidate)

            # Coba langsung di parent_dir
            candidate_direct = os.path.join(parent_dir, rel_part)
            if os.path.exists(candidate_direct):
                return os.path.abspath(candidate_direct)

    # ----------------------------------------------------------------
    # PRIORITAS 3: Fallback — cari nama file di storage Mobile-App
    # ----------------------------------------------------------------
    filename = os.path.basename(path)

    if os.path.exists(mobile_app_dir):
        mobile_storage = os.path.join(mobile_app_dir, "storage")
        if os.path.exists(mobile_storage):
            for root, _, files in os.walk(mobile_storage):
                if filename in files:
                    return os.path.abspath(os.path.join(root, filename))

    # Fallback generik: semua sibling dengan folder storage
    if os.path.exists(parent_dir):
        for folder in os.listdir(parent_dir):
            folder_path = os.path.join(parent_dir, folder)
            if os.path.isdir(folder_path):
                sibling_storage = os.path.join(folder_path, "storage")
                if os.path.exists(sibling_storage):
                    for root, _, files in os.walk(sibling_storage):
                        if filename in files:
                            return os.path.abspath(os.path.join(root, filename))

    # Jika semua gagal, kembalikan path asli
    return path



# ---------------------------------------------------------------------------
# Feature Database (metadata.json)
# ---------------------------------------------------------------------------

def load_feature_database(db_path: str) -> dict[str, Any]:
    """
    Load feature database dari file JSON.

    Format:
    {
        "images": [
            {
                "id": 1,
                "path": "/abs/path/to/image.jpg",
                "metadata": { "type": "product", "owner_id": 5, ... },
                "features": {
                    "deep_features": [...],
                    "color_histogram": [...],
                    "texture_features": [...],
                    "combined_features": [...]
                }
            },
            ...
        ]
    }

    Args:
        db_path: Path ke file metadata.json.

    Returns:
        dict dengan key 'images'.
    """
    if not os.path.exists(db_path):
        return {"images": []}

    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalisasi format lama (list) ke format baru (dict)
    if isinstance(data, list):
        normalized = {"images": []}
        for i, item in enumerate(data):
            p = resolve_portable_path(item.get("path", ""))
            m = item.get("metadata", item)
            if "image_path" in m:
                m["image_path"] = resolve_portable_path(m["image_path"])
            normalized["images"].append({
                "id"       : i + 1,
                "path"     : p,
                "metadata" : m,
                "features" : item.get("features", {}),
            })
        return normalized

    if isinstance(data, dict) and "images" in data:
        for img in data["images"]:
            img["path"] = resolve_portable_path(img.get("path", ""))
            m = img.get("metadata", {})
            if "image_path" in m:
                m["image_path"] = resolve_portable_path(m["image_path"])
        return data

    return {"images": []}


def save_feature_database(db: dict[str, Any], db_path: str) -> None:
    """
    Simpan feature database ke file JSON.

    Args:
        db     : dict dengan key 'images'.
        db_path: Path tujuan.
    """
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Dataset CSV (dari Laravel php artisan cbir:sync)
# ---------------------------------------------------------------------------

def load_dataset_csv(csv_path: str) -> list[dict[str, str]]:
    """
    Load dataset CSV yang di-generate oleh Laravel.

    Kolom CSV: ID, Type, Name, Category, Price, Discount_Price,
               Organizer, Image_Path, Description

    Args:
        csv_path: Path ke dataset.csv.

    Returns:
        List of row dicts.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV tidak ditemukan: {csv_path}\n"
            "Jalankan: php artisan cbir:sync"
        )

    csv.field_size_limit(10_000_000)  # Support large feature vector columns
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def dataset_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """
    Hitung statistik dari dataset CSV.

    Returns:
        dict berisi total, per_type, per_category.
    """
    per_type: dict[str, int]     = {}
    per_category: dict[str, int] = {}

    for row in rows:
        t = row.get("Type", "unknown").lower()
        c = row.get("Category", "unknown").lower()
        per_type[t]     = per_type.get(t, 0) + 1
        per_category[c] = per_category.get(c, 0) + 1

    return {
        "total"       : len(rows),
        "per_type"    : per_type,
        "per_category": per_category,
    }
