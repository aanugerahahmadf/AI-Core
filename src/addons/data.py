from __future__ import annotations

import csv
import json
import os
import threading
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
# Feature key canonicalization
# ---------------------------------------------------------------------------
# Beberapa reader lama (CBIREngine, optimize_weights, ai_sync CSV sync) masih
# mengakses fitur lewat nama alias. Alias di-expand di memori saat load dan
# dibuang lagi saat save, sehingga file di disk hanya menyimpan tiap vektor
# sekali saja tanpa merusak reader mana pun.
FEATURE_ALIASES: dict[str, str] = {
    "combined_features": "combined",
    "deep_features"     : "deep",
    "resnet50"          : "deep",
    "color_histogram"   : "color",
    "texture_features"  : "lbp",
}

# Vektor yang tetap kosong pada method umum ("combined"/"ultra"). Dilewati saat
# save supaya metadata.json tidak dijejakkan array kosong.
OPTIONAL_FEATURE_KEYS = frozenset({
    "efficientnet", "vgg16", "rgb_histogram", "dominant_color",
    "gabor", "hog", "sift", "akaze",
})

# Semua key yang nilainya vektor fitur (dipakai untuk pembulatan saat save).
FEATURE_VECTOR_KEYS = frozenset({
    "combined", "deep", "color", "lbp",
}) | OPTIONAL_FEATURE_KEYS

# Desimal yang disimpan per komponen fitur. Vektor CBIR ternormalisasi
# (L1/cosine) jadi 6 desimal jauh melebihi presisi yang dibutuhkan, sementara
# repr(float) default menulis 17 digit sehingga ukuran file ~2x lebih besar
# tanpa gunanya.
FLOAT_PRECISION = 6

# Cache in-memory berbasis sidik jari file: abspath -> (mtime, size, database).
# Dipakai supaya setiap request Flask tidak parse ulang metadata.json yang besar.
_DB_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}
_DB_CACHE_LOCK = threading.Lock()


def invalidate_feature_cache(db_path: str | None = None) -> None:
    """
    Buang cache feature database.

    Args:
        db_path: Path spesifik. None = kosongkan semua cache.
    """
    with _DB_CACHE_LOCK:
        if db_path is None:
            _DB_CACHE.clear()
        else:
            _DB_CACHE.pop(os.path.abspath(db_path), None)


def _canonicalize_features(features: dict[str, Any]) -> dict[str, Any]:
    """
    Expand alias jadi key kanonik (in-place) agar reader lama tetap nemu.

    Alias ditunjuk ke objek vektor yang SAMA dengan key kanonik, bukan salinan —
    ini membuat file yang masih menyimpan alias (mis. hasil rebuild versi lama)
    langsung hemat ~40% RAM begitu dibaca.
    """
    for alias, canonical in FEATURE_ALIASES.items():
        vec = features.get(alias)
        if not vec:
            continue
        canonical_vec = features.get(canonical)
        features[canonical] = canonical_vec if canonical_vec else vec
        features[alias] = features[canonical]
    return features


def _strip_features_for_disk(features: dict[str, Any]) -> dict[str, Any]:
    """
    Siapkan dict fitur untuk ditulis ke disk.

    - Alias duplikat dibuang (hanya key kanonik yang disimpan).
    - Vektor opsional yang kosong dilewati.
    - Vektor fitur dibulatkan ke FLOAT_PRECISION desimal via numpy (vektorized,
      jauh lebih murah daripada `round()` per-elemen di Python).
    """
    import numpy as np

    out: dict[str, Any] = {}
    for key, value in features.items():
        if key in FEATURE_ALIASES:
            continue
        if key in FEATURE_VECTOR_KEYS:
            if not value:
                if key in OPTIONAL_FEATURE_KEYS:
                    continue
                out[key] = value
                continue
            # WAJIB float64: float32 tidak bisa mewakili hasil round 6 desimal,
            # sehingga .tolist() memunculkan kembali angka 17 digit.
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 1 and arr.size:
                value = np.round(arr, FLOAT_PRECISION).tolist()
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Feature Database (metadata.json)
# ---------------------------------------------------------------------------

def load_feature_database(db_path: str) -> dict[str, Any]:
    """
    Load feature database dari file JSON, dengan cache in-memory.

    Cache diinvalidasi otomatis lewat sidik jari file (mtime + size), jadi
    `php artisan cbir:sync` / rebuild otomatis terbaca tanpa restart server.

    Format:
    {
        "images": [
            {
                "id": 1,
                "path": "/abs/path/to/image.jpg",
                "metadata": { "type": "product", "owner_id": 5, ... },
                "features": {
                    "combined": [...],
                    "deep": [...],
                    "color": [...],
                    "lbp": [...]
                }
            },
            ...
        ]
    }

    Args:
        db_path: Path ke file metadata.json.

    Returns:
        dict dengan key 'images'. Alias fitur sudah di-expand.

    Warning:
        Objek hasil load TIDAK boleh dimutasi tanpa langsung memanggil
        save_feature_database() afterwards — karena instance-nya dipakai bersama.
    """
    if not os.path.exists(db_path):
        return {"images": []}

    cache_key = os.path.abspath(db_path)
    stat      = os.stat(cache_key)
    fingerprint = (stat.st_mtime, stat.st_size)

    with _DB_CACHE_LOCK:
        hit = _DB_CACHE.get(cache_key)
        if hit is not None and (hit[0], hit[1]) == fingerprint:
            return hit[2]

    with open(cache_key, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalisasi format lama (list) ke format baru (dict)
    if isinstance(data, list):
        normalized = {"images": []}
        for i, item in enumerate(data):
            p = item.get("path", "")
            m = item.get("metadata", item)
            if "image_path" in m:
                m["image_path"] = resolve_portable_path(m["image_path"])
            normalized["images"].append({
                "id"       : i + 1,
                "path"     : resolve_portable_path(p),
                "metadata" : m,
                "features" : _canonicalize_features(item.get("features", {})),
            })
        data = normalized

    elif isinstance(data, dict) and "images" in data:
        for img in data["images"]:
            img["path"] = resolve_portable_path(img.get("path", ""))
            m = img.get("metadata", {})
            if "image_path" in m:
                m["image_path"] = resolve_portable_path(m["image_path"])
            img["features"] = _canonicalize_features(img.get("features", {}))

    else:
        data = {"images": []}

    with _DB_CACHE_LOCK:
        _DB_CACHE[cache_key] = (fingerprint[0], fingerprint[1], data)

    return data


def save_feature_database(db: dict[str, Any], db_path: str) -> None:
    """
    Simpan feature database ke file JSON (compact, tanpa alias duplikat).

    - Alias fitur dibuang (hanya key kanonik yang ditulis).
    - Vektor opsional yang kosong dilewati.
    - Separator ringkas + penulisan atomik via os.replace().

    Args:
        db     : dict dengan key 'images'.
        db_path: Path tujuan.
    """
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)

    payload: dict[str, Any] = dict(db)
    if isinstance(payload.get("images"), list):
        payload["images"] = [
            {**img, "features": _strip_features_for_disk(img.get("features", {}))}
            for img in payload["images"]
        ]

    # Tulis via file sementara supaya reader tidak pernah melihat index setengah jadi
    tmp_path = db_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp_path, db_path)

    # Segarkan cache dengan instance yang sama supaya load berikutnya free
    stat = os.stat(db_path)
    with _DB_CACHE_LOCK:
        _DB_CACHE[os.path.abspath(db_path)] = (stat.st_mtime, stat.st_size, db)


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
