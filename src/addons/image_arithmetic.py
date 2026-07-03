from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np

from src.addons.extraction.extractor import get_extractor, extractors
from src.addons.finder import get_finder
from src.addons.data import load_feature_database, resolve_portable_path

_EPS = 1e-10


def _validate_dims(vecs: list[np.ndarray]) -> int:
    if not vecs:
        raise ValueError("Minimal 1 vektor diperlukan")
    dim = vecs[0].shape[0]
    for v in vecs:
        if v.shape[0] != dim:
            raise ValueError(f"Dimensi tidak seragam: {dim} vs {v.shape[0]}")
    return dim


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def add_features(vecs: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """
    PENJUMLAHAN (weighted sum).
    Fusi beberapa gambar: hasil = Σ(wi × vi), L2-normalized.
    """
    _validate_dims(vecs)
    n = len(vecs)
    if weights is None:
        weights = [1.0 / n] * n
    total = sum(v.astype(np.float64) * w for v, w in zip(vecs, weights))
    return _l2(total.astype(np.float32))


def subtract_features(vec_a: np.ndarray, vec_b: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    PENGURANGAN: A - α·B.
    Cari gambar mirip A tapi beda dari B.
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Shape mismatch: {vec_a.shape} vs {vec_b.shape}")
    result = (vec_a.astype(np.float64) - alpha * vec_b.astype(np.float64)).astype(np.float32)
    return _l2(result)


def multiply_features(vecs: list[np.ndarray]) -> np.ndarray:
    """
    PERKALIAN element-wise.
    Tonjolkan fitur yang kuat di SEMUA gambar (intersection).
    """
    _validate_dims(vecs)
    result = np.ones_like(vecs[0], dtype=np.float64)
    for v in vecs:
        result *= np.abs(v.astype(np.float64)) + _EPS
    return _l2(result.astype(np.float32))


def divide_features(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """
    PEMBAGIAN element-wise: A / (B + ε).
    Tonjolkan fitur kuat di A tapi lemah di B.
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Shape mismatch: {vec_a.shape} vs {vec_b.shape}")
    result = (vec_a.astype(np.float64) / (np.abs(vec_b.astype(np.float64)) + _EPS)).astype(np.float32)
    return _l2(result)


def average_features(vecs: list[np.ndarray]) -> np.ndarray:
    """Rata-rata (sama dengan add_features bobot sama)."""
    return add_features(vecs)


# ---------------------------------------------------------------------------
# Registry operasi
# ---------------------------------------------------------------------------

QueryOp = Callable[[list[np.ndarray]], np.ndarray]

OPS: dict[str, tuple[QueryOp, str, str]] = {
    "add":      (lambda v: add_features(v),         "+",  "Penjumlahan (fusi) fitur beberapa gambar"),
    "average":  (lambda v: average_features(v),     "avg","Rata-rata fitur beberapa gambar"),
    "subtract": (lambda v: subtract_features(v[0], v[1]), "-", "Gambar A MINUS gambar B"),
    "multiply": (lambda v: multiply_features(v),    "×",  "Perkalian element-wise fitur"),
    "divide":   (lambda v: divide_features(v[0], v[1]), "÷", "Gambar A DIBAGI gambar B"),
}


def arithmetic_search(
    image_paths: list[str],
    operation: str,
    method: str = "combined",
    metric: str = "euclidean",
    top_k: int = 20,
    db_path: str = "data/metadata.json",
    weights: list[float] | None = None,
) -> dict:
    """
    Pipeline lengkap: ekstrak fitur → aritmetika → search.

    Args:
        image_paths: Path ke gambar (add/average: ≥1, subtract/divide: ≥2).
        operation: 'add', 'average', 'subtract', 'multiply', 'divide'.
        method: Metode ekstraksi (default 'combined').
        metric: Metrik pencarian (default 'euclidean').
        top_k: Jumlah hasil (default 20).
        db_path: Path metadata.json.
        weights: Bobot per gambar (khusus 'add').

    Returns:
        Dict hasil pencarian + metadata operasi.
    """
    t0 = time.time()

    if operation not in OPS:
        raise ValueError(f"Operasi '{operation}' tak dikenal. Pilihan: {list(OPS.keys())}")

    op_func, op_sym, op_desc = OPS[operation]
    extractor = get_extractor(method)
    finder = get_finder(metric)

    vecs = []
    for p in image_paths:
        resolved = resolve_portable_path(p) if not os.path.exists(p) else p
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {resolved}")
        vecs.append(extractor.extract(resolved))

    if operation in ("subtract", "divide") and len(vecs) < 2:
        raise ValueError(f"Operasi '{operation}' butuh ≥2 gambar")

    if operation == "add":
        query_vec = add_features(vecs, weights)
    elif operation == "average":
        query_vec = average_features(vecs)
    else:
        query_vec = op_func(vecs)

    db = load_feature_database(db_path)
    images = db.get("images", [])
    is_sim = finder.is_similarity()

    def _calc_sim(raw: float) -> float:
        if is_sim:
            return round(max(0.0, raw * 100.0), 2)
        linear = max(0.0, 100.0 - (raw / 25.0 * 100.0))
        s = (linear / 100.0) ** 2 * 100.0
        return round(s if s >= 15.0 else 0.0, 2)

    scores = []
    for entry in images:
        feat_dict = entry.get("features", {})
        feat_list = (
            feat_dict.get(method)
            or feat_dict.get("combined")
            or feat_dict.get("combined_features")
            or feat_dict.get("deep_features")
        )
        if feat_list is None:
            continue
        candidate = np.array(feat_list, dtype=np.float32)
        if candidate.shape != query_vec.shape:
            continue
        scores.append((finder.compute(query_vec, candidate), entry))

    scores.sort(key=lambda x: x[0], reverse=is_sim)
    scores = scores[:top_k]

    results = [
        {
            "rank": i + 1,
            "id": entry.get("id"),
            "type": m.get("type", "product"),
            "owner_id": m.get("owner_id", ""),
            "name": m.get("name", "Unknown"),
            "category": m.get("category", ""),
            "price": m.get("price", 0),
            "similarity": _calc_sim(raw),
            "distance": round(raw, 4),
            "path": entry.get("path"),
        }
        for i, (raw, entry) in enumerate(scores)
        if (m := entry.get("metadata", {}))
    ]

    return {
        "success": True,
        "results": results,
        "total_results": len(results),
        "query_time_seconds": round(time.time() - t0, 4),
        "operation": {"name": operation, "symbol": op_sym, "description": op_desc},
        "method": method,
        "metric": metric,
        "source_images": image_paths,
    }
