from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np

from src.addons.data import load_feature_database
from src.addons.finder import get_finder
from src.addons.metrics import mean_average_precision


def _get_feature(entry: dict, key: str) -> np.ndarray | None:
    feat_dict = entry.get("features", {})
    raw = (
        feat_dict.get(key)
        or feat_dict.get({"deep": "deep_features", "color": "color_histogram", "texture": "texture_features"}.get(key, key))
    )
    if raw is None:
        return None
    return np.array(raw, dtype=np.float32)


def _search_with_weights(
    db: list[dict],
    query_id: int,
    w_deep: float,
    w_color: float,
    w_texture: float,
    metric: str = "cosine",
) -> list[int]:
    q_entry = db[query_id]
    q_deep = _get_feature(q_entry, "deep_features")
    q_color = _get_feature(q_entry, "color_histogram")
    q_texture = _get_feature(q_entry, "texture_features")

    if q_deep is None:
        return []

    finder = get_finder(metric)
    is_sim = finder.is_similarity()

    scores: list[tuple[float, int]] = []
    for i, entry in enumerate(db):
        if i == query_id:
            continue
        d = _get_feature(entry, "deep_features")
        c = _get_feature(entry, "color_histogram")
        t = _get_feature(entry, "texture_features")
        if d is None:
            continue

        sim = 0.0
        n = 0
        if w_deep > 0:
            sim += w_deep * finder.compute(q_deep, d)
            n += 1
        if w_color > 0 and c is not None and q_color is not None:
            try:
                sim += w_color * finder.compute(q_color, c)
                n += 1
            except Exception:
                pass
        if w_texture > 0 and t is not None and q_texture is not None:
            try:
                sim += w_texture * finder.compute(q_texture, t)
                n += 1
            except Exception:
                pass

        if n > 0:
            scores.append((sim / n, i))

    scores.sort(key=lambda x: x[0], reverse=is_sim)
    return [db[idx].get("id", idx) for _, idx in scores]


def grid_search_weights(
    db_path: str = "data/metadata.json",
    metric: str = "cosine",
    w_candidates: list[float] | None = None,
    top_k: int = 20,
    sample_pct: float = 1.0,
) -> dict[str, Any]:
    if w_candidates is None:
        w_candidates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    db_data = load_feature_database(db_path)
    images = db_data.get("images", [])

    if len(images) < 2:
        return {"error": "Need at least 2 images"}

    sample_size = max(2, int(len(images) * sample_pct))
    if sample_size < len(images):
        rng = np.random.default_rng(42)
        sample_idx = sorted(rng.choice(len(images), sample_size, replace=False).tolist())
    else:
        sample_idx = list(range(len(images)))

    results: list[tuple[float, float, float, float]] = []
    total = len(list(itertools.product(w_candidates, w_candidates, w_candidates)))
    done = 0

    for wd, wc, wt in itertools.product(w_candidates, w_candidates, w_candidates):
        if wd + wc + wt == 0:
            done += 1
            continue

        queries: list[tuple[list[int], set[int]]] = []
        for qid in sample_idx:
            q_entry = images[qid]
            q_cat = q_entry.get("metadata", {}).get("category", "")
            q_owner = q_entry.get("metadata", {}).get("owner_id")

            retrieved = _search_with_weights(images, qid, wd, wc, wt, metric)[:top_k]

            relevant = set()
            for i, entry in enumerate(images):
                if i == qid:
                    continue
                m = entry.get("metadata", {})
                if q_cat and m.get("category") == q_cat:
                    relevant.add(entry.get("id", i))
                elif q_owner and m.get("owner_id") == q_owner:
                    relevant.add(entry.get("id", i))

            if relevant:
                queries.append((retrieved, relevant))

        map_score = mean_average_precision(queries) if queries else 0.0
        results.append((map_score, wd, wc, wt))
        done += 1

    results.sort(key=lambda x: x[0], reverse=True)

    best = results[0]
    return {
        "best_map": round(float(best[0]), 4),
        "best_weights": {"deep": float(best[1]), "color": float(best[2]), "texture": float(best[3])},
        "top_5": [
            {"map": round(float(r[0]), 4), "weights": {"deep": float(r[1]), "color": float(r[2]), "texture": float(r[3])}}
            for r in results[:5]
        ],
        "n_queries": len(sample_idx),
        "metric": metric,
        "top_k": top_k,
    }
