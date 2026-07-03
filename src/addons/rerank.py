from __future__ import annotations

import numpy as np

from src.addons.finder import get_finder


def _extract_feat(entry: dict, key: str, fallbacks: list[str] | None = None) -> np.ndarray | None:
    feat_dict = entry.get("features", {})
    for k in [key] + (fallbacks or []):
        raw = feat_dict.get(k)
        if raw is not None:
            return np.array(raw, dtype=np.float32)
    return None


def rerank_by_ensemble(
    query_emb: np.ndarray,
    candidates: list[tuple[float, dict]],
    db_entries: list[dict],
    metric: str = "cosine",
    top_k: int = 20,
) -> list[dict]:
    finder = get_finder(metric)
    q_color = _extract_feat({"features": {}}, "", [])
    q_texture = _extract_feat({"features": {}}, "", [])

    reranked: list[tuple[float, dict]] = []
    for orig_score, candidate in candidates:
        meta = candidate.get("metadata", {})
        img_path = candidate.get("path", "")

        rerank_score = orig_score * 0.6

        color_feat = _extract_feat(candidate, "color_histogram", ["color"])
        if color_feat is not None:
            try:
                cs = finder.compute(query_emb, color_feat)
                rerank_score += cs * 0.25
            except Exception:
                pass

        texture_feat = _extract_feat(candidate, "texture_features", ["texture", "lbp"])
        if texture_feat is not None:
            try:
                ts = finder.compute(query_emb, texture_feat)
                rerank_score += ts * 0.15
            except Exception:
                pass

        desc = (meta.get("description") or "").lower()
        name = (meta.get("name") or "").lower()

        reranked.append((rerank_score, candidate, desc, name))

    reranked.sort(key=lambda x: x[0], reverse=True)

    seen_owners: set[str] = set()
    final: list[dict] = []
    for score, cand, desc, name in reranked:
        owner = str(cand.get("metadata", {}).get("owner_id", ""))
        key = f"{cand.get('metadata', {}).get('type', 'product')}_{owner}"
        if key in seen_owners:
            continue
        seen_owners.add(key)
        cand["_rerank_score"] = round(float(score), 4)
        final.append(cand)
        if len(final) >= top_k:
            break

    return final
