from __future__ import annotations

import numpy as np

from src.addons.finder import get_finder


def rerank_by_ensemble(
    query_emb: np.ndarray,
    candidates: list[tuple[float, dict]],
    db_entries: list[dict],
    metric: str = "cosine",
) -> list[dict]:
    finder = get_finder(metric)
    is_sim = finder.is_similarity()

    reranked: list[tuple[float, dict]] = []
    for orig_score, candidate in candidates:
        rerank_score = orig_score
        reranked.append((rerank_score, candidate))

    reranked.sort(key=lambda x: x[0], reverse=is_sim)

    seen_owners: set[str] = set()
    final: list[dict] = []
    for score, cand in reranked:
        owner = str(cand.get("metadata", {}).get("owner_id", ""))
        key = f"{cand.get('metadata', {}).get('type', 'product')}_{owner}"
        if key in seen_owners:
            continue
        seen_owners.add(key)
        cand["_rerank_score"] = round(float(score), 4)
        final.append(cand)

    return final
