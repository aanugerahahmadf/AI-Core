from __future__ import annotations

import time
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Average Precision untuk satu query
# ---------------------------------------------------------------------------

def average_precision(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """
    Hitung Average Precision untuk satu query.

    Args:
        retrieved_ids: List ID hasil retrieval, diurutkan dari paling relevan.
        relevant_ids : Set ID yang benar-benar relevan (ground truth).

    Returns:
        float: Average Precision [0, 1].
    """
    if not relevant_ids:
        return 0.0

    hits       = 0
    sum_prec   = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits     += 1
            sum_prec += hits / rank

    return sum_prec / len(relevant_ids)


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------

def mean_average_precision(
    queries: list[tuple[list[int], set[int]]]
) -> float:
    """
    Mean Average Precision (MAP) untuk sekumpulan query.

    Args:
        queries: List of (retrieved_ids, relevant_ids) tuples.

    Returns:
        float: MAP [0, 1].
    """
    if not queries:
        return 0.0
    aps = [average_precision(ret, rel) for ret, rel in queries]
    return float(np.mean(aps))


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(
    queries: list[tuple[list[int], set[int]]]
) -> float:
    """
    Mean Reciprocal Rank (MRR).

    Args:
        queries: List of (retrieved_ids, relevant_ids) tuples.

    Returns:
        float: MRR [0, 1].
    """
    if not queries:
        return 0.0

    rrs = []
    for retrieved_ids, relevant_ids in queries:
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    return float(np.mean(rrs))


# ---------------------------------------------------------------------------
# First Rank Accuracy
# ---------------------------------------------------------------------------

def first_rank_accuracy(
    queries: list[tuple[list[int], set[int]]]
) -> float:
    """
    Persentase query di mana hasil pertama adalah relevan.

    Args:
        queries: List of (retrieved_ids, relevant_ids) tuples.

    Returns:
        float: First Rank Accuracy [0, 1].
    """
    if not queries:
        return 0.0

    correct = sum(
        1 for retrieved_ids, relevant_ids in queries
        if retrieved_ids and retrieved_ids[0] in relevant_ids
    )
    return correct / len(queries)


# ---------------------------------------------------------------------------
# Query Time Benchmark
# ---------------------------------------------------------------------------

def benchmark_query_time(
    query_fn: Callable,
    image_paths: list[str],
    n_runs: int = 5,
) -> dict[str, float]:
    """
    Ukur rata-rata waktu query.

    Args:
        query_fn   : Fungsi yang menerima image_path dan mengembalikan hasil.
        image_paths: List path gambar untuk benchmark.
        n_runs     : Jumlah run per gambar.

    Returns:
        dict dengan 'mean', 'std', 'min', 'max' dalam detik.
    """
    times = []
    for path in image_paths:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            query_fn(path)
            times.append(time.perf_counter() - t0)

    arr = np.array(times)
    return {
        "mean" : float(arr.mean()),
        "std"  : float(arr.std()),
        "min"  : float(arr.min()),
        "max"  : float(arr.max()),
        "n"    : len(times),
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def evaluation_report(
    queries: list[tuple[list[int], set[int]]],
    avg_query_time: float | None = None,
) -> dict[str, float]:
    """
    Buat laporan evaluasi lengkap.

    Returns:
        dict berisi MAP, MRR, first_rank_accuracy, dan avg_query_time.
    """
    return {
        "map"                 : round(mean_average_precision(queries), 4),
        "mrr"                 : round(mean_reciprocal_rank(queries), 4),
        "first_rank_accuracy" : round(first_rank_accuracy(queries), 4),
        "avg_query_time_s"    : round(avg_query_time, 4) if avg_query_time else None,
        "n_queries"           : len(queries),
    }
