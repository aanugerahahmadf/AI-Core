from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import distance as scipy_distance


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Finder(ABC):
    """
    Base class untuk semua similarity/distance finders.
    """

    @abstractmethod
    def compute(self, query: np.ndarray, candidate: np.ndarray) -> float:
        """Hitung skor kemiripan antara dua vektor fitur."""

    def is_similarity(self) -> bool:
        """True jika skor lebih tinggi berarti lebih mirip."""
        return False

    def __call__(self, query: np.ndarray, candidate: np.ndarray) -> float:
        return self.compute(query, candidate)


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

class CosineFinder(Finder):
    """
    Cosine Similarity: 1 - cosine_distance.
    Range: [-1, 1], lebih tinggi = lebih mirip.
    """

    def compute(self, query: np.ndarray, candidate: np.ndarray) -> float:
        return float(1.0 - scipy_distance.cosine(query, candidate))

    def is_similarity(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Manhattan Distance (L1)
# ---------------------------------------------------------------------------

class ManhattanFinder(Finder):
    """
    Manhattan Distance (L1 norm).
    Lebih rendah = lebih mirip.
    """

    def compute(self, query: np.ndarray, candidate: np.ndarray) -> float:
        return float(scipy_distance.cityblock(query, candidate))

    def is_similarity(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Euclidean Distance (L2)  — default produksi
# ---------------------------------------------------------------------------

class EuclideanFinder(Finder):
    """
    Euclidean Distance (L2 norm).
    Lebih rendah = lebih mirip.
    Default untuk produksi karena performa terbaik (lihat README).
    """

    def compute(self, query: np.ndarray, candidate: np.ndarray) -> float:
        return float(scipy_distance.euclidean(query, candidate))

    def is_similarity(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Registry — sama persis dengan pola repo referensi
# ---------------------------------------------------------------------------

finders: dict[str, type] = {
    "cosine"    : CosineFinder,
    "manhattan" : ManhattanFinder,
    "euclidean" : EuclideanFinder,
}


def get_finder(metric: str = "euclidean") -> Finder:
    """
    Kembalikan instance finder berdasarkan nama metrik.

    Args:
        metric: Nama metrik dari `finders` dict.

    Returns:
        Instance Finder yang sudah siap digunakan.

    Raises:
        ValueError: Jika nama metrik tidak dikenal.
    """
    if metric not in finders:
        raise ValueError(
            f"Metrik '{metric}' tidak dikenal. "
            f"Pilihan: {list(finders.keys())}"
        )
    return finders[metric]()
