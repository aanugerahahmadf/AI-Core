from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gray(image_path: str) -> np.ndarray:
    """Load gambar sebagai grayscale uint8."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca gambar: {image_path}")
    return img


def _load_bgr(image_path: str) -> np.ndarray:
    """Load gambar sebagai BGR uint8."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca gambar: {image_path}")
    return img


def _bow_histogram(descriptors: np.ndarray | None, n_bins: int = 256) -> np.ndarray:
    """
    Konversi raw keypoint descriptors ke histogram BoW sederhana.
    Jika tidak ada keypoint, kembalikan vektor nol.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(n_bins, dtype=np.float32)

    flat = descriptors.flatten().astype(np.float32)
    hist, _ = np.histogram(flat, bins=n_bins, range=(0, 256))
    total = hist.sum() + 1e-7
    return (hist / total).astype(np.float32)


# ---------------------------------------------------------------------------
# AKAZE
# ---------------------------------------------------------------------------

class AKAZEDescriptor:
    """
    AKAZE feature descriptor.
    Menghasilkan vektor histogram 256-dim dari keypoint descriptors.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins
        self._detector = cv2.AKAZE_create()

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_gray(image_path)
        _, descriptors = self._detector.detectAndCompute(img, None)
        return _bow_histogram(descriptors, self.n_bins)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# ORB
# ---------------------------------------------------------------------------

class ORBDescriptor:
    """
    ORB feature descriptor.
    Menghasilkan vektor histogram 256-dim dari keypoint descriptors.
    """

    def __init__(self, n_features: int = 500, n_bins: int = 256):
        self.n_bins = n_bins
        self._detector = cv2.ORB_create(nfeatures=n_features)

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_gray(image_path)
        _, descriptors = self._detector.detectAndCompute(img, None)
        return _bow_histogram(descriptors, self.n_bins)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# Color Histogram (HSV) — 512-dim
# ---------------------------------------------------------------------------

class ColorHistogramDescriptor:
    """
    3-D HSV color histogram (8×8×8 = 512 bins).
    Berguna sebagai fitur pelengkap deep features.
    """

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_bgr(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)  # (512,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# RGB Histogram — 768-dim (256 per channel)
# ---------------------------------------------------------------------------

class RGBHistogramDescriptor:
    """
    Per-channel RGB histogram (256 bins × 3 channels = 768-dim).
    Menangkap distribusi warna mentah tanpa transformasi ruang warna.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_bgr(image_path)
        feats = []
        for ch in range(3):  # B, G, R
            hist = cv2.calcHist([img], [ch], None, [self.n_bins], [0, 256])
            cv2.normalize(hist, hist)
            feats.append(hist.flatten())
        return np.concatenate(feats).astype(np.float32)  # (768,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# LBP Texture Descriptor — 256-dim
# ---------------------------------------------------------------------------

class LBPDescriptor:
    """
    Local Binary Pattern texture descriptor.
    Menghasilkan histogram 256-dim.
    """

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_gray(image_path)
        img = cv2.resize(img, (128, 128))

        lbp = np.zeros_like(img, dtype=np.uint8)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                center = int(img[i, j])
                code = 0
                code |= (int(img[i - 1, j - 1]) > center) << 7
                code |= (int(img[i - 1, j])     > center) << 6
                code |= (int(img[i - 1, j + 1]) > center) << 5
                code |= (int(img[i,     j + 1]) > center) << 4
                code |= (int(img[i + 1, j + 1]) > center) << 3
                code |= (int(img[i + 1, j])     > center) << 2
                code |= (int(img[i + 1, j - 1]) > center) << 1
                code |= (int(img[i,     j - 1]) > center) << 0
                lbp[i, j] = code

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7
        return hist  # (256,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# HOG (Histogram of Oriented Gradients) — shape & edge features
# ---------------------------------------------------------------------------

class HOGDescriptor:
    """
    HOG descriptor untuk menangkap bentuk dan tepi objek.
    Resize gambar ke 128×128 → menghasilkan ~1764-dim vektor.
    """

    def __init__(self, win_size=(128, 128), block_size=(16, 16),
                 block_stride=(8, 8), cell_size=(8, 8), n_bins=9):
        self._hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, n_bins
        )

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_gray(image_path)
        img = cv2.resize(img, (128, 128))
        feat = self._hog.compute(img)
        feat = feat.flatten().astype(np.float32)
        # L2-normalize
        norm = np.linalg.norm(feat) + 1e-7
        return (feat / norm).astype(np.float32)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# Gabor Texture — multi-scale multi-orientation
# ---------------------------------------------------------------------------

class GaborDescriptor:
    """
    Gabor filter bank descriptor.
    5 skala × 8 orientasi = 40 filter → rata-rata & std energy per filter = 80-dim.
    Sangat efektif untuk mengenali pola tekstur kain, dekorasi, dll.
    """

    def __init__(self):
        self._kernels = self._build_kernels()

    @staticmethod
    def _build_kernels() -> list:
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 8):          # 8 orientasi
            for sigma in (1, 3, 5, 7, 9):                      # 5 skala
                kern = cv2.getGaborKernel(
                    (21, 21), sigma, theta,
                    lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F
                )
                kern /= kern.sum() + 1e-7
                kernels.append(kern)
        return kernels  # 40 kernels

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_gray(image_path).astype(np.float32) / 255.0
        feats = []
        for kern in self._kernels:
            filtered = cv2.filter2D(img, cv2.CV_32F, kern)
            feats.append(filtered.mean())
            feats.append(filtered.std())
        feat = np.array(feats, dtype=np.float32)  # (80,)
        norm = np.linalg.norm(feat) + 1e-7
        return (feat / norm).astype(np.float32)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# SIFT BoW Descriptor — 128-dim per keypoint → histogram 256-dim
# ---------------------------------------------------------------------------

class SIFTDescriptor:
    """
    SIFT feature descriptor dengan Bag-of-Words pooling.
    Menghasilkan histogram 256-dim yang scale & rotation invariant.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins
        self._sift = cv2.SIFT_create(nfeatures=500)

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_gray(image_path)
        _, descriptors = self._sift.detectAndCompute(img, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_bins, dtype=np.float32)
        # Gunakan nilai absolut karena SIFT sudah float
        flat = np.clip(descriptors.flatten(), 0, 255).astype(np.float32)
        hist, _ = np.histogram(flat, bins=self.n_bins, range=(0, 256))
        total = hist.sum() + 1e-7
        return (hist / total).astype(np.float32)  # (256,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# Dominant Color (K-Means palette) — 48-dim (16 colors × 3 channels)
# ---------------------------------------------------------------------------

class DominantColorDescriptor:
    """
    Ekstrak K warna dominan dari gambar menggunakan K-Means clustering.
    Menghasilkan vektor 48-dim (16 cluster × 3 channel RGB, L2-normalized).
    Sangat berguna untuk matching tema warna dekorasi pernikahan.
    """

    def __init__(self, k: int = 16, resize: int = 64):
        self.k = k
        self.resize = resize

    def extract(self, image_path: str) -> np.ndarray:
        img = _load_bgr(image_path)
        img = cv2.resize(img, (self.resize, self.resize))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        pixels = img_rgb.reshape(-1, 3)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, self.k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
        # Urutkan berdasarkan frekuensi warna (dominan dulu)
        counts = np.bincount(labels.flatten(), minlength=self.k)
        order  = np.argsort(-counts)
        palette = centers[order].flatten() / 255.0  # normalize ke [0,1]

        feat = palette.astype(np.float32)  # (48,)
        norm = np.linalg.norm(feat) + 1e-7
        return (feat / norm).astype(np.float32)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)
