from __future__ import annotations

import numpy as np

from src.addons.extraction.compressor import (
    EfficientNetCompressor,
    ResNet50Compressor,
    VGG16Compressor,
)
from src.addons.extraction.descriptor import (
    AKAZEDescriptor,
    ColorHistogramDescriptor,
    DominantColorDescriptor,
    GaborDescriptor,
    HOGDescriptor,
    LBPDescriptor,
    ORBDescriptor,
    RGBHistogramDescriptor,
    SIFTDescriptor,
)


# ---------------------------------------------------------------------------
# Combined extractor (default produksi — ResNet50 + ColorHist + LBP)
# ---------------------------------------------------------------------------

class CombinedExtractor:
    """
    Menggabungkan ResNet50 (deep) + ColorHistogram + LBP dengan bobot:
      deep: 70%, color: 20%, texture: 10%

    Ini adalah extractor default untuk produksi karena memberikan
    keseimbangan terbaik antara akurasi dan kecepatan.
    """

    def __init__(self):
        self._deep    = ResNet50Compressor()
        self._color   = ColorHistogramDescriptor()
        self._texture = LBPDescriptor()

    def extract(self, image_path: str) -> np.ndarray:
        deep_feat    = self._deep.extract(image_path)      # (2048,)
        color_feat   = self._color.extract(image_path)     # (512,)
        texture_feat = self._texture.extract(image_path)   # (256,)

        combined = np.concatenate([
            deep_feat    * 0.70,
            color_feat   * 0.20,
            texture_feat * 0.10,
        ])
        return combined.astype(np.float32)  # (2816,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# Ultra Combined extractor — semua fitur digabung (PALING LENGKAP)
# ---------------------------------------------------------------------------

class UltraCombinedExtractor:
    """
    Gabungan SEMUA descriptor yang tersedia dengan bobot optimal.

    Komposisi vektor:
      - ResNet50   (2048) × 0.40  → semantic deep features
      - EfficientNet (1280) × 0.20 → efficient deep features
      - VGG16      (4096) × 0.10  → legacy deep features
      - HOG        (~1764) × 0.10  → shape & edge
      - ColorHist  (512)  × 0.10  → HSV color distribution
      - RGBHist    (768)  × 0.05  → raw RGB color
      - Gabor      (80)   × 0.05  → texture
      - LBP        (256)  × 0.05  → micro-texture
      - SIFT       (256)  × 0.025 → keypoint
      - AKAZE      (256)  × 0.015 → keypoint robust
      - DominantColor (48) × 0.01 → color palette

    Total ≈ 11364-dim (bervariasi tergantung HOG windowing)
    """

    def __init__(self):
        self._resnet    = ResNet50Compressor()
        self._effnet    = EfficientNetCompressor()
        self._vgg       = VGG16Compressor()
        self._hog       = HOGDescriptor()
        self._color_hsv = ColorHistogramDescriptor()
        self._color_rgb = RGBHistogramDescriptor()
        self._gabor     = GaborDescriptor()
        self._lbp       = LBPDescriptor()
        self._sift      = SIFTDescriptor()
        self._akaze     = AKAZEDescriptor()
        self._dominant  = DominantColorDescriptor()

    def extract(self, image_path: str) -> np.ndarray:
        resnet_f   = self._resnet.extract(image_path)    * 0.40
        effnet_f   = self._effnet.extract(image_path)    * 0.20
        vgg_f      = self._vgg.extract(image_path)       * 0.10
        hog_f      = self._hog.extract(image_path)       * 0.10
        color_hsv  = self._color_hsv.extract(image_path) * 0.10
        color_rgb  = self._color_rgb.extract(image_path) * 0.05
        gabor_f    = self._gabor.extract(image_path)     * 0.05
        lbp_f      = self._lbp.extract(image_path)       * 0.05
        sift_f     = self._sift.extract(image_path)      * 0.025
        akaze_f    = self._akaze.extract(image_path)     * 0.015
        dominant_f = self._dominant.extract(image_path)  * 0.010

        combined = np.concatenate([
            resnet_f, effnet_f, vgg_f, hog_f, color_hsv, color_rgb,
            gabor_f, lbp_f, sift_f, akaze_f, dominant_f,
        ])
        return combined.astype(np.float32)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# Registry — semua extractor tersedia via get_extractor()
# ---------------------------------------------------------------------------

extractors: dict[str, type] = {
    # ── Deep Learning ─────────────────────────────────────────────────────
    "resnet50"       : ResNet50Compressor,
    "efficientnet"   : EfficientNetCompressor,
    "vgg16"          : VGG16Compressor,
    "deep"           : ResNet50Compressor,       # alias

    # ── Traditional CV ────────────────────────────────────────────────────
    "akaze"          : AKAZEDescriptor,
    "orb"            : ORBDescriptor,
    "sift"           : SIFTDescriptor,
    "hog"            : HOGDescriptor,
    "gabor"          : GaborDescriptor,

    # ── Color Features ────────────────────────────────────────────────────
    "color_histogram": ColorHistogramDescriptor,
    "color"          : ColorHistogramDescriptor,  # alias HSV
    "rgb_histogram"  : RGBHistogramDescriptor,
    "dominant_color" : DominantColorDescriptor,

    # ── Texture ───────────────────────────────────────────────────────────
    "lbp"            : LBPDescriptor,

    # ── Combined ──────────────────────────────────────────────────────────
    "combined"       : CombinedExtractor,        # default produksi (cepat)
    "ultra"          : UltraCombinedExtractor,   # paling lengkap (lambat)
}


def get_extractor(method: str = "combined"):
    """
    Kembalikan instance extractor berdasarkan nama metode.

    Args:
        method: Nama metode dari `extractors` dict.

    Returns:
        Instance extractor yang sudah siap digunakan.

    Raises:
        ValueError: Jika nama metode tidak dikenal.
    """
    if method not in extractors:
        raise ValueError(
            f"Metode '{method}' tidak dikenal. "
            f"Pilihan: {list(extractors.keys())}"
        )
    return extractors[method]()
