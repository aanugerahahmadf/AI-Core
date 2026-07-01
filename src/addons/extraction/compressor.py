from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Shared transform pipeline
# ---------------------------------------------------------------------------

_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return _IMAGENET_TRANSFORM(img).unsqueeze(0)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# ResNet50 — 2048-dim
# ---------------------------------------------------------------------------

class ResNet50Compressor:
    """
    ResNet50 pre-trained on ImageNet.
    Menghapus layer klasifikasi terakhir → embedding 2048-dim.
    """

    def __init__(self):
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self._extractor = nn.Sequential(*list(base.children())[:-1])
        self._extractor.eval()

    def extract(self, image_path: str) -> np.ndarray:
        tensor = _load_image(image_path)
        with torch.no_grad():
            feat = self._extractor(tensor)
        return feat.squeeze().numpy().astype(np.float32)  # (2048,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# EfficientNet-B0 — 1280-dim
# ---------------------------------------------------------------------------

class EfficientNetCompressor:
    """
    EfficientNet-B0 pre-trained on ImageNet.
    Menggunakan avgpool output → embedding 1280-dim.
    """

    def __init__(self):
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Hapus classifier, pertahankan features + avgpool
        self._features  = base.features
        self._avgpool   = base.avgpool
        self._features.eval()
        self._avgpool.eval()

    def extract(self, image_path: str) -> np.ndarray:
        tensor = _load_image(image_path)
        with torch.no_grad():
            x = self._features(tensor)
            x = self._avgpool(x)
        return x.squeeze().numpy().astype(np.float32)  # (1280,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)


# ---------------------------------------------------------------------------
# VGG16 — 4096-dim
# ---------------------------------------------------------------------------

class VGG16Compressor:
    """
    VGG16 pre-trained on ImageNet.
    Menggunakan output layer fc2 (4096-dim) sebagai embedding.
    """

    def __init__(self):
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # features + avgpool + classifier[0..4] (fc1 + relu + dropout + fc2)
        self._features = base.features
        self._avgpool  = base.avgpool
        self._fc       = nn.Sequential(*list(base.classifier.children())[:4])
        self._features.eval()
        self._avgpool.eval()
        self._fc.eval()

    def extract(self, image_path: str) -> np.ndarray:
        tensor = _load_image(image_path)
        with torch.no_grad():
            x = self._features(tensor)
            x = self._avgpool(x)
            x = torch.flatten(x, 1)
            x = self._fc(x)
        return x.squeeze().numpy().astype(np.float32)  # (4096,)

    def __call__(self, image_path: str) -> np.ndarray:
        return self.extract(image_path)
