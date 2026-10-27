"""
Shared image helpers: backbones, preprocessing and the classifier head.
Used by Phase 3 (training) and later by the agent's classify tool, so an image
is always prepared in exactly the same way.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BACKBONES


class PadToSquare:
    """Person crops are tall and thin. A plain centre-crop would cut off heads and feet,
    so the crop is padded to a square first (gray borders), then resized."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        return ImageOps.pad(img, (side, side), color=(124, 116, 104))


def load_backbone(name: str, pretrained: bool = True):
    """Return (model, transform). Model outputs a feature vector (no classification layer)."""
    import timm
    from timm.data import resolve_model_data_config, create_transform
    from torchvision import transforms as T

    model = timm.create_model(BACKBONES[name], pretrained=pretrained, num_classes=0).eval()
    cfg = resolve_model_data_config(model)
    cfg["crop_pct"] = 1.0                          # image is already square: keep all of it
    base = create_transform(**cfg, is_training=False)
    transform = T.Compose([PadToSquare(), base])
    return model, transform


def augment_transform(transform):
    """Training-time copy: horizontal flip + brightness change (as in the methodology)."""
    from torchvision import transforms as T
    return T.Compose([T.RandomHorizontalFlip(p=1.0), T.ColorJitter(brightness=0.3), transform])


class Head(nn.Module):
    """Small classifier on top of frozen features: dense -> ReLU -> dropout -> dense."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 512, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
