"""Runs once while the container is built: downloads the pretrained image backbones
and the YOLO person detector, so visitors don't wait for downloads."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from config import IMAGE_MODELS_DIR, YOLO_WEIGHTS
from image_pipeline.img_common import load_backbone

# a failed download never stops the build: the app downloads again on first use
ck = torch.load(IMAGE_MODELS_DIR / "heads.pt", map_location="cpu", weights_only=False)
for b in ck["backbones"]:
    try:
        load_backbone(b, pretrained=True)
        print("backbone ready:", b, flush=True)
    except Exception as e:
        print("backbone", b, "not downloaded now:", type(e).__name__, flush=True)

try:
    from ultralytics import YOLO
    os.chdir(YOLO_WEIGHTS.parent)
    YOLO(YOLO_WEIGHTS.name)
    print("YOLO ready")
except Exception as e:
    print("YOLO not downloaded now:", type(e).__name__)
print("warm-up done")
