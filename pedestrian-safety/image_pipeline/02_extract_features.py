"""
PHASE 3A - Extract "fingerprints" (feature vectors) with 5 pretrained backbones
-------------------------------------------------------------------------------
Each backbone looks at every crop ONCE and turns it into a list of numbers:
    VGG16 4096 | ResNet50 2048 | InceptionV3 2048 | Xception 2048 | ViT 768
These are saved to disk, so all later training takes minutes instead of hours.

The train split is processed twice: the original crops and an augmented copy
(horizontal flip + brightness change), giving the heads more variety to learn from.

Run on a GPU (Kaggle / Colab):   python image_pipeline/02_extract_features.py
Options:  --backbones vit xception     only these backbones
          --no-pretrained              random weights (pipeline test only)
          --limit 200                  only N crops per split (pipeline test only)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CROPS_DIR, FEATURES_DIR, IMAGE_CLASSES, BACKBONES, RANDOM_SEED
from image_pipeline.img_common import load_backbone, augment_transform, device


class Crops(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # crops_index.csv made on Windows uses "\\" - normalise so it also works on Linux (Kaggle)
        with Image.open(CROPS_DIR / self.paths[i].replace("\\", "/")) as im:
            return self.transform(im.convert("RGB"))


@torch.no_grad()
def extract(model, paths, transform, dev, batch=64, workers=2, desc=""):
    loader = DataLoader(Crops(paths, transform), batch_size=batch, num_workers=workers,
                        pin_memory=dev.type == "cuda")
    out = []
    for x in tqdm(loader, desc=desc, leave=False):
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            f = model(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES))
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    dev = device()
    print(f"Device: {dev}" + ("" if dev.type == "cuda" else "   [WARNING] no GPU - this will be very slow"))
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(CROPS_DIR / "crops_index.csv")
    index["label"] = index["class"].map({c: i for i, c in enumerate(IMAGE_CLASSES)})
    if args.limit:
        index = index.groupby(["split", "class"], group_keys=False).head(args.limit // 4)
    for split in ["train", "val", "test"]:
        part = index[index["split"] == split]
        np.save(FEATURES_DIR / f"labels_{split}.npy", part["label"].to_numpy())
        print(f"{split:<5}: {len(part):>6,} crops  {part['class'].value_counts().to_dict()}")

    for name in args.backbones:
        t0 = time.time()
        model, transform = load_backbone(name, pretrained=not args.no_pretrained)
        model = model.to(dev)
        jobs = [("train", transform), ("train_aug", augment_transform(transform)),
                ("val", transform), ("test", transform)]
        for tag, tf in jobs:
            split = tag.replace("_aug", "")
            paths = index.loc[index["split"] == split, "path"]
            feats = extract(model, paths, tf, dev, workers=args.workers, desc=f"{name}:{tag}")
            np.save(FEATURES_DIR / f"{name}_{tag}.npy", feats)
        print(f"{name:<13} dim {feats.shape[1]:>5}   done in {time.time() - t0:5.0f}s")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nFeatures saved in {FEATURES_DIR}")


if __name__ == "__main__":
    main()
