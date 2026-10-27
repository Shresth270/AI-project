"""
PHASE 1A - Make person crops from WiderPerson
---------------------------------------------
WiderPerson gives boxes around people. Our image models classify SMALL CROPS,
so this script cuts every usable box out into its own image:

    data/processed/crops/<split>/<class>/<image_id>_<n>.jpg
        split : train | val | test
        class : pedestrian | rider | partially_visible | background

Splits are made by IMAGE (never by crop), so crops of the same photo never
appear in both train and test.
    WiderPerson train.txt -> our train (85%) + val (15%)
    WiderPerson val.txt   -> our test   (the official test set has no labels)

Run:  python image_pipeline/01_make_crops.py
"""
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (WIDERPERSON_DIR, CROPS_DIR, TABLES_DIR, WIDERPERSON_LABELS, IMAGE_CLASSES,
                    MIN_CROP_SIZE, CROP_PAD, CROP_SAVE_MAX_SIDE, VAL_FRACTION_OF_TRAIN,
                    CROP_CAPS, BACKGROUND_PER_IMAGE, RANDOM_SEED)

random.seed(RANDOM_SEED)


# ----------------------------------------------------------------- locate dataset
def find_root(base: Path) -> Path:
    if (base / "Images").is_dir():
        return base
    for p in base.rglob("Images"):
        if p.is_dir() and (p.parent / "Annotations").is_dir():
            return p.parent
    sys.exit(f"[STOP] Images/ and Annotations/ not found under {base}")


def read_ids(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def read_boxes(ann_file: Path):
    """Return list of (label, x1, y1, x2, y2). First line of the file is the count."""
    boxes = []
    for row in ann_file.read_text().split("\n")[1:]:
        parts = row.split()
        if len(parts) == 5:
            lab = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:])
            boxes.append((lab, x1, y1, x2, y2))
    return boxes


def overlaps(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


# ----------------------------------------------------------------- crop helpers
def padded(box, w, h):
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    return (max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px)), min(h, int(y2 + py)))


def save_crop(img, box, out_path: Path):
    crop = img.crop(box)
    crop.thumbnail((CROP_SAVE_MAX_SIDE, CROP_SAVE_MAX_SIDE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, "JPEG", quality=90)


def background_boxes(all_boxes, w, h, person_sizes, n):
    """Random patches, person-shaped, that touch no annotated box (any label)."""
    found, tries = [], 0
    while len(found) < n and tries < 60:
        tries += 1
        bw, bh = random.choice(person_sizes) if person_sizes else (40, 100)
        bw, bh = int(bw), int(bh)
        if bw >= w or bh >= h or min(bw, bh) < MIN_CROP_SIZE:
            continue
        x1, y1 = random.randint(0, w - bw), random.randint(0, h - bh)
        cand = (x1, y1, x1 + bw, y1 + bh)
        if not any(overlaps(cand, b[1:]) for b in all_boxes) and not any(overlaps(cand, f) for f in found):
            found.append(cand)
    return found


# ----------------------------------------------------------------- main
def main():
    root = find_root(WIDERPERSON_DIR)
    img_dir, ann_dir = root / "Images", root / "Annotations"
    train_ids, test_ids = read_ids(root / "train.txt"), read_ids(root / "val.txt")
    random.shuffle(train_ids)
    n_val = int(len(train_ids) * VAL_FRACTION_OF_TRAIN)
    splits = {"train": train_ids[n_val:], "val": train_ids[:n_val], "test": test_ids}
    print(f"Images per split: { {k: len(v) for k, v in splits.items()} }")

    # pass 1: collect every candidate crop (no image loading yet)
    candidates = {s: defaultdict(list) for s in splits}   # split -> class -> [(img_id, box)]
    person_sizes = []
    image_boxes = {}
    for split, ids in splits.items():
        for img_id in ids:
            ann = ann_dir / f"{img_id}.jpg.txt"
            if not ann.exists():
                continue
            boxes = read_boxes(ann)
            image_boxes[img_id] = boxes
            for lab, x1, y1, x2, y2 in boxes:
                if lab in WIDERPERSON_LABELS and min(x2 - x1, y2 - y1) >= MIN_CROP_SIZE:
                    candidates[split][WIDERPERSON_LABELS[lab]].append((img_id, (x1, y1, x2, y2)))
                    if lab == 1:
                        person_sizes.append((x2 - x1, y2 - y1))

    # pass 2: cap the common classes, keep all of the rare ones
    chosen = defaultdict(list)                              # img_id -> [(split, class, box)]
    for split in splits:
        for cls, items in candidates[split].items():
            random.shuffle(items)
            for img_id, box in items[:CROP_CAPS[split]]:
                chosen[img_id].append((split, cls, box))
    bg_needed = {s: CROP_CAPS[s] for s in splits}
    split_of = {i: s for s, ids in splits.items() for i in ids}

    # pass 3: open each image once, save its crops (+ background patches)
    rows, counter = [], Counter()
    all_ids = [i for ids in splits.values() for i in ids if i in image_boxes]
    for img_id in tqdm(all_ids, desc="Cropping"):
        split = split_of[img_id]
        jobs = chosen.get(img_id, [])
        want_bg = bg_needed[split] > 0
        if not jobs and not want_bg:
            continue
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        w, h = img.size
        if want_bg:
            for box in background_boxes(image_boxes[img_id], w, h, person_sizes, BACKGROUND_PER_IMAGE):
                if bg_needed[split] <= 0:
                    break
                jobs.append((split, "background", box))
                bg_needed[split] -= 1
        for n, (sp, cls, box) in enumerate(jobs):
            out = CROPS_DIR / sp / cls / f"{img_id}_{n}.jpg"
            pbox = padded(box, w, h) if cls != "background" else tuple(map(int, box))
            if pbox[2] - pbox[0] < 2 or pbox[3] - pbox[1] < 2:
                continue
            save_crop(img, pbox, out)
            rows.append({"path": out.relative_to(CROPS_DIR).as_posix(), "split": sp, "class": cls,
                         "image_id": img_id, "box_w": int(box[2] - box[0]), "box_h": int(box[3] - box[1])})
            counter[(sp, cls)] += 1

    index = pd.DataFrame(rows)
    index.to_csv(CROPS_DIR / "crops_index.csv", index=False)
    table = (pd.Series(counter).rename("crops").rename_axis(["split", "class"]).reset_index()
             .pivot(index="class", columns="split", values="crops").reindex(IMAGE_CLASSES)
             .fillna(0).astype(int)[["train", "val", "test"]])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "crop_counts.csv")
    print("\nCrops saved per class and split:")
    print(table.to_string())
    print(f"\nIndex: {CROPS_DIR / 'crops_index.csv'}")


if __name__ == "__main__":
    main()
