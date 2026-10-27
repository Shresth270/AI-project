"""
PHASE 0 - Data check
Looks at both datasets and writes a report to results/data_check.txt.
Run:  python 00_check_data.py
Then send results/data_check.txt so Phase 1 can be written for your exact files.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from config import WIDERPERSON_DIR, MELBOURNE_DIR, RESULTS_DIR

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


# ====================================================================== WiderPerson
def find_dir(base: Path, name: str):
    """Find a folder called `name` anywhere under base (handles nested zips)."""
    if (base / name).is_dir():
        return base / name
    hits = [p for p in base.rglob(name) if p.is_dir()]
    return hits[0] if hits else None


def check_widerperson():
    log("=" * 70)
    log("WIDERPERSON")
    log("=" * 70)
    img_dir = find_dir(WIDERPERSON_DIR, "Images")
    ann_dir = find_dir(WIDERPERSON_DIR, "Annotations")
    if img_dir is None or ann_dir is None:
        log(f"[MISSING] Could not find Images/ and Annotations/ under {WIDERPERSON_DIR}")
        log("          Contents found: " + ", ".join(p.name for p in WIDERPERSON_DIR.iterdir()))
        return
    root = img_dir.parent
    log(f"Root folder      : {root}")

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    anns = list(ann_dir.glob("*.txt"))
    log(f"Images           : {len(images):,}")
    log(f"Annotation files : {len(anns):,}")

    for split in ["train.txt", "val.txt", "test.txt"]:
        f = root / split
        if f.exists():
            ids = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            log(f"{split:<17}: {len(ids):,} ids (e.g. {ids[0] if ids else '-'})")
        else:
            log(f"{split:<17}: [MISSING]")

    # parse all annotations: first line = count, then "label x1 y1 x2 y2"
    labels, widths, heights, bad = Counter(), [], [], 0
    for f in anns:
        try:
            rows = f.read_text().split("\n")[1:]
            for r in rows:
                parts = r.split()
                if len(parts) != 5:
                    continue
                lab, x1, y1, x2, y2 = int(parts[0]), *map(float, parts[1:])
                labels[lab] += 1
                widths.append(x2 - x1)
                heights.append(y2 - y1)
        except Exception:
            bad += 1
    names = {1: "pedestrians", 2: "riders", 3: "partially-visible", 4: "ignore regions", 5: "crowd"}
    log("Boxes per label  :")
    for k in sorted(labels):
        log(f"   {k} {names.get(k, '?'):<18} {labels[k]:>9,}")
    if widths:
        s = pd.Series(widths), pd.Series(heights)
        log(f"Box width  median/p10/p90 : {s[0].median():.0f} / {s[0].quantile(.1):.0f} / {s[0].quantile(.9):.0f} px")
        log(f"Box height median/p10/p90 : {s[1].median():.0f} / {s[1].quantile(.1):.0f} / {s[1].quantile(.9):.0f} px")
        small = sum(1 for w, h in zip(widths, heights) if min(w, h) < 24)
        log(f"Boxes smaller than 24 px  : {small:,} ({small / len(widths):.0%})")
    if bad:
        log(f"Unreadable annotation files: {bad}")
    log("Sample annotation file:")
    if anns:
        for line in anns[0].read_text().splitlines()[:4]:
            log("   " + line)


# ====================================================================== Melbourne
def check_melbourne():
    log("")
    log("=" * 70)
    log("MELBOURNE PEDESTRIAN COUNTING")
    log("=" * 70)
    csvs = sorted(MELBOURNE_DIR.rglob("*.csv"))
    if not csvs:
        log(f"[MISSING] No CSV found under {MELBOURNE_DIR}")
        return
    for f in csvs:
        log(f"File   : {f.name}  ({f.stat().st_size / 1e6:,.1f} MB)")
        sample = pd.read_csv(f, nrows=2000, low_memory=False)
        log(f"Columns ({len(sample.columns)}):")
        for c in sample.columns:
            ex = sample[c].dropna().iloc[0] if sample[c].notna().any() else "-"
            log(f"   {c:<28} {str(sample[c].dtype):<9} e.g. {str(ex)[:40]}")

        cols = {c.lower(): c for c in sample.columns}
        date_col = next((cols[k] for k in cols if any(t in k for t in ("date", "time", "sensing"))), None)
        sensor_col = next((cols[k] for k in cols if any(t in k for t in ("sensor_name", "location_id", "sensor_id", "sensor"))), None)

        # full pass in chunks: row count, date range, sensors, missing values
        n, missing, sensors, dmin, dmax = 0, Counter(), set(), None, None
        for chunk in pd.read_csv(f, chunksize=500_000, low_memory=False):
            n += len(chunk)
            missing.update(chunk.isna().sum().to_dict())
            if sensor_col:
                sensors.update(chunk[sensor_col].dropna().unique().tolist())
            if date_col:
                d = pd.to_datetime(chunk[date_col], errors="coerce")
                lo, hi = d.min(), d.max()
                dmin = lo if dmin is None or (pd.notna(lo) and lo < dmin) else dmin
                dmax = hi if dmax is None or (pd.notna(hi) and hi > dmax) else dmax
        log(f"Rows       : {n:,}")
        log(f"Date column: {date_col}  range {dmin} -> {dmax}")
        log(f"Sensor col : {sensor_col}  unique sensors {len(sensors):,}")
        miss = {k: v for k, v in missing.items() if v}
        log(f"Missing    : {miss if miss else 'none'}")
        log("First 3 rows:")
        log(sample.head(3).to_string())
        log("")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    try:
        check_widerperson()
    except Exception as e:
        log(f"[ERROR] WiderPerson check failed: {e!r}")
    try:
        check_melbourne()
    except Exception as e:
        log(f"[ERROR] Melbourne check failed: {e!r}")
    out = RESULTS_DIR / "data_check.txt"
    out.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nReport saved to {out}")
    sys.exit(0)
