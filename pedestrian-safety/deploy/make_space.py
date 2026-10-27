"""
DEPLOY - put the dashboard online as a Hugging Face Space (free)
----------------------------------------------------------------
Step 1 (build):  python deploy/make_space.py
    Copies only what the website needs into deploy/space/ :
    code, trained models, a slimmed sensor table, 12 demo photos, charts and tables.
    (No WiderPerson images, crops or Melbourne CSV - those stay on your laptop.)

Step 2 (upload): python deploy/make_space.py --push
    Logs in with your Hugging Face token, creates the Space, uploads deploy/space/
    and stores GEMINI_API_KEY from your .env as a hidden Space secret.

Options:  --name crowd-safety   Space name (default)
          --private             only you can open it
"""
import argparse
import getpass
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (MODELS_DIR, IMAGE_MODELS_DIR, SENSOR_FEATURES_FILE, EVENTS_CSV, CAMERAS_CSV,
                    FIGURES_DIR, TABLES_DIR, LAG_HOURS, YOLO_WEIGHTS)

HERE = Path(__file__).resolve().parent
OUT = HERE / "space"

CODE = ["config.py", "agent/__init__.py", "agent/agent.py", "agent/tools.py", "agent/rules.py",
        "agent/evaluate_agent.py", "agent/test_scenarios.json",
        "sensor_pipeline/__init__.py", "sensor_pipeline/common.py",
        "image_pipeline/__init__.py", "image_pipeline/img_common.py"]
REQUIRED_MODELS = ["footfall_model.joblib", "level_thresholds.csv", "sensor_baseline.parquet"]
OPTIONAL_MODELS = ["warning_threshold.json", "sensors.csv"]
# the only sensor-table columns the website reads (agent/tools.py)
SENSOR_COLS = ["ts", "sensor_id", "sensor_name", "hour", "weekday", "month", "is_weekend", "is_holiday", "is_event",
               "event_type"] + [f"lag_{L}" for L in LAG_HOURS] + ["trend", "baseline", "surge_so_far", "count", "level"]


def mb(p):
    p = Path(p)
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return n / 1e6


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build():
    import pandas as pd

    missing = [n for n in REQUIRED_MODELS if not (MODELS_DIR / n).exists()]
    for f in (IMAGE_MODELS_DIR / "heads.pt", SENSOR_FEATURES_FILE):
        if not f.exists():
            missing.append(str(f.relative_to(ROOT)))
    if missing:
        sys.exit("Missing (run the earlier phases first): " + ", ".join(missing))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print("1/6 code")
    for f in CODE:
        copy(ROOT / f, OUT / f)
    shutil.copytree(ROOT / "app", OUT / "app", ignore=shutil.ignore_patterns("uploads", "__pycache__"))
    for f in ("Dockerfile", "requirements-space.txt", "warmup.py"):
        copy(HERE / f, OUT / f)
    copy(HERE / "space_README.md", OUT / "README.md")

    print("2/6 models")
    for n in REQUIRED_MODELS + OPTIONAL_MODELS:
        if (MODELS_DIR / n).exists():
            copy(MODELS_DIR / n, OUT / "saved_models" / n)
    copy(IMAGE_MODELS_DIR / "heads.pt", OUT / "saved_models" / "image" / "heads.pt")
    if YOLO_WEIGHTS.exists():
        copy(YOLO_WEIGHTS, OUT / "saved_models" / YOLO_WEIGHTS.name)

    print("3/6 sensor table (only the columns the website uses)")
    df = pd.read_parquet(SENSOR_FEATURES_FILE, columns=SENSOR_COLS)
    dst = OUT / "data" / "processed" / SENSOR_FEATURES_FILE.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False, compression="zstd")
    print(f"    {len(df):,} rows, {mb(SENSOR_FEATURES_FILE):.0f} MB -> {mb(dst):.0f} MB")

    print("4/6 calendar + cameras")
    for f in (EVENTS_CSV, CAMERAS_CSV):
        if f.exists():
            copy(f, OUT / "data" / f.name)

    print("5/6 demo photos (12 most crowded WiderPerson test images)")
    from agent.evaluate_agent import crowded_images
    photos = crowded_images(12)
    for p in photos:
        copy(Path(p), OUT / "data" / "samples" / Path(p).name)
    if not photos:
        print("    WiderPerson not found: the website will only accept uploaded photos")

    print("6/6 results")
    if FIGURES_DIR.exists():
        for p in FIGURES_DIR.glob("*.png"):
            copy(p, OUT / "results" / "figures" / p.name)
    if TABLES_DIR.exists():
        for p in list(TABLES_DIR.glob("*.csv")) + list(TABLES_DIR.glob("*.json")) + list(TABLES_DIR.glob("*.txt")):
            copy(p, OUT / "results" / "tables" / p.name)

    print(f"\nReady: {OUT}  ({mb(OUT):.0f} MB)")
    for p in sorted(OUT.iterdir()):
        print(f"    {p.name:<24}{mb(p):8.1f} MB")


def read_env():
    env = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def push(name, private):
    from huggingface_hub import HfApi

    if not OUT.exists():
        sys.exit("Build first:  python deploy/make_space.py")
    token = os.environ.get("HF_TOKEN") or getpass.getpass("Paste your Hugging Face token (Write access; it stays hidden): ").strip()
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = f"{user}/{name}"

    print(f"Creating Space {repo} ...")
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=private, exist_ok=True)

    env = read_env()
    key = env.get("GEMINI_API_KEY", "")
    if key and key != "paste-your-key-here":
        api.add_space_secret(repo, "GEMINI_API_KEY", key)
        print("    GEMINI_API_KEY saved as a hidden Space secret")
    else:
        print("    No GEMINI_API_KEY in .env: the agent will use the offline plan")
    api.add_space_variable(repo, "GEMINI_MODEL", env.get("GEMINI_MODEL") or "gemini-3.8-flash")

    print(f"Uploading {mb(OUT):.0f} MB (big files go up in parts, give it a few minutes) ...")
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=OUT,
                      commit_message="Deploy Crowd Safety Console")
    print(f"\nDone. Build log + website:  https://huggingface.co/spaces/{repo}")
    print(f"Direct link (after the build): https://{user.lower()}-{name.lower()}.hf.space")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upload deploy/space to Hugging Face")
    ap.add_argument("--name", default="crowd-safety", help="Space name")
    ap.add_argument("--private", action="store_true", help="only you can open the Space")
    a = ap.parse_args()
    if a.push:
        push(a.name, a.private)
    else:
        build()
