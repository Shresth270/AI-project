"""
Central settings for the whole project.
Every script imports paths and constants from here, so nothing is hard-coded twice.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
WIDERPERSON_DIR = DATA_DIR / "widerperson"      # Images/, Annotations/, train.txt, val.txt
MELBOURNE_DIR = DATA_DIR / "melbourne"          # the hourly counts CSV goes here
EVENTS_CSV = DATA_DIR / "events.csv"            # events calendar (built in Phase 1)
CAMERAS_CSV = DATA_DIR / "cameras.csv"          # camera registry (built in Phase 5)
SAMPLES_DIR = DATA_DIR / "samples"              # bundled demo photos (used when WiderPerson is not present, e.g. online)

PROCESSED_DIR = DATA_DIR / "processed"
# person crops, one folder per class (PS_CROPS_DIR lets Kaggle point at its read-only input folder)
CROPS_DIR = Path(os.environ.get("PS_CROPS_DIR", PROCESSED_DIR / "crops"))
FEATURES_DIR = PROCESSED_DIR / "features"       # saved backbone fingerprints (.npy)

MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---------------------------------------------------------------- image pipeline
# WiderPerson annotation labels -> our crop classes
#   1 pedestrians, 2 riders, 3 partially-visible persons, 4 ignore regions, 5 crowd
WIDERPERSON_LABELS = {1: "pedestrian", 2: "rider", 3: "partially_visible"}
IMAGE_CLASSES = ["pedestrian", "rider", "partially_visible", "background"]
MIN_CROP_SIZE = 24          # skip boxes smaller than this (pixels, either side)
CROP_PAD = 0.10             # add 10% context around each box
CROP_SAVE_MAX_SIDE = 256    # crops are saved with longest side <= this (saves disk)
VAL_FRACTION_OF_TRAIN = 0.15  # part of WiderPerson train images held out for validation
# max crops per class per split (pedestrians are very common; riders are rare and all kept)
CROP_CAPS = {"train": 6000, "val": 1200, "test": 1500}
BACKGROUND_PER_IMAGE = 2    # random empty patches sampled per image
RANDOM_SEED = 42

# pretrained ImageNet weights from the timm library (each model's own input size is used:
# 224 px for VGG16 / ResNet50 / ViT, 299 px for InceptionV3 / Xception)
BACKBONES = {
    "vgg16": "vgg16.tv_in1k",
    "resnet50": "resnet50.a1_in1k",
    "inception_v3": "inception_v3.tv_in1k",
    "xception": "legacy_xception.tf_in1k",
    "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}
HYBRIDS = [("vit", "xception"), ("vit", "inception_v3"), ("xception", "inception_v3")]
IMAGE_MODELS_DIR = MODELS_DIR / "image"     # trained heads + feature statistics

# ---------------------------------------------------------------- sensor pipeline
# crowd level for the NEXT hour, by percentile of that sensor's own history
LEVEL_PERCENTILES = {"Low": 0.25, "Medium": 0.75, "High": 0.95}   # above 0.95 = Very High
FOOTFALL_LEVELS = ["Low", "Medium", "High", "Very High"]
TEST_FRACTION = 0.2         # split by DATE: the last 20% of the time span is the test set
# Use post-lockdown data (Melbourne's 2020-21 lockdowns make those years unlike normal life).
# If the file has less than a year after this date, the script falls back to PRE_COVID_WINDOW.
START_DATE = "2022-01-01"
PRE_COVID_WINDOW = ("2014-01-01", "2020-02-29")
MIN_SENSOR_DAYS = 180       # drop sensors with less history than this
LAG_HOURS = [1, 2, 3, 24, 168]   # past counts used as features (168 h = same hour last week)
SENSOR_FEATURES_FILE = PROCESSED_DIR / "sensor_features.parquet"

# ---------------------------------------------------------------- risk rules (tunable)
RISK_RULES = {
    "act_density": 4.0,             # people per m2 -> Act now
    "act_density_tight": 3.0,       # people per m2 ...
    "act_hidden_share": 0.40,       # ... combined with this share partly hidden -> Act now
    "watch_density": 2.0,           # people per m2 -> Watch
    "watch_surge_ratio": 1.5,       # crowd vs normal for that slot -> Watch
}

# ---------------------------------------------------------------- agent
# LLM: Google Gemini (free tier). Key and model name go in .env:
#   GEMINI_API_KEY=...        GEMINI_MODEL=gemini-3.8-flash
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
YOLO_WEIGHTS = MODELS_DIR / "yolo11s.pt"      # person detector (downloads automatically, ~19 MB)
YOLO_IMG_SIZE = 1280                          # large input so small, far-away people are found
DEFAULT_CAMERA_AREA_M2 = 200                  # visible ground area if a camera has none set
UPLOADS_DIR = ROOT / "app" / "static" / "uploads"
AGENT_LOGS_DIR = RESULTS_DIR / "agent_logs"
