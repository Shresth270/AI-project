"""
PHASE 2A - Train and compare the 8 sensor models
------------------------------------------------
Question each model answers:
    "At the end of hour t-1, what crowd level will hour t have?"  (Low / Medium / High / Very High)

Steps
  1. Split by time:  fit (older 85% of train)  |  val (newest 15% of train)  |  test (Phase-1 test)
  2. Take a stratified sample of the fit period (a laptop can't SMOTE 1 million rows quickly)
  3. Scale numeric features (StandardScaler) - fitted on the fit sample only
  4. SMOTENC on the fit sample only -> every level gets the same number of rows
  5. Train 8 models, pick the best on VAL (macro F1), report everything on TEST
  6. Two simple rules are scored too, so we can show the models beat common sense:
       "Same as last hour"  and  "Usual for this time slot"

Outputs
  saved_models/footfall_model.joblib          best model + scaler + feature list
  results/tables/sensor_model_comparison.csv  all metrics
  results/tables/sensor_best_report.txt       per-class report of the best model
  results/figures/sensor_model_comparison.png, sensor_confusion_best.png, sensor_feature_importance.png

Run:  python sensor_pipeline/02_train_compare.py          (about 10-25 min on a laptop)
      python sensor_pipeline/02_train_compare.py --quick  (small sample, ~3 min, for a dry run)
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import SENSOR_FEATURES_FILE, MODELS_DIR, TABLES_DIR, FIGURES_DIR, FOOTFALL_LEVELS, RANDOM_SEED
from sensor_pipeline.common import FEATURES, CATEGORICAL, load_thresholds, add_features, level_from_count, SoftVote

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
QUICK = "--quick" in sys.argv
FIT_SAMPLE = 40_000 if QUICK else 150_000     # rows sampled from the fit period before SMOTE
VAL_SHARE = 0.15                              # newest part of the train period used for model selection
VH = 3                                        # index of "Very High"


# ======================================================================== data
def load_data():
    df = pd.read_parquet(SENSOR_FEATURES_FILE)
    th = load_thresholds()
    df = add_features(df[df["sensor_id"].isin(th.index)], th)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    cut = train["ts"].min() + (train["ts"].max() - train["ts"].min()) * (1 - VAL_SHARE)
    fit, val = train[train["ts"] < cut], train[train["ts"] >= cut]
    print(f"fit : {fit['ts'].min().date()} -> {fit['ts'].max().date()}  {len(fit):>9,} rows")
    print(f"val : {val['ts'].min().date()} -> {val['ts'].max().date()}  {len(val):>9,} rows")
    print(f"test: {test['ts'].min().date()} -> {test['ts'].max().date()}  {len(test):>9,} rows")

    # stratified sample of the fit period
    frac = min(1.0, FIT_SAMPLE / len(fit))
    sample = fit.groupby("level", group_keys=False).sample(frac=frac, random_state=RANDOM_SEED)
    return sample, val, test, th


def prepare(sample, val, test):
    num_cols = [c for c in FEATURES if c not in CATEGORICAL]
    scaler = StandardScaler().fit(sample[num_cols])

    def to_X(d):
        X = d[FEATURES].astype("float32").copy()
        X[num_cols] = scaler.transform(d[num_cols])
        return X

    X_fit, y_fit = to_X(sample), sample["level"].to_numpy()
    print(f"\nBefore SMOTE: {np.bincount(y_fit, minlength=4).tolist()}")
    smote = SMOTENC(categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
                    random_state=RANDOM_SEED, k_neighbors=5)
    X_fit, y_fit = smote.fit_resample(X_fit, y_fit)
    print(f"After  SMOTE: {np.bincount(y_fit, minlength=4).tolist()}  (train only; val/test untouched)")
    return X_fit, y_fit, to_X(val), to_X(test), scaler


# ======================================================================== models
def build_models():
    rs, nj = RANDOM_SEED, -1
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=nj, random_state=rs),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.08, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=nj, random_state=rs),
        "CatBoost": CatBoostClassifier(iterations=500, depth=7, learning_rate=0.1, verbose=0,
                                       thread_count=nj, random_seed=rs),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=rs),
        "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200,
                                       learning_rate=0.5, random_state=rs),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=20, min_samples_leaf=2), n_estimators=50,
                                     max_samples=0.5, n_jobs=nj, random_state=rs),
    }


def build_stacking():
    rs = RANDOM_SEED
    base = [
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=rs)),
        ("xgb", XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08, tree_method="hist",
                              n_jobs=-1, random_state=rs)),
        ("hgb", HistGradientBoostingClassifier(max_iter=200, random_state=rs)),
    ]
    return StackingClassifier(base, final_estimator=LogisticRegression(max_iter=1000), cv=3,
                              stack_method="predict_proba")


# ======================================================================== metrics
def score(name, y_true, y_pred, split, secs=None):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(4), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"model": name, "split": split, "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
            "very_high_precision": p[VH], "very_high_recall": r[VH], "very_high_f1": f[VH],
            "train_seconds": secs}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if QUICK:
        print("[QUICK MODE] small sample - results are only for a dry run\n")

    sample, val, test, th = load_data()
    X_fit, y_fit, X_val, X_test, scaler = prepare(sample, val, test)
    y_val, y_test = val["level"].to_numpy(), test["level"].to_numpy()

    rows, fitted = [], {}

    # ---- two common-sense rules (no learning)
    for name, col in [("Rule: same as last hour", "lag_1"), ("Rule: usual for this slot", "baseline")]:
        for split, d, y in [("val", val, y_val), ("test", test, y_test)]:
            rows.append(score(name, y, level_from_count(d[col], d["sensor_id"], th), split))

    # ---- 6 single models + soft voting + stacking
    models = build_models()
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_fit, y_fit)
        secs = time.time() - t0
        fitted[name] = model
        for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            rows.append(score(name, y, np.asarray(model.predict(X)).ravel().astype(int), split, secs))
        v = rows[-2]
        print(f"{name:<20} {secs:6.0f}s   val macro-F1 {v['macro_f1']:.3f}   val Very-High recall {v['very_high_recall']:.3f}")

    vote = SoftVote({k: fitted[k] for k in ["Random Forest", "XGBoost", "CatBoost"]})
    fitted["Voting (soft)"] = vote
    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        rows.append(score("Voting (soft)", y, vote.predict(X), split, 0.0))
    print(f"{'Voting (soft)':<20} {'reuse':>6}    val macro-F1 {rows[-2]['macro_f1']:.3f}")

    t0 = time.time()
    stack = build_stacking().fit(X_fit, y_fit)
    secs = time.time() - t0
    fitted["Stacking"] = stack
    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        rows.append(score("Stacking", y, stack.predict(X), split, secs))
    print(f"{'Stacking':<20} {secs:6.0f}s   val macro-F1 {rows[-2]['macro_f1']:.3f}")

    # ---- table
    res = pd.DataFrame(rows)
    wide = res.pivot(index="model", columns="split")
    table = pd.DataFrame({
        "val_macro_f1": wide[("macro_f1", "val")],
        "test_accuracy": wide[("accuracy", "test")],
        "test_macro_precision": wide[("macro_precision", "test")],
        "test_macro_recall": wide[("macro_recall", "test")],
        "test_macro_f1": wide[("macro_f1", "test")],
        "test_very_high_precision": wide[("very_high_precision", "test")],
        "test_very_high_recall": wide[("very_high_recall", "test")],
        "test_very_high_f1": wide[("very_high_f1", "test")],
        "train_seconds": wide[("train_seconds", "test")],
    }).sort_values("val_macro_f1", ascending=False).round(4)
    table.to_csv(TABLES_DIR / "sensor_model_comparison.csv")

    ml = table[~table.index.str.startswith("Rule")]
    best_name = ml.index[0]
    best = fitted[best_name]
    print("\n" + table.drop(columns=["train_seconds"]).to_string())
    print(f"\nBEST (chosen on validation macro-F1): {best_name}")

    # ---- best model details
    pred = np.asarray(best.predict(X_test)).ravel().astype(int)
    report = classification_report(y_test, pred, target_names=FOOTFALL_LEVELS, digits=3, zero_division=0)
    (TABLES_DIR / "sensor_best_report.txt").write_text(f"Best model: {best_name}\n\n{report}", encoding="utf-8")
    print(report)

    joblib.dump({"name": best_name, "model": best, "scaler": scaler, "features": FEATURES,
                 "categorical": CATEGORICAL, "levels": FOOTFALL_LEVELS}, MODELS_DIR / "footfall_model.joblib")
    print(f"Saved: {MODELS_DIR / 'footfall_model.joblib'}")

    imp_model = best if hasattr(best, "feature_importances_") else fitted["XGBoost"]
    imp_name = best_name if imp_model is best else "XGBoost"
    make_figures(table, confusion_matrix(y_test, pred, labels=range(4)), best_name,
                 pd.Series(imp_model.feature_importances_, index=FEATURES), imp_name)


# ======================================================================== figures
def make_figures(table, cm, best_name, importances, imp_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1) comparison: test macro-F1 and Very-High recall, side by side (same 0-1 scale)
    t = table.sort_values("test_macro_f1")
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(t) + 1.4), sharey=True)
    for ax, col, title in [(axes[0], "test_macro_f1", "Macro F1 (test)"),
                           (axes[1], "test_very_high_recall", "Very-High recall (test)")]:
        colors = [GRAY if n.startswith("Rule") else (ORANGE if n == best_name else BLUE) for n in t.index]
        ax.barh(y, t[col], color=colors, height=0.62)
        for yi, v in zip(y, t[col]):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, t.index)
    fig.suptitle(f"Sensor models vs simple rules  (orange = best: {best_name}, gray = rules)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensor_model_comparison.png", dpi=200)
    plt.close(fig)

    # 2) confusion matrix of the best model (row-normalised)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(4), FOOTFALL_LEVELS)
    ax.set_yticks(range(4), FOOTFALL_LEVELS)
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("Actual level")
    ax.set_title(f"Confusion matrix on test data: {best_name}", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensor_confusion_best.png", dpi=200)
    plt.close(fig)

    # 3) feature importance
    imp = (importances / importances.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.62)
    ax.set_xlabel("share of importance")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"What the model looks at - {imp_name}", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensor_feature_importance.png", dpi=200)
    plt.close(fig)
    print(f"Figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
