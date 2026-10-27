"""
PHASE 3B + 4 - Train the image classifiers on the saved fingerprints
--------------------------------------------------------------------
  5 single models   : VGG16, ResNet50, InceptionV3, Xception, ViT
  3 hybrid models   : fingerprints of two backbones joined side by side (feature fusion)
                      ViT+Xception, ViT+InceptionV3, Xception+InceptionV3
  1 ensemble        : soft voting = average of the 3 hybrids' probabilities

Every model is the same small head:  dense(512) -> ReLU -> dropout -> dense(4 classes)
Riders are rare, so the loss gives each class equal total weight.
Each head keeps the epoch with the best VALIDATION macro-F1; TEST is only used for the final scores.

Outputs
  saved_models/image/heads.pt                     hybrid heads + feature statistics (used by the agent)
  results/tables/image_model_comparison.csv
  results/tables/image_ensemble_report.txt
  results/figures/image_model_comparison.png, image_confusion_ensemble.png, image_per_class_f1.png

Run (Kaggle, after 02_extract_features.py; also fine on a laptop CPU, ~10 min):
    python image_pipeline/03_train_heads.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, \
    precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (FEATURES_DIR, IMAGE_MODELS_DIR, TABLES_DIR, FIGURES_DIR, IMAGE_CLASSES,
                    BACKBONES, HYBRIDS, RANDOM_SEED)
from image_pipeline.img_common import Head, device

EPOCHS, PATIENCE, BATCH, LR = 40, 6, 256, 1e-3
NAMES = {"vgg16": "VGG16", "resnet50": "ResNet50", "inception_v3": "InceptionV3",
         "xception": "Xception", "vit": "ViT"}


# ======================================================================== data
def load_feats():
    y = {s: np.load(FEATURES_DIR / f"labels_{s}.npy") for s in ["train", "val", "test"]}
    y["train"] = np.concatenate([y["train"], y["train"]])          # original + augmented copy
    X, stats = {}, {}
    for b in BACKBONES:
        tr = np.concatenate([np.load(FEATURES_DIR / f"{b}_train.npy"),
                             np.load(FEATURES_DIR / f"{b}_train_aug.npy")]).astype(np.float32)
        mean, std = tr.mean(0), tr.std(0) + 1e-6                    # statistics from TRAIN only
        stats[b] = (mean, std)
        X[b] = {"train": (tr - mean) / std}
        for s in ["val", "test"]:
            X[b][s] = (np.load(FEATURES_DIR / f"{b}_{s}.npy").astype(np.float32) - mean) / std
    return X, y, stats


def join(X, parts, split):
    return np.concatenate([X[p][split] for p in parts], axis=1)


# ======================================================================== training
def train_head(Xtr, ytr, Xva, yva, dev):
    torch.manual_seed(RANDOM_SEED)
    n_cls = len(IMAGE_CLASSES)
    counts = np.bincount(ytr, minlength=n_cls)
    weights = torch.tensor(len(ytr) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    head = Head(Xtr.shape[1], n_cls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    Xv = torch.tensor(Xva, device=dev)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS):
        head.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss_fn(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        f1 = f1_score(yva, predict(head, Xv).argmax(1), average="macro")
        if f1 > best:
            best, best_state, bad = f1, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    head.load_state_dict(best_state)
    return head.cpu().eval(), best, epoch + 1


@torch.no_grad()
def predict(head, X):
    head.eval()
    if not torch.is_tensor(X):
        X = torch.tensor(X, device=next(head.parameters()).device)
    return torch.softmax(head(X), dim=1).cpu().numpy()


def score(name, kind, y, prob, val_f1, secs, epochs):
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(len(IMAGE_CLASSES)), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    row = {"model": name, "type": kind, "val_macro_f1": val_f1, "test_accuracy": accuracy_score(y, pred),
           "test_macro_precision": mp, "test_macro_recall": mr, "test_macro_f1": mf,
           "train_seconds": secs, "epochs": epochs}
    for i, c in enumerate(IMAGE_CLASSES):
        row[f"f1_{c}"] = f[i]
    return row


# ======================================================================== main
def main():
    dev = device()
    IMAGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X, y, stats = load_feats()
    print(f"Device: {dev} | train {len(y['train']):,} (incl. augmented) | val {len(y['val']):,} | test {len(y['test']):,}")

    rows, probs_val, probs_test, heads = [], {}, {}, {}
    plan = [(NAMES[b], "single", [b]) for b in BACKBONES] + \
           [(f"{NAMES[a]} + {NAMES[b]}", "hybrid", [a, b]) for a, b in HYBRIDS]
    for name, kind, parts in plan:
        t0 = time.time()
        head, val_f1, ep = train_head(join(X, parts, "train"), y["train"], join(X, parts, "val"), y["val"], dev)
        secs = time.time() - t0
        probs_val[name] = predict(head, join(X, parts, "val"))
        probs_test[name] = predict(head, join(X, parts, "test"))
        heads[name] = (head, parts)
        rows.append(score(name, kind, y["test"], probs_test[name], val_f1, secs, ep))
        print(f"{name:<26} {kind:<7} val macro-F1 {val_f1:.3f}  test macro-F1 {rows[-1]['test_macro_f1']:.3f}  ({secs:.0f}s, {ep} epochs)")

    hybrid_names = [n for n, k, _ in plan if k == "hybrid"]
    ens_val = np.mean([probs_val[n] for n in hybrid_names], axis=0)
    ens_test = np.mean([probs_test[n] for n in hybrid_names], axis=0)
    ens_name = "Soft-voting ensemble (3 hybrids)"
    rows.append(score(ens_name, "ensemble", y["test"], ens_test,
                      f1_score(y["val"], ens_val.argmax(1), average="macro"), 0.0, 0))
    print(f"{ens_name:<26} val macro-F1 {rows[-1]['val_macro_f1']:.3f}  test macro-F1 {rows[-1]['test_macro_f1']:.3f}")

    table = pd.DataFrame(rows).set_index("model").round(4)
    table.to_csv(TABLES_DIR / "image_model_comparison.csv")
    pd.set_option("display.width", 220)
    print("\n" + table.drop(columns=["train_seconds", "epochs"]).to_string())

    report = classification_report(y["test"], ens_test.argmax(1), target_names=IMAGE_CLASSES, digits=3, zero_division=0)
    (TABLES_DIR / "image_ensemble_report.txt").write_text(f"{ens_name}\n\n{report}", encoding="utf-8")
    print(f"\n{ens_name} on TEST:\n{report}")

    # ---- save what the live system needs: the 3 hybrid heads + feature statistics
    used = sorted({p for n in hybrid_names for p in heads[n][1]})
    torch.save({
        "classes": IMAGE_CLASSES,
        "backbones": {b: BACKBONES[b] for b in used},
        "stats": {b: (torch.tensor(stats[b][0]), torch.tensor(stats[b][1])) for b in used},
        "heads": {n: {"parts": heads[n][1], "in_dim": int(sum(stats[p][0].shape[0] for p in heads[n][1])),
                      "state_dict": heads[n][0].state_dict()} for n in hybrid_names},
    }, IMAGE_MODELS_DIR / "heads.pt")
    print(f"Saved: {IMAGE_MODELS_DIR / 'heads.pt'}")

    make_figures(table, confusion_matrix(y["test"], ens_test.argmax(1), labels=range(len(IMAGE_CLASSES))), ens_name)


# ======================================================================== figures
def make_figures(table, cm, ens_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
    kind_color = {"single": BLUE, "hybrid": AQUA, "ensemble": ORANGE}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1) macro-F1 for all 9 models, grouped by type
    t = table.iloc[::-1]
    lo = max(0.0, float(t["test_macro_f1"].min()) - 0.1)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 2.0))
    yy = np.arange(len(t))
    ax.barh(yy, t["test_macro_f1"], color=[kind_color[k] for k in t["type"]], height=0.62)
    for yi, v in zip(yy, t["test_macro_f1"]):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(yy, t.index)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("macro F1 on test crops")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in kind_color.values()]
    ax.legend(handles, ["Single backbone", "Hybrid (feature fusion)", "Soft-voting ensemble"],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.set_title("Image models: single vs hybrid vs ensemble", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "image_model_comparison.png", dpi=200)
    plt.close(fig)

    # 2) confusion matrix of the ensemble
    n = len(IMAGE_CLASSES)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = [c.replace("_", "\n") for c in IMAGE_CLASSES]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix on test crops: ensemble", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "image_confusion_ensemble.png", dpi=200)
    plt.close(fig)

    # 3) per-class F1: best single vs best hybrid vs ensemble
    singles = table[table["type"] == "single"]
    hybrids = table[table["type"] == "hybrid"]
    pick = [singles["val_macro_f1"].idxmax(), hybrids["val_macro_f1"].idxmax(), ens_name]
    cols = [f"f1_{c}" for c in IMAGE_CLASSES]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(n)
    w = 0.26
    for i, (m, c) in enumerate(zip(pick, [BLUE, AQUA, ORANGE])):
        vals = table.loc[m, cols].astype(float).to_numpy()
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=c, label=f"{m}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [c.replace("_", " ") for c in IMAGE_CLASSES])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 on test crops")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Per-class F1: best single vs best hybrid vs ensemble", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "image_per_class_f1.png", dpi=200)
    plt.close(fig)
    print(f"Figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
