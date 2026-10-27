"""
Redraw the 'single vs hybrid vs ensemble' chart from the saved results table
(legend moved below the plot so it no longer covers the bars). Runs on a laptop, no GPU.

Run:  python image_pipeline/04_redraw_comparison.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TABLES_DIR, FIGURES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
kind_color = {"single": "#2a78d6", "hybrid": "#1baf7a", "ensemble": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

table = pd.read_csv(TABLES_DIR / "image_model_comparison.csv", index_col=0)
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
out = FIGURES_DIR / "image_model_comparison.png"
fig.savefig(out, dpi=200)
print(f"Redrawn: {out}")
