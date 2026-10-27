"""
PHASE 2B - Early-warning replay
-------------------------------
Replays the TEST months hour by hour, as if the system were running live, and asks:
  * When a place hits "Very High", had the system warned 1 hour before?
  * How often does it raise a false alarm?
  * On the event days hidden from training (e.g. AFL Grand Final), what happened?

A "surge episode" = consecutive Very-High hours at one sensor.
"Warned at onset"  = the forecast made at the end of the hour BEFORE the episode
                     already said Very High (a real 1-hour head start).

The warning threshold (how sure the model must be before it warns) is chosen on the
VALIDATION months, never on test, and saved for the agent.

Outputs
  saved_models/warning_threshold.json
  results/tables/early_warning_summary.csv, early_warning_events.csv, warning_threshold_sweep.csv
  results/figures/early_warning_onset.png, early_warning_timeline.png

Run:  python sensor_pipeline/03_early_warning.py      (after 02_train_compare.py)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import SENSOR_FEATURES_FILE, MODELS_DIR, TABLES_DIR, FIGURES_DIR, EVENTS_CSV
from sensor_pipeline.common import load_thresholds, add_features, level_from_count

VH = 3
VAL_SHARE = 0.15   # must match 02_train_compare.py


# ======================================================================== helpers
def predict_proba(bundle, df):
    num = [c for c in bundle["features"] if c not in bundle["categorical"]]
    X = df[bundle["features"]].astype("float32").copy()
    X[num] = bundle["scaler"].transform(df[num])
    return bundle["model"].predict_proba(X)[:, VH]


def choose_threshold(y_true_vh, p_vh):
    """Pick the probability cut that maximises F2 (recall counts double: a missed surge costs more)."""
    rows = []
    for t in np.round(np.arange(0.10, 0.91, 0.05), 2):
        warn = p_vh >= t
        tp = (warn & y_true_vh).sum()
        prec = tp / max(warn.sum(), 1)
        rec = tp / max(y_true_vh.sum(), 1)
        f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f2": f2})
    sweep = pd.DataFrame(rows)
    return float(sweep.loc[sweep["f2"].idxmax(), "threshold"]), sweep


def episodes(df, th):
    """Label surge episodes (consecutive Very-High hours at one sensor).
    onset = a Very-High hour whose previous hour was NOT Very High (the surge is just starting)."""
    df = df.sort_values(["sensor_id", "ts"]).copy()
    vh = df["level"].to_numpy() == VH
    same_sensor = df["sensor_id"].to_numpy()[1:] == df["sensor_id"].to_numpy()[:-1]
    next_hour = (df["ts"].diff().dt.total_seconds().to_numpy()[1:] == 3600)
    continues = np.r_[False, same_sensor & next_hour & vh[:-1]]
    new_episode = vh & ~continues
    df["episode"] = np.where(vh, np.cumsum(new_episode), 0)
    prev_vh = level_from_count(df["lag_1"], df["sensor_id"], th) == VH
    df["onset"] = vh & ~prev_vh
    return df


def evaluate(df, warn_col, weeks):
    vh = df["level"] == VH
    w = df[warn_col]
    ep = df[df["episode"] > 0]
    caught_any = ep.groupby("episode")[warn_col].any()
    onset = df[df["onset"]]
    unusual = onset[~onset["routine"]]
    return {
        "surge_hours_caught": (w & vh).sum() / max(vh.sum(), 1),
        "warning_precision": (w & vh).sum() / max(w.sum(), 1),
        "false_alarms_per_sensor_week": (w & (df["level"] <= 1)).sum() / weeks,
        "episodes": int(ep["episode"].nunique()),
        "onsets": len(onset),
        "warned_at_onset": onset[warn_col].mean(),
        "caught_during_episode": caught_any.mean(),
        "unusual_onsets": len(unusual),
        "unusual_warned_at_onset": unusual[warn_col].mean() if len(unusual) else np.nan,
    }


# ======================================================================== main
def main():
    bundle = joblib.load(MODELS_DIR / "footfall_model.joblib")
    th = load_thresholds()
    df = pd.read_parquet(SENSOR_FEATURES_FILE)
    df = add_features(df[df["sensor_id"].isin(th.index)], th)

    # ---- choose the warning threshold on VALIDATION months
    train = df[df["split"] == "train"]
    cut = train["ts"].min() + (train["ts"].max() - train["ts"].min()) * (1 - VAL_SHARE)
    val = train[train["ts"] >= cut]
    thr, sweep = choose_threshold(val["level"].to_numpy() == VH, predict_proba(bundle, val))
    sweep.round(4).to_csv(TABLES_DIR / "warning_threshold_sweep.csv", index=False)
    (MODELS_DIR / "warning_threshold.json").write_text(json.dumps({"very_high_probability": thr}, indent=2))
    print(f"Model: {bundle['name']}  |  warning threshold chosen on validation: P(Very High) >= {thr}")

    # ---- replay TEST months
    test = df[df["split"] == "test"].copy()
    test["p_vh"] = predict_proba(bundle, test)
    test["warn_model"] = test["p_vh"] >= thr
    test["warn_last_hour"] = level_from_count(test["lag_1"], test["sensor_id"], th) == VH
    test["warn_usual"] = level_from_count(test["baseline"], test["sensor_id"], th) == VH
    test["routine"] = test["warn_usual"]          # Very High is normal for this slot (e.g. every Saturday 1 pm)
    test = episodes(test, th)
    weeks = test.groupby("sensor_id")["ts"].agg(lambda s: (s.max() - s.min()).days / 7).sum()

    methods = {"AI model (1 h ahead)": "warn_model", "Rule: same as last hour": "warn_last_hour",
               "Rule: usual for this slot": "warn_usual"}
    summary = pd.DataFrame({m: evaluate(test, c, weeks) for m, c in methods.items()}).T
    summary.to_csv(TABLES_DIR / "early_warning_summary.csv")
    pd.set_option("display.width", 200)
    print("\nEarly-warning replay on TEST months")
    print(summary.round(3).to_string())

    # ---- event days hidden from training
    ev = pd.read_csv(EVENTS_CSV)
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[(ev["date"] >= test["ts"].min().normalize()) & (ev["date"] <= test["ts"].max())]
    ev_rows = []
    for r in ev.itertuples():
        mask = (test["ts"].dt.normalize() == r.date) & test["hour"].between(r.start_hour, r.end_hour)
        d = test[mask]
        vh = d["level"] == VH
        ev_rows.append({"date": r.date.date(), "event": r.event_name, "sensor_hours": len(d),
                        "very_high_hours": int(vh.sum()),
                        "caught_by_model": (d["warn_model"] & vh).sum() / max(vh.sum(), 1),
                        "caught_by_usual_rule": (d["warn_usual"] & vh).sum() / max(vh.sum(), 1),
                        "onsets": int(d["onset"].sum()),
                        "onsets_warned_by_model": int((d["onset"] & d["warn_model"]).sum())})
    ev_table = pd.DataFrame(ev_rows)
    ev_table.to_csv(TABLES_DIR / "early_warning_events.csv", index=False)
    if len(ev_table):
        print("\nEvent days in the test period (never seen in training):")
        print(ev_table.round(3).to_string(index=False))
    else:
        print("\nNo listed events fall in the test period.")

    make_figures(summary, test, ev, th)


# ======================================================================== figures
def make_figures(summary, test, ev, th):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1) onset warnings: all surges vs unusual surges
    cols = ["warned_at_onset", "unusual_warned_at_onset"]
    labels = ["All surges", "Unusual surges\n(not a routine peak)"]
    colors = {"AI model (1 h ahead)": ORANGE, "Rule: same as last hour": GRAY, "Rule: usual for this slot": BLUE}
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cols))
    wdt = 0.26
    for i, (m, row) in enumerate(summary.iterrows()):
        vals = row[cols].astype(float).fillna(0).to_numpy()
        bars = ax.bar(x + (i - 1) * wdt, vals, wdt - 0.03, color=colors[m], label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("surges warned 1 hour before they began")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Early warning: who saw the surge coming?", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "early_warning_onset.png", dpi=200)
    plt.close(fig)

    # 2) timeline for the busiest event day in test (or the biggest unusual surge)
    if len(ev):
        day_vh = {r.date: (test["ts"].dt.normalize() == r.date) & (test["level"] == VH) for r in ev.itertuples()}
        day = max(day_vh, key=lambda d: day_vh[d].sum())
        sensor = test[day_vh[day]]["sensor_id"].value_counts().index[0] if day_vh[day].any() else test["sensor_id"].iloc[0]
        title_bit = ev[ev["date"] == day]["event_name"].iloc[0]
    else:
        onset = test[test["onset"] & ~test["routine"]]
        pick = onset.iloc[0] if len(onset) else test.iloc[0]
        day, sensor, title_bit = pick["ts"].normalize(), pick["sensor_id"], "largest unusual surge"
    d = test[(test["sensor_id"] == sensor) & (test["ts"] >= day - pd.Timedelta(hours=6))
             & (test["ts"] < day + pd.Timedelta(hours=30))].sort_values("ts")
    if d.empty:
        return
    p95 = th.loc[sensor, "p95"]
    name = th.loc[sensor, "sensor_name"] if "sensor_name" in th.columns else sensor
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["ts"], d["baseline"], color=GRAY, linewidth=2, label="Usual for this hour")
    ax.plot(d["ts"], d["count"], color=BLUE, linewidth=2, label="Actual count")
    ax.axhline(p95, color=MUTED, linewidth=1, linestyle="--")
    ax.text(d["ts"].iloc[0], p95, "  Very High line", va="bottom", color=MUTED, fontsize=9)
    w = d[d["warn_model"]]
    ax.scatter(w["ts"], w["count"], s=60, color=ORANGE, zorder=3, edgecolor="white", linewidth=1.5,
               label="Very High forecast 1 h ahead")
    miss = d[(d["level"] == VH) & ~d["warn_model"]]
    if len(miss):
        ax.scatter(miss["ts"], miss["count"], s=60, facecolor="white", edgecolor=MUTED, linewidth=1.5,
                   zorder=3, label="Very High hour missed")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_ylabel("people / hour")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"{title_bit}, {pd.Timestamp(day).date()} - sensor {name}", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "early_warning_timeline.png", dpi=200)
    plt.close(fig)
    print(f"Figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
