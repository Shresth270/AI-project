"""
PHASE 1B - Clean the Melbourne counts and build features
--------------------------------------------------------
For every sensor and every hour t, one row with:
  * time features  : hour, weekday, month, weekend, public holiday
  * event features : event yes/no + type (from data/events.csv)
  * recent history : counts 1, 2, 3, 24 and 168 hours earlier, trend, surge-so-far
  * normal level   : median count for this sensor / weekday / hour (TRAIN data only)
  * target         : crowd level at hour t -> Low / Medium / High / Very High
                     (by that sensor's own TRAIN percentiles: 25 / 75 / 95)

The question the model answers:
  "Standing at the end of hour t-1, how crowded will hour t be?"
The count of hour t itself is never an input.

Split is by TIME: the last 20% of the period is the test set.

Outputs
  data/processed/sensor_features.parquet
  saved_models/sensor_baseline.parquet, level_thresholds.csv, sensors.csv
  results/tables/sensor_summary.csv,  results/figures/sensor_*.png

Run:  python sensor_pipeline/01_clean_features.py
"""
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (MELBOURNE_DIR, EVENTS_CSV, MODELS_DIR, TABLES_DIR, FIGURES_DIR, PROCESSED_DIR,
                    SENSOR_FEATURES_FILE, FOOTFALL_LEVELS, LEVEL_PERCENTILES, TEST_FRACTION,
                    START_DATE, PRE_COVID_WINDOW, MIN_SENSOR_DAYS, LAG_HOURS)

EVENT_TYPES = ["none", "celebration", "sport", "festival", "shopping", "other"]


# ======================================================================== 1. load
def load_counts() -> pd.DataFrame:
    """Read every CSV in data/melbourne (current or 2009-2022 format) -> sensor_id, sensor_name, ts, count."""
    files = sorted(MELBOURNE_DIR.rglob("*.csv"))
    if not files:
        sys.exit(f"[STOP] No CSV in {MELBOURNE_DIR}")
    parts = []
    for f in files:
        cols = {c.lower(): c for c in pd.read_csv(f, nrows=0).columns}
        if {"sensing_date", "hourday", "pedestriancount"} <= cols.keys():          # current export
            sid = cols.get("location_id") or cols.get("sensor_id")
            use = [cols["sensing_date"], cols["hourday"], cols["pedestriancount"], sid, cols["sensor_name"]]
            df = pd.read_csv(f, usecols=use, low_memory=False)[use]   # usecols keeps file order; reorder
            df.columns = ["date", "hour", "count", "sensor_id", "sensor_name"]
            df["ts"] = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(
                pd.to_numeric(df["hour"], errors="coerce"), unit="h")
        elif {"year", "month", "mdate", "time", "hourly_counts"} <= cols.keys():   # 2009-2022 format
            use = [cols[c] for c in ("year", "month", "mdate", "time", "hourly_counts", "sensor_id", "sensor_name")]
            df = pd.read_csv(f, usecols=use, low_memory=False)[use]   # usecols keeps file order; reorder
            df.columns = ["year", "month", "mdate", "hour", "count", "sensor_id", "sensor_name"]
            day = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-"
                                 + df["mdate"].astype(str), format="%Y-%B-%d", errors="coerce")
            df["ts"] = day + pd.to_timedelta(pd.to_numeric(df["hour"], errors="coerce"), unit="h")
        else:
            print(f"[SKIP] {f.name}: columns not recognised -> {list(cols.values())}")
            continue
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce")
        df = df.dropna(subset=["ts", "count", "sensor_id"])
        parts.append(df[["sensor_id", "sensor_name", "ts", "count"]])
        print(f"Loaded {f.name}: {len(df):,} rows")
    if not parts:
        sys.exit("[STOP] No usable CSV found.")
    df = pd.concat(parts, ignore_index=True)
    df["sensor_id"] = df["sensor_id"].astype(int)
    df = df[df["count"] >= 0]
    # one value per sensor-hour (files can overlap)
    df = df.sort_values("ts").drop_duplicates(["sensor_id", "ts"], keep="last")
    return df


# ======================================================================== 2. calendar
def build_events(years) -> pd.DataFrame:
    """Use data/events.csv if it has rows; otherwise generate recurring Melbourne events from rules."""
    if EVENTS_CSV.exists():
        ev = pd.read_csv(EVENTS_CSV)
        if len(ev):
            print(f"Events: using {len(ev)} rows from {EVENTS_CSV.name}")
            return ev
    rows = []
    for y in years:
        vic = holidays.AU(subdiv="VIC", years=y)
        rows += [(f"{y}-12-31", "New Year's Eve", "celebration", 17, 23),
                 (f"{y}-01-01", "New Year's Eve (after midnight)", "celebration", 0, 2),
                 (f"{y}-12-26", "Boxing Day sales", "shopping", 8, 20)]
        for day, name in vic.items():
            if "Grand Final" in name:   # Friday before the AFL Grand Final: city parade
                rows.append((str(day), "AFL Grand Final parade", "sport", 8, 18))
                rows.append((str(day + pd.Timedelta(days=1)), "AFL Grand Final day", "sport", 10, 21))
            if name == "Labour Day":    # Moomba festival runs over the Labour Day long weekend
                for k in (-2, -1, 0):
                    rows.append((str(day + pd.Timedelta(days=k)), "Moomba festival", "festival", 10, 22))
    ev = pd.DataFrame(rows, columns=["date", "event_name", "event_type", "start_hour", "end_hour"])
    ev = ev.sort_values(["date", "start_hour"]).reset_index(drop=True)
    ev.to_csv(EVENTS_CSV, index=False)
    print(f"Events: generated {len(ev)} rows -> {EVENTS_CSV.name} (add more events there any time)")
    return ev


def event_lookup(ev: pd.DataFrame) -> pd.DataFrame:
    """Expand events to one row per (date-hour) with its type."""
    rows = []
    for r in ev.itertuples():
        for h in range(int(r.start_hour), int(r.end_hour) + 1):
            t = pd.Timestamp(r.date) + pd.Timedelta(hours=h)
            et = r.event_type if r.event_type in EVENT_TYPES else "other"
            rows.append((t, et))
    out = pd.DataFrame(rows, columns=["ts", "event_type"]).drop_duplicates("ts", keep="first")
    return out


# ======================================================================== 3. features
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_counts()
    print(f"All data: {raw['ts'].min()} -> {raw['ts'].max()}, {raw['sensor_id'].nunique()} sensors")

    # ---- choose the study window
    start, end = pd.Timestamp(START_DATE), raw["ts"].max()
    if (end - start).days < 365:
        start, end = map(pd.Timestamp, PRE_COVID_WINDOW)
        print(f"[NOTE] Less than a year after {START_DATE}; using the pre-COVID window instead.")
    start, end = max(start, raw["ts"].min().normalize()), min(end, raw["ts"].max())   # stay inside the data
    print(f"Study window: {start.date()} -> {end.date()}")

    names = raw.sort_values("ts").groupby("sensor_id")["sensor_name"].last()

    # ---- full hourly grid (wide: rows = hours, columns = sensors). Missing hour = sensor offline = NaN
    lag_start = start - pd.Timedelta(hours=max(LAG_HOURS))
    win = raw[(raw["ts"] >= lag_start) & (raw["ts"] <= end)]
    wide = win.pivot_table(index="ts", columns="sensor_id", values="count", aggfunc="last")
    wide = wide.reindex(pd.date_range(lag_start, end, freq="h")).astype("float32")

    # keep sensors with enough history inside the window
    days = wide.loc[start:].notna().sum() / 24
    keep = days[days >= MIN_SENSOR_DAYS].index
    wide = wide[keep]
    print(f"Sensors kept: {len(keep)} (dropped {len(days) - len(keep)} with < {MIN_SENSOR_DAYS} days)")

    # ---- lags computed on the grid, then turned into long format
    frames = {"count": wide}
    for L in LAG_HOURS:
        frames[f"lag_{L}"] = wide.shift(L)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    long.index.names = ["ts", "sensor_id"]
    long = long.reset_index()
    long = long[long["ts"] >= start].dropna()
    print(f"Rows with full history: {len(long):,}")

    # ---- time features
    ts = long["ts"]
    long["hour"] = ts.dt.hour.astype("int8")
    long["weekday"] = ts.dt.dayofweek.astype("int8")
    long["month"] = ts.dt.month.astype("int8")
    long["is_weekend"] = (long["weekday"] >= 5).astype("int8")
    years = range(ts.dt.year.min() - 1, ts.dt.year.max() + 2)
    vic = holidays.AU(subdiv="VIC", years=years)
    hol_days = pd.to_datetime(list(vic.keys()))
    long["is_holiday"] = ts.dt.normalize().isin(hol_days).astype("int8")

    # ---- event features
    ev = event_lookup(build_events(years))
    long = long.merge(ev, on="ts", how="left")
    long["event_type"] = long["event_type"].fillna("none")
    long["is_event"] = (long["event_type"] != "none").astype("int8")

    # ---- split by time
    cutoff = start + (end - start) * (1 - TEST_FRACTION)
    long["split"] = np.where(long["ts"] >= cutoff, "test", "train")
    train = long[long["split"] == "train"]
    print(f"Train: {start.date()} -> {cutoff.date()}  |  Test: {cutoff.date()} -> {end.date()}")

    # ---- normal level (median) per sensor / weekday / hour, from TRAIN only
    base = (train.groupby(["sensor_id", "weekday", "hour"])["count"].median()
            .rename("baseline").reset_index())
    long = long.merge(base, on=["sensor_id", "weekday", "hour"], how="left")
    prev = long["ts"] - pd.Timedelta(hours=1)
    prev_key = pd.DataFrame({"sensor_id": long["sensor_id"], "weekday": prev.dt.dayofweek.astype("int8"),
                             "hour": prev.dt.hour.astype("int8")})
    long["baseline_prev"] = prev_key.merge(base, on=["sensor_id", "weekday", "hour"], how="left")["baseline"].values
    long["trend"] = long["lag_1"] - long["lag_2"]
    long["surge_so_far"] = (long["lag_1"] + 1) / (long["baseline_prev"] + 1)   # >1 = busier than usual right now

    # ---- target: level by each sensor's TRAIN percentiles
    q = train.groupby("sensor_id")["count"].quantile(list(LEVEL_PERCENTILES.values())).unstack()
    q.columns = ["p25", "p75", "p95"]
    long = long.merge(q, left_on="sensor_id", right_index=True, how="inner")
    c = long["count"]
    long["level"] = np.select([c <= long["p25"], c <= long["p75"], c <= long["p95"]], [0, 1, 2], 3).astype("int8")
    long["level_name"] = long["level"].map(dict(enumerate(FOOTFALL_LEVELS)))
    long = long.dropna(subset=["baseline", "baseline_prev"])
    long["sensor_name"] = long["sensor_id"].map(names)

    # ---- save
    cols = (["ts", "sensor_id", "sensor_name", "split", "hour", "weekday", "month", "is_weekend",
             "is_holiday", "is_event", "event_type"] + [f"lag_{L}" for L in LAG_HOURS]
            + ["trend", "baseline", "surge_so_far", "count", "level", "level_name"])
    long = long[cols].sort_values(["ts", "sensor_id"]).reset_index(drop=True)
    long.to_parquet(SENSOR_FEATURES_FILE, index=False)
    base.to_parquet(MODELS_DIR / "sensor_baseline.parquet", index=False)
    q.assign(sensor_name=q.index.map(names)).to_csv(MODELS_DIR / "level_thresholds.csv")
    names.loc[keep].rename("sensor_name").to_csv(MODELS_DIR / "sensors.csv")

    summary = (long.groupby(["split", "level_name"]).size().unstack(0).reindex(FOOTFALL_LEVELS))
    summary.loc["TOTAL"] = summary.sum()
    summary["share_train"] = (summary["train"] / summary.loc["TOTAL", "train"]).round(3)
    summary.to_csv(TABLES_DIR / "sensor_summary.csv")
    print("\nRows per level and split:")
    print(summary.to_string())
    print(f"\nEvent hours: {long['is_event'].sum():,}  |  Holiday hours: {long['is_holiday'].sum():,}")
    print(f"Saved: {SENSOR_FEATURES_FILE}")

    make_figures(long)


# ======================================================================== 4. figures for the report
def make_figures(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1) average footfall by weekday x hour (all sensors)
    tr = df[df["split"] == "train"]
    heat = tr.pivot_table(index="weekday", columns="hour", values="count", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(heat.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.set_title("Average pedestrians per sensor per hour", loc="left")
    fig.colorbar(im, ax=ax, label="people / hour", fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensor_weekday_hour_heatmap.png", dpi=200)
    plt.close(fig)

    # 2) level distribution (train), ordered light -> dark
    shades = ["#cfe1f7", "#8bb6ea", "#2a78d6", "#174c8c"]
    counts = tr["level_name"].value_counts().reindex(FOOTFALL_LEVELS)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(FOOTFALL_LEVELS, counts.values, color=shades, width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v / counts.sum():.0%}", ha="center", va="bottom", color=INK)
    ax.set_ylabel("sensor-hours")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Crowd-level classes (training data)", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensor_level_distribution.png", dpi=200)
    plt.close(fig)

    # 3) event days vs normal days of the same weekday
    df = df.assign(date=df["ts"].dt.date)
    event_dates = set(df.loc[df["is_event"] == 1, "date"])
    if event_dates:
        is_ev_day = df["date"].isin(event_dates)
        ev_days = df[is_ev_day].groupby("hour")["count"].mean()
        wds = df.loc[is_ev_day, "weekday"].unique()
        normal = df[~is_ev_day & df["weekday"].isin(wds) & (df["is_holiday"] == 0)].groupby("hour")["count"].mean()
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(normal.index, normal.values, color=BLUE, linewidth=2, label="Normal days (same weekdays)")
        ax.plot(ev_days.index, ev_days.values, color=ORANGE, linewidth=2, label="Event days")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("avg people / sensor / hour")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, loc="upper left")
        ax.set_title("Event days are busier than normal days", loc="left")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sensor_event_vs_normal.png", dpi=200)
        plt.close(fig)
    print(f"Figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
