"""
The agent's TOOLS - plain Python functions that wrap our trained models.
The LLM can only act through these; every number in a report comes from here.

  list_locations      cameras + sensors the system knows
  get_events          public holidays and listed events on a date
  get_baseline        what is normal for a place at a given hour
  forecast_footfall   next-hour crowd level from the sensor model (XGBoost)
  detect_people       finds every person in a camera image (YOLO)
  classify_people     labels each detected person with our soft-voting ensemble
  assess_risk         fixed safety rules -> Normal / Watch / Act now

Sensor data is a REPLAY of the real Melbourne history: asking about a time that exists
in the data uses the real counts of the hours before it ("replay" mode). A time in the
future uses the calendar and normal patterns only ("planning" mode).
"""
import json
import os
import sys
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (SENSOR_FEATURES_FILE, MODELS_DIR, IMAGE_MODELS_DIR, EVENTS_CSV, CAMERAS_CSV,
                    FOOTFALL_LEVELS, IMAGE_CLASSES, CROP_PAD, CROP_SAVE_MAX_SIDE, YOLO_WEIGHTS,
                    YOLO_IMG_SIZE, DEFAULT_CAMERA_AREA_M2, UPLOADS_DIR, LAG_HOURS)
from sensor_pipeline.common import load_thresholds, add_features, level_from_count, EVENT_TYPES
from agent.rules import assess_risk as _assess_risk

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLASS_COLORS = {"pedestrian": (42, 120, 214), "rider": (27, 175, 122),
                "partially_visible": (235, 104, 52), "background": (163, 162, 156)}


def clean(obj):
    """Make tool results JSON-safe (numpy -> Python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 3)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d %H:%M")
    return obj


class Toolbox:
    def __init__(self):
        import holidays
        import joblib

        self.th = load_thresholds()
        # some Melbourne sensors have no name in the data: give them one from their id
        names = self.th["sensor_name"].astype("object")
        missing = names.isna() | (names.map(lambda v: str(v).strip().lower()).isin(["", "nan", "none"]))
        names[missing] = [f"Sensor{i}" for i in self.th.index[missing]]
        self.th["sensor_name"] = names.map(str)
        self.base = pd.read_parquet(MODELS_DIR / "sensor_baseline.parquet").set_index(["sensor_id", "weekday", "hour"])["baseline"]
        cols = ["ts", "sensor_id", "sensor_name", "hour", "weekday", "month", "is_weekend", "is_holiday", "is_event",
                "event_type"] + [f"lag_{L}" for L in LAG_HOURS] + ["trend", "baseline", "surge_so_far", "count", "level"]
        self.feats = pd.read_parquet(SENSOR_FEATURES_FILE, columns=cols).set_index(["sensor_id", "ts"]).sort_index()
        self.replay_now = self.feats.index.get_level_values("ts").max()
        self.bundle = joblib.load(MODELS_DIR / "footfall_model.joblib")
        wt = MODELS_DIR / "warning_threshold.json"
        self.warn_thr = json.loads(wt.read_text())["very_high_probability"] if wt.exists() else 0.5
        self.events = pd.read_csv(EVENTS_CSV, parse_dates=["date"]) if EVENTS_CSV.exists() else pd.DataFrame()
        years = range(self.replay_now.year - 3, self.replay_now.year + 3)
        self.holidays = holidays.AU(subdiv="VIC", years=years)
        self.cameras = self._load_cameras()
        self._yolo = None
        self._img_models = None
        self._detections = {}

    # ------------------------------------------------------------------ registry
    def _load_cameras(self):
        """Camera registry (data/cameras.csv). If missing or old-style, create demo cameras
        at the 6 busiest sensors. You can rename cameras and set visible_area_m2 in the file."""
        cams = pd.read_csv(CAMERAS_CSV) if CAMERAS_CSV.exists() else pd.DataFrame()
        valid = len(cams) and "sensor_id" in cams.columns and cams["sensor_id"].isin(self.th.index).all()
        if not valid:
            top = self.th.sort_values("p75", ascending=False).head(6)
            cams = pd.DataFrame({
                "camera_id": [f"CAM-{i + 1}" for i in range(len(top))],
                "camera_name": [f"Camera {i + 1} near sensor {n}" for i, n in enumerate(top["sensor_name"])],
                "sensor_id": top.index.astype(int),
                "sensor_name": top["sensor_name"].values,
                "visible_area_m2": DEFAULT_CAMERA_AREA_M2,
            })
            cams.to_csv(CAMERAS_CSV, index=False)
        cams["camera_id"] = cams["camera_id"].astype(str)
        cams["camera_name"] = cams["camera_name"].astype(str)
        cams["sensor_id"] = cams["sensor_id"].astype(int)
        return cams

    def _sensor(self, location):
        """Accept a camera id/name, sensor id or sensor name (fuzzy) -> sensor_id."""
        q = str(location).strip()
        cam = self.cameras[(self.cameras["camera_id"].str.lower() == q.lower()) |
                           (self.cameras["camera_name"].str.lower() == q.lower())]
        if len(cam):
            return int(cam["sensor_id"].iloc[0])
        if q.isdigit() and int(q) in self.th.index:
            return int(q)
        names = self.th["sensor_name"]
        low = names.str.lower()
        exact = names[low == q.lower()]
        if len(exact):
            return int(exact.index[0])
        part = names[low.str.contains(q.lower(), regex=False)]
        if len(part):
            return int(part.index[0])
        close = get_close_matches(q, names.tolist(), n=1, cutoff=0.6)
        if close:
            return int(names[names == close[0]].index[0])
        raise ValueError(f"Unknown location '{location}'. Call list_locations to see valid cameras and sensors.")

    def _time(self, when):
        if when in (None, "", "now"):
            return self.replay_now
        return pd.Timestamp(when).floor("h")

    def _events_at(self, ts):
        if self.events.empty:
            return "none", None
        d = self.events[(self.events["date"] == ts.normalize()) &
                        (self.events["start_hour"] <= ts.hour) & (self.events["end_hour"] >= ts.hour)]
        if len(d):
            et = d["event_type"].iloc[0]
            return (et if et in EVENT_TYPES else "other"), d["event_name"].iloc[0]
        return "none", None

    def _baseline(self, sid, ts):
        return float(self.base.get((sid, ts.dayofweek, ts.hour), np.nan))

    # ------------------------------------------------------------------ tools
    def list_locations(self):
        cams = self.cameras.to_dict("records")
        return clean({"replay_now": self.replay_now, "cameras": cams,
                      "sensor_count": len(self.th),
                      "example_sensors": self.th["sensor_name"].head(15).tolist(),
                      "note": "Pass a camera_id (e.g. CAM-1) or a sensor name as 'location'."})

    def get_events(self, date):
        day = pd.Timestamp(date).normalize()
        ev = [] if self.events.empty else self.events[self.events["date"] == day][
            ["event_name", "event_type", "start_hour", "end_hour"]].to_dict("records")
        return clean({"date": day.strftime("%Y-%m-%d"), "weekday": WEEKDAYS[day.dayofweek],
                      "public_holiday": self.holidays.get(day.date()), "events": ev})

    def get_baseline(self, location, when=None):
        sid, ts = self._sensor(location), self._time(when)
        t = self.th.loc[sid]
        return clean({"sensor": t["sensor_name"], "weekday": WEEKDAYS[ts.dayofweek], "hour": ts.hour,
                      "usual_people_per_hour": self._baseline(sid, ts),
                      "level_lines_people_per_hour": {"Low_up_to": t["p25"], "Medium_up_to": t["p75"],
                                                      "High_up_to": t["p95"], "Very_High_above": t["p95"]}})

    def forecast_footfall(self, location, when=None):
        sid, ts = self._sensor(location), self._time(when)
        name = self.th.loc[sid, "sensor_name"]
        key = (sid, ts)
        if key in self.feats.index:
            row = self.feats.loc[[key]].reset_index()
            mode = "replay (uses the real counts of the previous hours)"
        else:
            row, mode = self._planning_row(sid, ts), "planning (future time: assumes normal recent counts)"
        X_df = add_features(row, self.th)
        b = self.bundle
        num = [c for c in b["features"] if c not in b["categorical"]]
        X = X_df[b["features"]].astype("float32").copy()
        X[num] = b["scaler"].transform(X_df[num])
        probs = b["model"].predict_proba(X)[0]
        p_vh = float(probs[3])
        et, ev_name = self._events_at(ts)
        r = row.iloc[0]
        return clean({
            "sensor": name, "forecast_for_hour": ts, "weekday": WEEKDAYS[ts.dayofweek], "mode": mode,
            "predicted_level": FOOTFALL_LEVELS[int(np.argmax(probs))],
            "probabilities": dict(zip(FOOTFALL_LEVELS, probs)),
            "p_very_high": p_vh, "warning_threshold": self.warn_thr, "early_warning": p_vh >= self.warn_thr,
            "usual_people_this_hour": r["baseline"], "very_high_line_people_per_hour": self.th.loc[sid, "p95"],
            "people_1h_before": r["lag_1"], "people_2h_before": r["lag_2"], "people_3h_before": r["lag_3"],
            "same_hour_last_week": r["lag_168"],
            "last_hour_vs_usual": r["surge_so_far"],
            "last_hour_level": FOOTFALL_LEVELS[int(level_from_count([r["lag_1"]], [sid], self.th)[0])],
            "event": ev_name, "public_holiday": self.holidays.get(ts.date()),
            "model": b["name"]})

    def forecast_many(self, pairs):
        """Batch version of forecast_footfall for the dashboard: [(location, when), ...] -> list of dicts.
        One model call for all rows (fast for a 24-hour x 6-camera grid)."""
        rows, info = [], []
        for loc, when in pairs:
            sid, ts = self._sensor(loc), self._time(when)
            key = (sid, ts)
            if key in self.feats.index:
                row, replay = self.feats.loc[[key]].reset_index(), True
            else:
                row, replay = self._planning_row(sid, ts), False
            rows.append(row)
            info.append((sid, ts, replay))
        df = pd.concat(rows, ignore_index=True)
        X_df = add_features(df, self.th)
        b = self.bundle
        num = [c for c in b["features"] if c not in b["categorical"]]
        X = X_df[b["features"]].astype("float32").copy()
        X[num] = b["scaler"].transform(X_df[num])
        P = b["model"].predict_proba(X)
        out = []
        for (sid, ts, replay), probs, (_, r) in zip(info, P, df.iterrows()):
            p_vh = float(probs[3])
            out.append(clean({
                "sensor_id": sid, "ts": ts, "replay": replay,
                "predicted_level": FOOTFALL_LEVELS[int(np.argmax(probs))],
                "probabilities": dict(zip(FOOTFALL_LEVELS, probs)), "p_very_high": p_vh,
                "early_warning": p_vh >= self.warn_thr, "usual": r["baseline"],
                "actual": None, "actual_level": None,
                "event": self._events_at(ts)[1]}))
        for o, (sid, ts, _) in zip(out, info):
            c = self._count_at(sid, ts)
            if c is not None:
                o["actual"] = c
                o["actual_level"] = FOOTFALL_LEVELS[int(level_from_count([c], [sid], self.th)[0])]
        return out

    def _planning_row(self, sid, ts):
        lag = {L: self._baseline(sid, ts - pd.Timedelta(hours=L)) for L in LAG_HOURS}
        et, _ = self._events_at(ts)
        base = self._baseline(sid, ts)
        prev = self._baseline(sid, ts - pd.Timedelta(hours=1))
        return pd.DataFrame([{
            "sensor_id": sid, "ts": ts, "hour": ts.hour, "weekday": ts.dayofweek, "month": ts.month,
            "is_weekend": int(ts.dayofweek >= 5), "is_holiday": int(ts.date() in self.holidays),
            "is_event": int(et != "none"), "event_type": et,
            **{f"lag_{L}": v for L, v in lag.items()},
            "trend": lag[1] - lag[2], "baseline": base, "surge_so_far": (lag[1] + 1) / (prev + 1)}])

    def _count_at(self, sid, ts):
        """Real count at (sensor, hour). If that hour's own row was dropped (a gap in its history),
        recover the count from a later row's 'hours before' columns."""
        if (sid, ts) in self.feats.index:
            return float(self.feats.loc[(sid, ts), "count"])
        for L in LAG_HOURS:
            k = (sid, ts + pd.Timedelta(hours=L))
            if k in self.feats.index:
                v = self.feats.loc[k, f"lag_{L}"]
                if pd.notna(v):
                    return float(v)
        return None

    def actual_count(self, location, when):
        """For the dashboard only (not an agent tool): what really happened in the replay data."""
        sid, ts = self._sensor(location), self._time(when)
        c = self._count_at(sid, ts)
        if c is None:
            return None
        return clean({"count": c, "level": FOOTFALL_LEVELS[int(level_from_count([c], [sid], self.th)[0])]})

    # ------------------------------------------------------------------ images
    def detect_people(self, image_path):
        from ultralytics import YOLO
        if self._yolo is None:
            YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(YOLO_WEIGHTS.parent)            # weights download next to our other models
            try:
                self._yolo = YOLO(YOLO_WEIGHTS.name)
            finally:
                os.chdir(cwd)
        path = str(Path(image_path))
        res = self._yolo.predict(path, classes=[0], imgsz=YOLO_IMG_SIZE, conf=0.25, max_det=1000, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        self._detections[path] = {"boxes": boxes, "conf": conf}
        h, w = res.orig_shape
        heights = boxes[:, 3] - boxes[:, 1] if len(boxes) else np.array([])
        return clean({"image": Path(path).name, "image_size": f"{w}x{h}", "people_detected": len(boxes),
                      "median_box_height_px": float(np.median(heights)) if len(heights) else None,
                      "small_people_under_40px": int((heights < 40).sum()) if len(heights) else 0})

    def _load_image_models(self):
        import torch
        from image_pipeline.img_common import load_backbone, Head
        ck = torch.load(IMAGE_MODELS_DIR / "heads.pt", map_location="cpu", weights_only=False)
        pretrained = os.environ.get("PS_NO_PRETRAINED") != "1"
        backbones = {b: load_backbone(b, pretrained=pretrained) for b in ck["backbones"]}
        heads = {}
        for name, h in ck["heads"].items():
            m = Head(h["in_dim"], len(ck["classes"]))
            m.load_state_dict(h["state_dict"])
            heads[name] = (m.eval(), h["parts"])
        self._img_models = {"backbones": backbones, "heads": heads, "stats": ck["stats"], "classes": ck["classes"]}

    def classify_people(self, image_path):
        import torch
        from PIL import Image, ImageDraw
        path = str(Path(image_path))
        if path not in self._detections:
            self.detect_people(path)
        if self._img_models is None:
            self._load_image_models()
        M = self._img_models
        boxes = self._detections[path]["boxes"]
        with Image.open(path) as im:
            img = im.convert("RGB")
        W, H = img.size
        crops = []
        for x1, y1, x2, y2 in boxes:
            px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
            c = img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(W, int(x2 + px)), min(H, int(y2 + py))))
            c.thumbnail((CROP_SAVE_MAX_SIDE, CROP_SAVE_MAX_SIDE))
            crops.append(c)
        labels = []
        if crops:
            feats = {}
            with torch.no_grad():
                for b, (model, tf) in M["backbones"].items():
                    out = []
                    for i in range(0, len(crops), 32):
                        out.append(model(torch.stack([tf(c) for c in crops[i:i + 32]])))
                    mean, std = M["stats"][b]
                    feats[b] = (torch.cat(out) - mean) / std
                probs = torch.stack([torch.softmax(h(torch.cat([feats[p] for p in parts], 1)), 1)
                                     for h, parts in M["heads"].values()]).mean(0)
            labels = [M["classes"][i] for i in probs.argmax(1).tolist()]
        self._detections[path]["labels"] = labels
        self._detections[path]["size"] = (W, H)
        counts = {c: labels.count(c) for c in IMAGE_CLASSES}
        people = len(labels) - counts["background"]
        hidden = counts["partially_visible"] / people if people else 0.0

        # annotated copy for the dashboard
        draw = ImageDraw.Draw(img)
        lw = max(2, W // 400)
        for (x1, y1, x2, y2), lab in zip(boxes, labels):
            draw.rectangle([x1, y1, x2, y2], outline=CLASS_COLORS[lab], width=lw)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        out = UPLOADS_DIR / f"annotated_{Path(path).stem}.jpg"
        img.save(out, quality=90)
        return clean({"image": Path(path).name, "people_detected": len(boxes),
                      "confirmed_people": people, "counts": counts,
                      "rejected_as_background": counts["background"],
                      "partly_hidden_share": hidden, "annotated_image": str(out)})

    def detections(self, image_path):
        """For the dashboard overlay (not an agent tool): boxes + labels of the last analysis."""
        d = self._detections.get(str(Path(image_path)))
        if not d:
            return None
        W, H = d.get("size", (None, None))
        labels = d.get("labels") or ["unknown"] * len(d["boxes"])
        return {"width": W, "height": H,
                "boxes": [{"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                           "label": lab, "conf": round(float(c), 3)}
                          for b, lab, c in zip(d["boxes"], labels, d["conf"])]}

    # ------------------------------------------------------------------ decision
    def assess_risk(self, people_count=None, area_m2=None, camera_id=None, hidden_share=None,
                    forecast_level=None, p_very_high=None, surge_ratio=None):
        if area_m2 is None and camera_id:
            cam = self.cameras[self.cameras["camera_id"].str.lower() == str(camera_id).lower()]
            if len(cam):
                area_m2 = float(cam["visible_area_m2"].iloc[0])
        out = _assess_risk(people_count, area_m2, hidden_share, forecast_level, p_very_high,
                           self.warn_thr, surge_ratio)
        out["inputs"] = {"people_count": people_count, "area_m2": area_m2, "hidden_share": hidden_share,
                         "forecast_level": forecast_level, "p_very_high": p_very_high, "surge_ratio": surge_ratio}
        return clean(out)
