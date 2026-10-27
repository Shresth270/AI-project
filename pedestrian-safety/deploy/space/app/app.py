"""
PHASE 6 - Crowd Safety Console (Flask)
--------------------------------------
One web app over everything built so far:
  Overview      live crowd simulation + status of every camera at the replay clock
  Plan ahead    24-hour forecast grid for every camera on a chosen day
  Live watch    replay a day hour by hour: actual crowd, usual crowd, early warnings
  Camera check  upload a photo -> YOLO + image ensemble -> density, tightness, risk
  Ask the agent Gemini plans the tools; every step streams in live
  Model results all scores and charts from Phases 2-5

Run from the project folder:   python app/app.py      then open http://127.0.0.1:5000
"""
import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, Response, abort, jsonify, render_template, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
from config import FIGURES_DIR, TABLES_DIR, UPLOADS_DIR, FOOTFALL_LEVELS, SAMPLES_DIR
from agent.tools import Toolbox, clean, WEEKDAYS
from agent.agent import CrowdAgent
from agent.evaluate_agent import crowded_images

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

print("Loading models and data (about 10-20 s) ...")
TOOLS = Toolbox()
AGENT = CrowdAgent(toolbox=TOOLS)
HEAVY = threading.Lock()              # one image / agent job at a time (CPU models are shared)
# demo photos: the bundled folder (online copy) or the 12 most crowded WiderPerson test photos (laptop)
SAMPLES = sorted(str(p) for p in SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.is_dir() else []
SAMPLES = SAMPLES or crowded_images(12)
MAX_UPLOADS = 200                     # keep only the newest uploads so a public copy cannot fill the disk
TS = TOOLS.feats.index.get_level_values("ts")
DATA_START, DATA_END = TS.min(), TS.max()
print(f"Ready. Replay data {DATA_START:%d %b %Y} -> {DATA_END:%d %b %Y}, {len(TOOLS.th)} sensors, "
      f"{len(SAMPLES)} sample images, agent: {'Gemini ' + AGENT.model if AGENT.client else 'offline plan'}")


def when_arg(default=None):
    w = request.args.get("when") or request.form.get("when") or default
    return pd.Timestamp(w).floor("h") if w else TOOLS.replay_now


def moments():
    """Interesting replay times: the latest occurrence of every listed event inside the data, plus the latest hour."""
    out = []
    ev = TOOLS.events
    if not ev.empty:
        inside = ev[(ev["date"] >= DATA_START.normalize()) & (ev["date"] <= DATA_END)]
        inside = inside[~inside["event_name"].str.contains("after midnight", case=False)]
        for name, g in inside.sort_values("date").groupby("event_name"):
            r = g.iloc[-1]
            hour = int(min(r["end_hour"], r["start_hour"] + 3))
            out.append({"label": f"{name}, {r['date']:%d %b %Y}", "when": f"{r['date']:%Y-%m-%d}T{hour:02d}:00"})
    out.sort(key=lambda m: m["when"], reverse=True)
    out.append({"label": f"Latest data, {DATA_END:%d %b %Y %H:%M}", "when": f"{DATA_END:%Y-%m-%dT%H:%M}"})
    return out


def risk_from_forecast(fc):
    args = {"forecast_level": fc["predicted_level"], "p_very_high": fc["p_very_high"]}
    if fc.get("last_hour_level") in ("High", "Very High"):
        args["surge_ratio"] = fc.get("last_hour_vs_usual")
    return TOOLS.assess_risk(**args)


# ====================================================================== pages
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/figures/<path:name>")
def figures(name):
    return send_from_directory(FIGURES_DIR, name)


@app.route("/samples/<int:i>")
def sample(i):
    if i < 0 or i >= len(SAMPLES):
        abort(404)
    return send_file(SAMPLES[i])


# ====================================================================== API
@app.route("/api/meta")
def meta():
    cams = TOOLS.cameras.to_dict("records")
    return jsonify(clean({
        "cameras": cams, "sensor_count": len(TOOLS.th),
        "data_start": DATA_START, "data_end": DATA_END, "moments": moments(),
        "default_when": next((m["when"] for m in moments() if "parade" in m["label"].lower()), moments()[0]["when"]),
        "agent": {"llm": AGENT.client is not None, "model": AGENT.model},
        "samples": [{"id": i, "url": f"/samples/{i}", "name": Path(p).name} for i, p in enumerate(SAMPLES)],
        "warning_threshold": TOOLS.warn_thr, "levels": FOOTFALL_LEVELS,
    }))


@app.route("/api/status")
def status():
    ts = when_arg()
    cams = []
    for c in TOOLS.cameras.to_dict("records"):
        fc = TOOLS.forecast_footfall(c["camera_id"], ts)
        r = risk_from_forecast(fc)
        cams.append({"camera": c, "forecast": fc, "risk": r, "actual": TOOLS.actual_count(c["camera_id"], ts)})
    worst = max(cams, key=lambda x: x["risk"]["risk_code"]) if cams else None
    return jsonify(clean({"when": ts, "weekday": WEEKDAYS[ts.dayofweek], "cameras": cams,
                          "overall": worst["risk"] if worst else None,
                          "day": TOOLS.get_events(ts.strftime("%Y-%m-%d"))}))


@app.route("/api/plan")
def plan():
    day = pd.Timestamp(request.args.get("date") or TOOLS.replay_now).normalize()
    cams = TOOLS.cameras.to_dict("records")
    pairs = [(c["camera_id"], day + pd.Timedelta(hours=h)) for c in cams for h in range(24)]
    fc = TOOLS.forecast_many(pairs)
    rows = []
    for k, c in enumerate(cams):
        hours = [{"hour": h, "level": f["predicted_level"], "p_very_high": f["p_very_high"], "warning": f["early_warning"],
                  "usual": f["usual"], "actual_level": f["actual_level"], "replay": f["replay"]}
                 for h, f in enumerate(fc[k * 24:(k + 1) * 24])]
        rows.append({"camera": c, "hours": hours})
    return jsonify(clean({"date": day.strftime("%Y-%m-%d"), "weekday": WEEKDAYS[day.dayofweek], "rows": rows,
                          "day": TOOLS.get_events(day.strftime("%Y-%m-%d"))}))


@app.route("/api/watch")
def watch():
    loc = request.args.get("location") or TOOLS.cameras["camera_id"].iloc[0]
    end = when_arg()
    hours = min(int(request.args.get("hours", 30)), 72)
    sid = TOOLS._sensor(loc)
    th = TOOLS.th.loc[sid]
    stamps = [end - pd.Timedelta(hours=k) for k in range(hours - 1, -1, -1)]
    series = []
    for ts, f in zip(stamps, TOOLS.forecast_many([(loc, ts) for ts in stamps])):
        series.append({"ts": ts, "label": ts.strftime("%a %d %b, %H:%M"), "hour": ts.hour,
                       "actual": f["actual"], "actual_level": f["actual_level"], "usual": f["usual"],
                       "predicted_level": f["predicted_level"], "p_very_high": f["p_very_high"],
                       "probabilities": f["probabilities"], "warning": f["early_warning"], "event": f["event"]})
    return jsonify(clean({"location": loc, "sensor": th["sensor_name"],
                          "lines": {"p25": th["p25"], "p75": th["p75"], "p95": th["p95"]},
                          "warning_threshold": TOOLS.warn_thr, "series": series}))


def save_upload(fs):
    from PIL import Image
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    old = sorted(UPLOADS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - MAX_UPLOADS)]:
        p.unlink(missing_ok=True)
    out = UPLOADS_DIR / f"upload_{uuid.uuid4().hex[:10]}.jpg"
    with Image.open(fs.stream) as im:
        img = im.convert("RGB")
    img.thumbnail((1920, 1920))
    img.save(out, quality=92)
    return out


def image_from_request():
    if "image" in request.files and request.files["image"].filename:
        return save_upload(request.files["image"])
    sid = request.form.get("sample") or request.args.get("sample")
    if sid not in (None, ""):
        i = int(sid)
        if 0 <= i < len(SAMPLES):
            return Path(SAMPLES[i])
    tok = request.form.get("upload") or request.args.get("upload")
    if tok:
        p = UPLOADS_DIR / Path(tok).name
        if p.exists():
            return p
    return None


@app.route("/api/upload", methods=["POST"])
def upload():
    p = image_from_request()
    if p is None:
        return jsonify({"error": "Choose an image first."}), 400
    return jsonify({"upload": p.name, "url": f"/static/uploads/{p.name}"})


@app.route("/api/camera", methods=["POST"])
def camera():
    img = image_from_request()
    if img is None:
        return jsonify({"error": "Choose a sample image or upload a photo first."}), 400
    cam = request.form.get("camera_id") or TOOLS.cameras["camera_id"].iloc[0]
    area = request.form.get("area_m2")
    area = float(area) if area not in (None, "") else None
    ts = when_arg()
    with HEAVY:
        cls = TOOLS.classify_people(str(img))
        det = TOOLS.detections(str(img))
    if "error" in cls:
        return jsonify(cls), 500
    fc = TOOLS.forecast_footfall(cam, ts)
    args = {"people_count": cls["confirmed_people"], "hidden_share": cls["partly_hidden_share"],
            "camera_id": cam, "forecast_level": fc["predicted_level"], "p_very_high": fc["p_very_high"]}
    if area:
        args["area_m2"] = area
    risk = TOOLS.assess_risk(**args)
    url = f"/samples/{SAMPLES.index(str(img))}" if str(img) in SAMPLES else f"/static/uploads/{img.name}"
    return jsonify(clean({"image_url": url, "detections": det, "classes": cls, "forecast": fc, "risk": risk}))


@app.route("/api/agent/stream")
def agent_stream():
    question = (request.args.get("q") or "").strip()
    if not question:
        return jsonify({"error": "Type a question first."}), 400
    img = image_from_request()
    cam = request.args.get("camera") or None
    when = request.args.get("when") or None
    q = queue.Queue()

    def work():
        try:
            with HEAVY:
                q.put({"type": "status", "text": "Starting"})
                res = AGENT.run(question, str(img) if img else None, cam, when, on_step=q.put)
                det = TOOLS.detections(str(img)) if img else None
            url = None
            if img:
                url = f"/samples/{SAMPLES.index(str(img))}" if str(img) in SAMPLES else f"/static/uploads/{img.name}"
            q.put({"type": "done", "answer": res["answer"], "risk": res.get("risk"), "mode": res["mode"],
                   "seconds": res["seconds"], "image_url": url, "detections": det})
        except Exception as e:
            q.put({"type": "error", "text": f"{type(e).__name__}: {e}"})

    threading.Thread(target=work, daemon=True).start()

    def events():
        while True:
            ev = q.get()
            yield f"data: {json.dumps(clean(ev), default=str)}\n\n"
            if ev["type"] in ("done", "error"):
                break

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/results")
def results():
    def csv(name, **kw):
        f = TABLES_DIR / name
        return pd.read_csv(f, **kw) if f.exists() else None

    def js(name):
        f = TABLES_DIR / name
        return json.loads(f.read_text()) if f.exists() else None

    out = {}
    s = csv("sensor_model_comparison.csv", index_col=0)
    out["sensor_models"] = None if s is None else s.reset_index().to_dict("records")
    e = csv("early_warning_summary.csv", index_col=0)
    out["early_warning"] = None if e is None else e.reset_index().rename(columns={"index": "method"}).to_dict("records")
    ev = csv("early_warning_events.csv")
    out["early_warning_events"] = None if ev is None else ev.to_dict("records")
    im = csv("image_model_comparison.csv", index_col=0)
    out["image_models"] = None if im is None else im.reset_index().to_dict("records")
    out["agent_eval"] = {"llm": js("agent_evaluation_llm_summary.json"), "offline": js("agent_evaluation_offline_summary.json")}
    order = ["sensor_model_comparison", "sensor_confusion_best", "sensor_feature_importance", "early_warning_onset",
             "early_warning_timeline", "image_model_comparison", "image_per_class_f1", "image_confusion_ensemble",
             "sensor_weekday_hour_heatmap", "sensor_event_vs_normal", "sensor_level_distribution"]
    have = {p.stem for p in FIGURES_DIR.glob("*.png")} if FIGURES_DIR.exists() else set()
    out["figures"] = [f"/figures/{n}.png" for n in order if n in have] + \
                     [f"/figures/{n}.png" for n in sorted(have - set(order))]
    return jsonify(clean(out))


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
