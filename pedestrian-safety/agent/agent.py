"""
PHASE 5 - The crowd-safety AI agent
-----------------------------------
An LLM (Google Gemini, free tier) that PLANS which tools to call, reads their results,
and writes a short report. It never makes up numbers or decides the risk level itself:
numbers come from the tools, the risk level comes from assess_risk (fixed rules).

If there is no API key, or Gemini fails, a simple fixed plan runs the same tools
in a set order (offline backup), so the demo always works.

Safety against a slow or blocked Gemini:
  - every Gemini call gives up after LLM_TIMEOUT_S seconds,
  - at most one quick retry, and no retry at all when the DAILY quota is used up,
  - the whole Gemini run has a time budget (LLM_BUDGET_S); past it, the offline plan answers.

Run from the project folder:
  python agent/agent.py "Is Bou283_T going to be dangerous at 2026-09-25 13:00?"
  python agent/agent.py "Check this camera" --image data/widerperson/Images/000040.jpg --camera CAM-1
  python agent/agent.py "..." --offline          (skip the LLM)
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_GEMINI_MODEL, AGENT_LOGS_DIR
from agent.tools import Toolbox

MAX_STEPS = 10
LLM_TIMEOUT_S = 60      # one Gemini call may take at most this long
LLM_BUDGET_S = 120      # the whole Gemini run may take at most this long, then the offline plan answers
THINKING_LEVEL = "LOW"  # less "thinking" before each tool choice = much faster replies; set to None to turn off
RETRY_WAIT_S = 5        # wait before the single retry on a "busy" reply

# ------------------------------------------------------------------ tool descriptions for the LLM
TOOL_SPECS = [
    ("list_locations", "List the cameras and sensors the system knows, and the current replay time. "
                       "Call this first if you are unsure which location the user means.", {}, []),
    ("get_events", "Public holiday and listed events (festivals, matches, celebrations) on a date.",
     {"date": {"type": "string", "description": "Date as YYYY-MM-DD"}}, ["date"]),
    ("get_baseline", "What is NORMAL for a location at a given hour: usual people per hour and the "
                     "people-per-hour lines for Low/Medium/High/Very High.",
     {"location": {"type": "string", "description": "Camera id (e.g. CAM-1) or sensor name"},
      "when": {"type": "string", "description": "Date-time 'YYYY-MM-DD HH:MM'; omit for the current replay time"}},
     ["location"]),
    ("forecast_footfall", "Sensor model forecast of the crowd level for the given hour (Low/Medium/High/Very High) "
                          "with probabilities, recent counts and whether an early warning is raised.",
     {"location": {"type": "string", "description": "Camera id (e.g. CAM-1) or sensor name"},
      "when": {"type": "string", "description": "Hour to forecast 'YYYY-MM-DD HH:MM'; omit for the current replay time"}},
     ["location"]),
    ("detect_people", "Find every person in a camera image (YOLO). Returns how many were detected.",
     {"image_path": {"type": "string", "description": "Path of the uploaded camera image"}}, ["image_path"]),
    ("classify_people", "Label each detected person as pedestrian / rider / partly hidden (or reject as background) "
                        "using the trained image ensemble. Returns confirmed people and the partly-hidden share.",
     {"image_path": {"type": "string", "description": "Path of the uploaded camera image"}}, ["image_path"]),
    ("assess_risk", "Apply the official safety rules and return the risk level (Normal / Watch / Act now) with "
                    "reasons and actions. ALWAYS call this before stating any risk level. Pass whatever evidence you have.",
     {"people_count": {"type": "number", "description": "Confirmed people in the camera image"},
      "area_m2": {"type": "number", "description": "Visible ground area in m2 (if known)"},
      "camera_id": {"type": "string", "description": "Camera id, used to look up its area if area_m2 is not given"},
      "hidden_share": {"type": "number", "description": "Share of people partly hidden, 0-1"},
      "forecast_level": {"type": "string", "description": "Predicted level from forecast_footfall"},
      "p_very_high": {"type": "number", "description": "Probability of Very High from forecast_footfall"},
      "surge_ratio": {"type": "number", "description": "last_hour_vs_usual from forecast_footfall - pass it ONLY when "
                                                       "last_hour_level is High or Very High"}},
     []),
]

SYSTEM_PROMPT = """You are a crowd-safety assistant for city operators (police, event staff).
You decide which tools to call, then explain the result. Rules:
1. Never invent numbers. Every number you write must come from a tool result or the user's message.
2. Never decide the risk level yourself: always call assess_risk and report exactly the level it returns.
3. Call only the tools you need. A forecast question does not need the image tools.
   If an image path is given, call detect_people and then classify_people, and use confirmed_people
   and partly_hidden_share in assess_risk (with the camera_id or area).
4. If a location name is unclear, call list_locations first.
5. The current time is the replay time: {now}. Times the user gives without a year are in {year}.
Answer format (plain text, short):
RISK: <icon> <level>
SITUATION: one or two sentences.
EVIDENCE: 2-4 short bullet points with the key numbers.
ACTIONS: the actions returned by assess_risk."""


class CrowdAgent:
    def __init__(self, toolbox=None, client=None, model=None):
        self._emit = lambda event: None
        self._deadline = float("inf")
        self.tools = toolbox or Toolbox()
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.client = client
        if self.client is None and os.environ.get("GEMINI_API_KEY", "").strip() not in ("", "paste-your-key-here"):
            try:
                from google import genai
                from google.genai import types
                # timeout is in milliseconds: a call that hangs now fails instead of freezing the app
                self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip(),
                                           http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000))
            except Exception as e:                       # library missing etc.
                print(f"[agent] Gemini unavailable ({e}); using offline plan.")

    # -------------------------------------------------------------- shared
    def _call(self, name, args):
        fn = getattr(self.tools, name, None)
        if fn is None or name not in {s[0] for s in TOOL_SPECS}:
            return {"error": f"Unknown tool {name}"}
        try:
            return fn(**{k: v for k, v in (args or {}).items() if v is not None})
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, question, image_path=None, camera_id=None, when=None, offline=False, on_step=None):
        """on_step(event) is called live for every tool call (used by the dashboard to stream steps)."""
        self._emit = on_step or (lambda event: None)
        prompt = question
        if image_path:
            prompt += f"\n[Camera image uploaded: image_path='{image_path}'"
            prompt += f", camera_id='{camera_id}']" if camera_id else "]"
        elif camera_id:
            prompt += f"\n[Camera: {camera_id}]"
        if when:
            prompt += f"\n[Time of interest: {when}]"
        t0 = time.time()
        result = None
        if self.client is not None and not offline:
            try:
                result = self._run_llm(prompt)
            except Exception as e:
                print(f"[agent] LLM failed ({type(e).__name__}: {str(e)[:200]}); switching to offline plan.")
                self._emit({"type": "status", "text": "Gemini unavailable, using the offline plan"})
        if result is None:
            result = self._run_offline(question, image_path, camera_id, when)
        result.update({"question": question, "image_path": image_path, "camera_id": camera_id,
                       "when": when, "seconds": round(time.time() - t0, 1)})
        self._log(result)
        return result

    def _log(self, result):
        AGENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        f = AGENT_LOGS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        f.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # -------------------------------------------------------------- LLM loop (Gemini function calling)
    def _run_llm(self, prompt):
        from google.genai import types
        self._deadline = time.time() + LLM_BUDGET_S
        self._use_plain = False
        decls = [types.FunctionDeclaration(name=n, description=d, parameters_json_schema={
            "type": "object", "properties": props, "required": req}) for n, d, props, req in TOOL_SPECS]
        now = self.tools.replay_now
        settings = dict(
            system_instruction=SYSTEM_PROMPT.format(now=now.strftime("%Y-%m-%d %H:%M"), year=now.year),
            tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.1)
        config = None
        if THINKING_LEVEL:
            try:      # older library versions do not know thinking_level: then just skip it
                config = types.GenerateContentConfig(
                    **settings, thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL))
            except Exception:
                config = None
        self._plain_config = types.GenerateContentConfig(**settings)
        config = config or self._plain_config
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        steps = []
        for _ in range(MAX_STEPS):
            self._emit({"type": "status", "text": "Gemini is planning the next step"})
            resp = self._generate(contents, config)
            calls = resp.function_calls or []
            if not calls:
                return {"answer": (resp.text or "").strip(), "steps": steps, "mode": f"LLM ({self.model})",
                        "risk": self._last_risk(steps)}
            contents.append(resp.candidates[0].content)
            parts = []
            for fc in calls:
                args = dict(fc.args or {})
                self._emit({"type": "tool_start", "tool": fc.name, "args": args})
                out = self._call(fc.name, args)
                steps.append({"tool": fc.name, "args": args, "result": out})
                self._emit({"type": "step", **steps[-1]})
                parts.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
            contents.append(types.Content(role="user", parts=parts))   # tool results go back as "user"
        return {"answer": "Stopped: too many tool calls.", "steps": steps, "mode": f"LLM ({self.model})",
                "risk": self._last_risk(steps)}

    def _generate(self, contents, config, tries=2):
        """Call Gemini with a time limit. One quick retry when busy; never retry a used-up daily quota."""
        if self._use_plain:
            config = self._plain_config
        for i in range(tries):
            if time.time() > self._deadline:
                raise TimeoutError(f"Gemini took longer than {LLM_BUDGET_S}s in total")
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except Exception as e:
                msg = str(e)
                print(f"[agent] Gemini error: {type(e).__name__}: {msg[:200]}")
                if "400" in msg and "think" in msg.lower() and config is not self._plain_config:
                    print("[agent] This model does not accept the thinking setting; retrying without it.")
                    config = self._plain_config
                    self._use_plain = True
                    try:
                        return self.client.models.generate_content(model=self.model, contents=contents, config=config)
                    except Exception as e2:
                        msg = str(e2)
                        print(f"[agent] Gemini error: {type(e2).__name__}: {msg[:200]}")
                busy = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
                daily = "PerDay" in msg or "per day" in msg.lower()
                if i < tries - 1 and busy and not daily:
                    print(f"[agent] Gemini busy / rate-limited, retrying in {RETRY_WAIT_S}s ...")
                    self._emit({"type": "status", "text": "Gemini is busy, retrying once"})
                    time.sleep(RETRY_WAIT_S)
                else:
                    raise

    @staticmethod
    def _last_risk(steps):
        for s in reversed(steps):
            if s["tool"] == "assess_risk" and "risk_level" in s["result"]:
                return s["result"]
        return None

    # -------------------------------------------------------------- offline backup plan
    def _run_offline(self, question, image_path, camera_id, when):
        steps = []

        def do(name, **args):
            self._emit({"type": "tool_start", "tool": name, "args": args})
            out = self._call(name, args)
            steps.append({"tool": name, "args": args, "result": out})
            self._emit({"type": "step", **steps[-1]})
            return out

        # the camera and time written in the question win over the dropdown and the replay clock
        location = self._guess_location(question) or camera_id
        when = self._guess_time(question) or when
        cam_in_q = location if location in set(self.tools.cameras["camera_id"]) else None
        risk_args = {"camera_id": cam_in_q or camera_id}
        if image_path:
            do("detect_people", image_path=image_path)
            cls = do("classify_people", image_path=image_path)
            if "error" not in cls:
                risk_args.update(people_count=cls["confirmed_people"], hidden_share=cls["partly_hidden_share"])
        fc = None
        if location:
            fc = do("forecast_footfall", location=location, when=when)
            if "error" not in fc:
                risk_args.update(forecast_level=fc["predicted_level"], p_very_high=fc["p_very_high"])
                # a surge only matters when the place is already busy (small numbers make big ratios)
                if fc.get("last_hour_level") in ("High", "Very High") and fc.get("last_hour_vs_usual") is not None:
                    risk_args["surge_ratio"] = round(fc["last_hour_vs_usual"], 2)
        risk = do("assess_risk", **risk_args)
        return {"answer": self._template(risk, steps), "steps": steps, "mode": "offline plan", "risk": risk}

    def _guess_location(self, q):
        for cam in self.tools.cameras["camera_id"]:
            if cam.lower() in q.lower():
                return cam
        for name in self.tools.th["sensor_name"]:
            name = str(name)
            if len(name) > 3 and name.lower() in q.lower():
                return name
        return None

    @staticmethod
    def _guess_time(q):
        m = re.search(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?", q)
        return m.group(0) if m else None

    @staticmethod
    def _template(risk, steps):
        if "error" in risk:
            return f"Could not assess risk: {risk['error']}"
        ev = []
        for s in steps:
            r = s["result"]
            if "error" in r:
                ev.append(f"- {s['tool']} failed: {r['error']}")
            elif s["tool"] == "classify_people":
                ev.append(f"- Camera: {r['confirmed_people']} people confirmed "
                          f"({r['counts']['pedestrian']} walking, {r['counts']['rider']} riders, "
                          f"{r['counts']['partially_visible']} partly hidden = {r['partly_hidden_share']:.0%}).")
            elif s["tool"] == "forecast_footfall":
                ev.append(f"- Sensor {r['sensor']} at {r['forecast_for_hour']}: forecast {r['predicted_level']} "
                          f"(P(Very High) = {r['p_very_high']:.0%}); usual {r['usual_people_this_hour']:.0f} people/hour, "
                          f"last hour {r['people_1h_before']:.0f}.")
        if risk.get("density_per_m2") is not None:
            ev.append(f"- Density: {risk['density_per_m2']} people/m2.")
        return (f"RISK: {risk['icon']} {risk['risk_level']}\n"
                f"SITUATION: {' '.join(risk['reasons'])}\n"
                "EVIDENCE:\n" + "\n".join(ev) + "\n"
                "ACTIONS:\n" + "\n".join(f"- {a}" for a in risk["actions"]))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # emoji-safe printing on Windows
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--image")
    ap.add_argument("--camera")
    ap.add_argument("--when")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    agent = CrowdAgent()
    res = agent.run(a.question, a.image, a.camera, a.when, offline=a.offline)
    print(f"\n=== Mode: {res['mode']}  ({res['seconds']} s) ===")
    for i, s in enumerate(res["steps"], 1):
        print(f"\nStep {i}: {s['tool']}({json.dumps(s['args'])})")
        print("   -> " + json.dumps(s["result"], default=str)[:400])
    print("\n=== Answer ===\n" + res["answer"])


if __name__ == "__main__":
    main()