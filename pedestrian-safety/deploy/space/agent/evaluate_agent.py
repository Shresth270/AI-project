"""
PHASE 5B - Evaluate the agent on 20 test scenarios
--------------------------------------------------
For every scenario we check four things:
  1. Tool selection   : did it call every tool it MUST call, and none it must NOT call?
  2. Risk consistency : does the risk level written in the answer match assess_risk's result?
  3. Risk correctness : is that level the same as the reference plan (fixed tool order, no LLM)?
  4. Grounded numbers : are the numbers in the answer found in tool results / the question?
                        (catches made-up numbers)

Image scenarios use the most crowded WiderPerson test photos (picked automatically).

Run:  python agent/evaluate_agent.py              (uses Gemini; ~5-10 min because of free-tier limits)
      python agent/evaluate_agent.py --offline    (reference plan only, no API key needed)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import WIDERPERSON_DIR, TABLES_DIR
from agent.agent import CrowdAgent

LEVEL_NAMES = ["Normal", "Watch", "Act now"]


def crowded_images(n=6):
    """The n WiderPerson TEST images (val.txt) with the most annotated people."""
    root = next((p.parent for p in WIDERPERSON_DIR.rglob("Images") if (p.parent / "Annotations").is_dir()), None)
    if root is None:
        return []
    ids = [l.strip() for l in (root / "val.txt").read_text().splitlines() if l.strip()]
    sizes = []
    for i in ids:
        a = root / "Annotations" / f"{i}.jpg.txt"
        if a.exists():
            first = a.read_text().split("\n", 1)[0].strip()
            sizes.append((int(first) if first.isdigit() else 0, i))
    sizes.sort(reverse=True)
    return [str(root / "Images" / f"{i}.jpg") for _, i in sizes[:n]]


def numbers_in(text):
    """Numbers written in the answer (skip dates, times and list numbering)."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(r"CAM-\d+", " ", text)
    text = re.sub(r"[A-Za-z]+\d+[A-Za-z_]*", " ", text)        # sensor codes like Bou283_T
    vals = []
    for m in re.finditer(r"(\d+(?:,\d{3})*(?:\.\d+)?)(\s*%)?", text):
        v = float(m.group(1).replace(",", ""))
        vals.append(v / 100 if m.group(2) else v)
    return vals


def numbers_from(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            numbers_from(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_from(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(numbers_in(obj))
    return out


def grounded(v, known):
    for k in known:
        if abs(v - k) <= max(0.51, 0.011 * abs(k)) or abs(v - k * 100) <= 0.51 or abs(v * 100 - k) <= 0.51:
            return True
    return v in (0, 1, 2, 3, 4)          # small counts like "2 locations" / list items


def stated_level(answer):
    m = re.search(r"RISK:\s*\S*\s*(Normal|Watch|Act now)", answer, re.I)
    if m:
        return next(l for l in LEVEL_NAMES if l.lower() == m.group(1).lower())
    found = [l for l in LEVEL_NAMES if l.lower() in answer.lower()]
    return found[0] if len(found) == 1 else None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # emoji-safe printing on Windows
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between scenarios (free-tier limits)")
    args = ap.parse_args()

    scenarios = json.loads((ROOT / "agent" / "test_scenarios.json").read_text())
    images = crowded_images()
    agent = CrowdAgent()
    reference = CrowdAgent(toolbox=agent.tools, client=None)        # same tools, fixed plan
    if agent.client is None and not args.offline:
        print("[eval] No GEMINI_API_KEY in .env -> evaluating the offline plan only.")

    rows = []
    for sc in scenarios:
        image = None
        if sc["image"]:
            k = int(sc["image"].split(":")[1])
            if k > len(images):
                print(f"{sc['id']}: skipped (WiderPerson images not found)")
                continue
            image = images[k - 1]
        res = agent.run(sc["question"], image, sc["camera"], sc["when"], offline=args.offline)
        called = [s["tool"] for s in res["steps"]]
        missing = [t for t in sc["must_call"] if t not in called]
        extra = [t for t in sc["must_not_call"] if t in called]
        risk = res.get("risk")
        level = stated_level(res["answer"])
        consistent = None if risk is None else (level == risk["risk_level"])

        correct = None
        if "assess_risk" in sc["must_call"]:
            ref = reference.run(sc["question"], image, sc["camera"], sc["when"], offline=True)
            correct = (level == ref["risk"]["risk_level"]) if ref.get("risk") else None

        known = numbers_from([s["result"] for s in res["steps"]]) + numbers_in(sc["question"])
        nums = numbers_in(res["answer"])
        ungrounded = [v for v in nums if not grounded(v, known)]

        rows.append({"id": sc["id"], "note": sc["note"], "mode": res["mode"], "tools_called": " > ".join(called),
                     "tools_ok": not missing and not extra, "missing": ",".join(missing), "unexpected": ",".join(extra),
                     "risk_reported": level, "risk_consistent": consistent, "risk_correct": correct,
                     "numbers_in_answer": len(nums), "numbers_ungrounded": len(ungrounded),
                     "ungrounded_values": ",".join(f"{v:g}" for v in ungrounded[:5]), "seconds": res["seconds"]})
        r = rows[-1]
        print(f"{sc['id']} tools {'OK ' if r['tools_ok'] else 'BAD'} | risk {level or '-':<8} "
              f"consistent={consistent} correct={correct} | ungrounded {len(ungrounded)}/{len(nums)} | {res['seconds']}s")
        if not args.offline and agent.client is not None:
            time.sleep(args.pause)

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tag = "offline" if (args.offline or agent.client is None) else "llm"
    df.to_csv(TABLES_DIR / f"agent_evaluation_{tag}.csv", index=False)

    def rate(col):
        s = df[col].dropna()
        return f"{s.mean():.0%} ({int(s.sum())}/{len(s)})" if len(s) else "n/a"

    nums = df["numbers_in_answer"].sum()
    summary = {
        "scenarios": len(df),
        "fell_back_to_offline": int((df["mode"] == "offline plan").sum()) if tag == "llm" else "n/a",
        "tool_selection_correct": rate("tools_ok"),
        "risk_level_consistent_with_assess_risk": rate("risk_consistent"),
        "risk_level_matches_reference_plan": rate("risk_correct"),
        "numbers_grounded": f"{1 - df['numbers_ungrounded'].sum() / nums:.0%} of {nums}" if nums else "n/a",
        "avg_seconds": round(df["seconds"].mean(), 1),
    }
    (TABLES_DIR / f"agent_evaluation_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Agent evaluation summary ===")
    for k, v in summary.items():
        print(f"{k:<42} {v}")
    print(f"\nSaved: {TABLES_DIR / f'agent_evaluation_{tag}.csv'}")


if __name__ == "__main__":
    main()
