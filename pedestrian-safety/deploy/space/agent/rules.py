"""
Risk rules - the ONLY place a Normal / Watch / Act-now decision is made.
Plain, fixed rules (not the LLM), so every safety decision is predictable and explainable.
Thresholds live in config.RISK_RULES.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RISK_RULES

LEVELS = {0: ("Normal", "🟢"), 1: ("Watch", "🟡"), 2: ("Act now", "🔴")}

ACTIONS = {
    0: ["Continue routine monitoring."],
    1: ["Send extra staff to the location and keep the camera under watch.",
        "Prepare crowd barriers and announce alternative routes.",
        "Re-check the camera every 10-15 minutes."],
    2: ["Stop or slow entry into the area immediately.",
        "Open extra exits and keep all exit routes clear.",
        "Send staff and police to manage the crowd; use PA announcements to spread people out.",
        "Alert emergency services to stand by."],
}


def assess_risk(people_count=None, area_m2=None, hidden_share=None, forecast_level=None,
                p_very_high=None, warn_threshold=0.2, surge_ratio=None):
    """Combine camera evidence and sensor forecast into a risk level with reasons."""
    level, reasons = 0, []
    density = None
    if people_count is not None and area_m2:
        density = float(people_count) / float(area_m2)

    # ---- camera evidence (what is happening now)
    if density is not None:
        if density >= RISK_RULES["act_density"]:
            level = max(level, 2)
            reasons.append(f"Density {density:.2f} people/m2 is at or above the danger line "
                           f"({RISK_RULES['act_density']}/m2).")
        elif density >= RISK_RULES["act_density_tight"] and (hidden_share or 0) >= RISK_RULES["act_hidden_share"]:
            level = max(level, 2)
            reasons.append(f"Density {density:.2f} people/m2 with {hidden_share:.0%} of people partly hidden: "
                           f"the crowd is tightly packed.")
        elif density >= RISK_RULES["watch_density"]:
            level = max(level, 1)
            reasons.append(f"Density {density:.2f} people/m2 is above the watch line ({RISK_RULES['watch_density']}/m2).")
        else:
            reasons.append(f"Density {density:.2f} people/m2 is below the watch line ({RISK_RULES['watch_density']}/m2).")
    if hidden_share is not None and density is None and hidden_share >= RISK_RULES["act_hidden_share"]:
        level = max(level, 1)
        reasons.append(f"{hidden_share:.0%} of people are partly hidden (crowd may be tight; area unknown).")

    # ---- sensor evidence (what is about to happen)
    if forecast_level == "Very High" or (p_very_high is not None and p_very_high >= warn_threshold):
        level = max(level, 1)
        p = f" (probability {p_very_high:.0%})" if p_very_high is not None else ""
        reasons.append(f"Sensor model forecasts a Very High crowd in the next hour{p}.")
    elif forecast_level is not None:
        reasons.append(f"Sensor model forecasts {forecast_level} for the next hour.")
    if surge_ratio is not None:
        if surge_ratio >= RISK_RULES["watch_surge_ratio"]:
            level = max(level, 1)
            reasons.append(f"Crowd is {surge_ratio:.1f}x the usual level for this time.")
        else:
            reasons.append(f"Crowd is {surge_ratio:.1f}x the usual level for this time.")

    if not reasons:
        reasons.append("No camera or sensor evidence was provided.")
    name, icon = LEVELS[level]
    return {"risk_level": name, "icon": icon, "risk_code": level,
            "density_per_m2": None if density is None else round(density, 2),
            "reasons": reasons, "actions": ACTIONS[level]}
