"""Keyword-based extraction of driving-experiment metadata.

Design
------
- Fully deterministic: no external services, no models.
- Returns a fixed schema so callers can trivially swap in an LLM later.
- Each field is scored independently; ``confidence`` is the mean of the
  individual field scores (0–1).

Returned dict schema
--------------------
{
    "task_type":          str | None,
    "driving_scenario":   str | None,
    "level_of_automation": str | None,
    "visual_cue":         str | None,
    "confidence":         float          # 0.0–1.0
}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: keyword scoring
# ---------------------------------------------------------------------------

@dataclass
class _Rule:
    label: str
    patterns: list[str]
    score: float = 1.0


def _score_text(text: str, rules: list[_Rule]) -> tuple[str | None, float]:
    """Return the best-matching label and a normalised score (0–1)."""
    low = text.lower()
    best_label: str | None = None
    best_score: float = 0.0
    for rule in rules:
        hits = sum(1 for p in rule.patterns if re.search(p, low))
        score = min(hits / max(len(rule.patterns), 1), 1.0) * rule.score
        if score > best_score:
            best_score = score
            best_label = rule.label
    # Normalise to [0, 1]
    return best_label, min(best_score, 1.0)


# ---------------------------------------------------------------------------
# Task-type rules
# ---------------------------------------------------------------------------

_TASK_RULES: list[_Rule] = [
    _Rule(
        "takeover_request",
        [
            r"\btakeover\b",
            r"\btake.?over request\b",
            r"\btor\b",
            r"\btime.to.takeover\b",
            r"\bttot\b",
            r"\bresponse.time\b",
            r"\bdriver.regain",
        ],
        score=1.0,
    ),
    _Rule(
        "lane_keeping",
        [
            r"\blane.keep",
            r"\blane.departure",
            r"\blateral.control",
            r"\blane.centering",
            r"\blane.position",
            r"\bsteering.wheel",
        ],
        score=1.0,
    ),
    _Rule(
        "hazard_detection",
        [
            r"\bhazard.detect",
            r"\bpedestrian.detect",
            r"\bobstacle.detect",
            r"\bcollision.warning",
            r"\bforward.collision",
            r"\bfcw\b",
            r"\bbraking\b.*\bwarning\b",
        ],
        score=1.0,
    ),
    _Rule(
        "car_following",
        [
            r"\bcar.follow",
            r"\bvehicle.follow",
            r"\btime.headway",
            r"\binter.vehicle.distance",
            r"\bacc\b",
            r"\badaptive.cruise",
        ],
        score=1.0,
    ),
    _Rule(
        "intersection_negotiation",
        [
            r"\bintersection",
            r"\bcrossing",
            r"\bright.of.way",
            r"\btraffic.signal",
            r"\bstop.sign",
        ],
        score=1.0,
    ),
    _Rule(
        "n_back_task",
        [
            r"\bn.back\b",
            r"\bsurrogate.secondary",
            r"\bsurrogate.reference",
            r"\bcognitive.load.task",
        ],
        score=0.9,
    ),
    _Rule(
        "driving_simulation",
        [
            r"\bdriving.simulation\b",
            r"\bdriving.simulator\b",
            r"\bsimulated.driving\b",
        ],
        score=0.7,
    ),
]


# ---------------------------------------------------------------------------
# Driving-scenario rules
# ---------------------------------------------------------------------------

_SCENARIO_RULES: list[_Rule] = [
    _Rule(
        "highway",
        [
            r"\bhighway\b",
            r"\bmotorway\b",
            r"\bfreeway\b",
            r"\bautobahn\b",
            r"\bhigh.speed.road\b",
        ],
        score=1.0,
    ),
    _Rule(
        "urban",
        [
            r"\burban\b",
            r"\bcity.road\b",
            r"\bcity.street\b",
            r"\bcity.traffic\b",
            r"\binner.city\b",
        ],
        score=1.0,
    ),
    _Rule(
        "rural",
        [
            r"\brural\b",
            r"\bcountry.road\b",
            r"\bsub.urban\b",
            r"\bsuburban\b",
        ],
        score=1.0,
    ),
    _Rule(
        "intersection",
        [
            r"\bintersection\b",
            r"\bjunction\b",
            r"\bcrossroads\b",
            r"\bt.junction\b",
        ],
        score=1.0,
    ),
    _Rule(
        "simulator",
        [
            r"\bdriving.simulator\b",
            r"\bsimulated.environment\b",
            r"\bvirtual.environment\b",
            r"\bsimulation.study\b",
            r"\bstatic.simulator\b",
            r"\bmotion.simulator\b",
        ],
        score=0.9,
    ),
    _Rule(
        "on_road",
        [
            r"\bon.road\b",
            r"\breal.world.driving\b",
            r"\bnaturalistic.driving\b",
            r"\btest.track\b",
            r"\bproving.ground\b",
            r"\binstrumented.vehicle\b",
        ],
        score=0.9,
    ),
    _Rule(
        "mixed",
        [
            r"\bmixed.traffic\b",
            r"\bmixed.road\b",
        ],
        score=0.7,
    ),
]


# ---------------------------------------------------------------------------
# Level-of-automation rules
# ---------------------------------------------------------------------------

_LOA_RULES: list[_Rule] = [
    _Rule(
        "SAE_L0",
        [
            r"\bsae.level.0\b",
            r"\blevel.0\b",
            r"\bno.automation\b",
            r"\bmanual.driving\b",
        ],
        score=1.0,
    ),
    _Rule(
        "SAE_L1",
        [
            r"\bsae.level.1\b",
            r"\blevel.1\b",
            r"\bdriver.assistance\b",
            r"\bacc\b",
        ],
        score=0.9,
    ),
    _Rule(
        "SAE_L2",
        [
            r"\bsae.level.2\b",
            r"\blevel.2\b",
            r"\bpartial.automation\b",
            r"\bsupervised.automation\b",
            r"\bcombined.lateral.longitudinal\b",
            r"\blane.keeping.*acc\b",
            r"\bacc.*lane.keep",
            r"\bautopilot\b",
        ],
        score=1.0,
    ),
    _Rule(
        "SAE_L3",
        [
            r"\bsae.level.3\b",
            r"\blevel.3\b",
            r"\bconditional.automation\b",
            r"\bhands.free\b",
            r"\beyes.off\b",
        ],
        score=1.0,
    ),
    _Rule(
        "SAE_L4",
        [
            r"\bsae.level.4\b",
            r"\blevel.4\b",
            r"\bhigh.automation\b",
            r"\bfully.automated.driving\b",
        ],
        score=1.0,
    ),
    _Rule(
        "SAE_L5",
        [
            r"\bsae.level.5\b",
            r"\blevel.5\b",
            r"\bfull.automation\b",
            r"\bfully.autonomous\b",
        ],
        score=1.0,
    ),
    _Rule(
        "assisted",
        [
            r"\badas\b",
            r"\badvanced.driver.assist",
            r"\bdriver.support",
            r"\bdriving.assist",
        ],
        score=0.8,
    ),
]


# ---------------------------------------------------------------------------
# Visual-cue rules
# ---------------------------------------------------------------------------

_VISUAL_CUE_RULES: list[_Rule] = [
    _Rule(
        "HUD",
        [
            r"\bhud\b",
            r"\bhead.up.display\b",
        ],
        score=1.0,
    ),
    _Rule(
        "AR_display",
        [
            r"\baugmented.reality\b",
            r"\bar\b.*\bdisplay\b",
            r"\bar.hud\b",
            r"\bar.windshield\b",
            r"\bwear.?able.*ar\b",
        ],
        score=1.0,
    ),
    _Rule(
        "dashboard",
        [
            r"\bdashboard\b",
            r"\binstrument.cluster\b",
            r"\bcentral.display\b",
            r"\binfotainment\b",
            r"\bin.vehicle.display\b",
        ],
        score=0.9,
    ),
    _Rule(
        "ambient_light",
        [
            r"\bambient.light",
            r"\bperipheral.light",
            r"\bledge.lighting\b",
            r"\bchromatic.warning\b",
            r"\bcolor.cue\b",
        ],
        score=1.0,
    ),
    _Rule(
        "auditory",
        [
            r"\bauditory\b",
            r"\baudible\b",
            r"\bbeep\b",
            r"\bsound.warning\b",
            r"\baudio.cue\b",
            r"\bspeech.cue\b",
        ],
        score=0.9,
    ),
    _Rule(
        "haptic",
        [
            r"\bhaptic\b",
            r"\bvibration\b",
            r"\bvibrotactile\b",
            r"\btactile.warning\b",
        ],
        score=0.9,
    ),
    _Rule(
        "iconography",
        [
            r"\bicon\b.*\bwarning\b",
            r"\bwarning.icon\b",
            r"\bsymbol.cue\b",
            r"\bpictogram\b",
        ],
        score=0.8,
    ),
    _Rule(
        "none_visual",
        [
            r"\bno.visual.cue\b",
            r"\bauditory.only\b",
            r"\bno.display\b",
        ],
        score=0.7,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(text: str, abstract: str = "") -> dict:
    """Extract driving-experiment metadata from *text*.

    If ``abstract`` is provided and non-empty it is appended so short abstracts
    alone can still contribute signal.

    Returns a dict with keys:
    ``task_type``, ``driving_scenario``, ``level_of_automation``,
    ``visual_cue``, ``confidence``.
    """
    combined = (text + "\n" + abstract).strip() if abstract else text

    task_label, task_score = _score_text(combined, _TASK_RULES)
    scen_label, scen_score = _score_text(combined, _SCENARIO_RULES)
    loa_label, loa_score = _score_text(combined, _LOA_RULES)
    cue_label, cue_score = _score_text(combined, _VISUAL_CUE_RULES)

    scores = [task_score, scen_score, loa_score, cue_score]
    confidence = round(sum(scores) / len(scores), 3)

    return {
        "task_type": task_label,
        "driving_scenario": scen_label,
        "level_of_automation": loa_label,
        "visual_cue": cue_label,
        "confidence": confidence,
    }
