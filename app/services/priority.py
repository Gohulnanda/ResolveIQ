"""Rule-based priority prediction.

Strategy (documented in README): priority is a transparent points heuristic, not a
trained model, because the sample dataset carries no priority labels. Points come from
(1) signals in the ticket text and (2) the requester's own urgency/impact, which are
kept separate from the prediction in API responses.

Each signal bucket counts at most once. Total points map to a level:
    >= 7 -> P1 Critical,  >= 4 -> P2 High,  >= 2 -> P3 Medium,  otherwise P4 Low.
"""

from __future__ import annotations

import re

from app.models.schemas import Impact, PriorityPrediction, Urgency

_TEXT_BUCKETS: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    (
        "outage or work stoppage",
        3,
        (
            r"outage", r"\bis down\b", r"\bare down\b", r"went down", r"completely down",
            r"cannot work", r"can'?t work", r"unable to work", r"data loss", r"corrupt",
            r"ransomware", r"breach", r"malware", r"security incident", r"compromised",
        ),
    ),
    (
        "wide scope (many users affected)",
        2,
        (
            r"everyone", r"all users", r"all staff", r"entire", r"whole (team|office|floor)",
            r"multiple users", r"several (people|users)", r"many users", r"company[- ]wide",
            r"department", r"\bfloor\b",
        ),
    ),
    (
        "business-critical system",
        2,
        (r"production", r"payroll", r"billing", r"checkout", r"customer[- ]facing", r"\berp\b"),
    ),
    (
        "degraded rather than failed",
        1,
        (r"\bslow", r"intermittent", r"sometimes", r"occasionally", r"degraded", r"lag"),
    ),
    (
        "workaround available",
        -1,
        (r"workaround", r"can still", r"able to work", r"in the meantime", r"not urgent"),
    ),
)

_URGENCY_POINTS: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_IMPACT_POINTS: dict[str, int] = {"individual": 0, "team": 1, "department": 2, "organization": 3}

_LEVELS = (
    (7, "P1", "Critical"),
    (4, "P2", "High"),
    (2, "P3", "Medium"),
    (-99, "P4", "Low"),
)


def predict_priority(
    text: str, urgency: Urgency | None = None, impact: Impact | None = None
) -> PriorityPrediction:
    lowered = text.lower()
    score = 0
    reasoning: list[str] = []

    for name, points, patterns in _TEXT_BUCKETS:
        matched = next((p for p in patterns if re.search(p, lowered)), None)
        if matched:
            score += points
            reasoning.append(f"Text signal: {name} ({points:+d})")

    if urgency:
        points = _URGENCY_POINTS[urgency]
        score += points
        reasoning.append(f"Requester-stated urgency '{urgency}' ({points:+d})")
    if impact:
        points = _IMPACT_POINTS[impact]
        score += points
        reasoning.append(f"Requester-stated impact '{impact}' ({points:+d})")

    if not reasoning:
        reasoning.append("No priority signals found in the text and no urgency/impact supplied.")

    level, label = next((lvl, lab) for threshold, lvl, lab in _LEVELS if score >= threshold)
    return PriorityPrediction(level=level, label=label, score=score, reasoning=reasoning)
