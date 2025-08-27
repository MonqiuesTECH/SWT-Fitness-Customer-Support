# intents.py — simple intent router + handoff detector

import re
from typing import Literal

_HANDOFF_PATTERNS = [
    r"\b(speak|talk|chat)\s+with\s+(a|the)?\s*(human|person|manager|staff|rep|someone)\b",
    r"\b(call|phone)\s+(me|us|someone)\b",
    r"\b(human|live\s*agent)\b",
]

def wants_handoff(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t, re.I) for p in _HANDOFF_PATTERNS)

def detect_intent(text: str) -> Literal["membership", "schedule", "general"]:
    t = text.lower()

    if re.search(r"\b(member(ship)?|price|cost|pricing|monthly|plan|tier)\b", t):
        return "membership"

    if re.search(r"\b(class|schedule|timetable|what'?s on|when|time)\b", t):
        # avoid false positive when talking about pricing times, etc.
        if not re.search(r"\b(price|cost)\b", t):
            return "schedule"

    return "general"
