# intents.py
import re

_PATTERNS = [
    r"\b(speak|talk|chat|call|phone)\b.*\b(human|person|someone|manager|staff|trainer|rep|representative)\b",
    r"\b(human please|talk to a human|speak with someone|speak to manager|live agent|customer service)\b",
]

def wants_handoff(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(re.search(p, t) for p in _PATTERNS)
