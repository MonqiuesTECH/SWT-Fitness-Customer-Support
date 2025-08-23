# intents.py
"""
Intent detection for human handoff (Calendly flow).

Exports:
    wants_handoff(text: str) -> bool

Detects when a user wants to speak to a person / manager / live agent,
or to book/schedule a call. This is used to short-circuit the chat flow
and render Calendly slots instead of sending the query to the bot.
"""

from __future__ import annotations
import re
from typing import Iterable

# Phrases that strongly indicate a human handoff
# We keep them as regexes to handle varied wording and word order.
_PATTERNS: Iterable[re.Pattern] = [
    # Direct requests to speak/talk with a human/person/manager/agent/rep
    re.compile(r"\b(speak|talk|chat)\b.*\b(human|person|someone|manager|staff|trainer|agent|rep|representative)\b", re.I),
    re.compile(r"\b(human|person|someone|manager|staff|trainer|agent|rep|representative)\b.*\b(speak|talk|chat)\b", re.I),
    re.compile(r"\b(speak|talk)\s+to\s+(a\s+)?(real\s+)?(human|person|manager|agent|rep|representative)\b", re.I),
    re.compile(r"\bconnect\s+me\s+to\s+(a\s+)?(human|person|manager|agent|rep|representative)\b", re.I),
    re.compile(r"\b(live\s+agent|live\s+person|customer\s+service|operator)\b", re.I),

    # Call/phone requests that imply a human
    re.compile(r"\b(call|phone)\s+me\b", re.I),
    re.compile(r"\b(i\s+want\s+to\s+)?(book|schedule|set\s+up)\s+(a\s+)?(call|phone\s+call)\b", re.I),
    re.compile(r"\b(can\s+i|may\s+i)\s+(talk|speak)\s+to\s+(someone|a\s+person|a\s+human|a\s+manager)\b", re.I),

    # Short, strong signals
    re.compile(r"^\s*(human\s+please|talk\s+to\s+a\s+human|speak\s+with\s+someone|real\s+person)\s*$", re.I),
]

# Optional hard negatives to reduce false positives (rare)
# Example: "Is human anatomy covered in class?" should not trigger.
_HARD_NEGATIVES: Iterable[re.Pattern] = [
    re.compile(r"\bhuman\s+(anatomy|body|performance|biology)\b", re.I),
]


def wants_handoff(text: str | None) -> bool:
    """
    Return True if the message should trigger a human handoff.

    Heuristics:
      - Looks for verbs like speak/talk/chat/call combined with human/person/manager/etc.
      - Accepts common short forms like "human please", "live agent", "customer service".
      - Ignores a few known false-positive academic phrases.
    """
    if not text:
        return False

    t = text.strip()
    if not t:
        return False

    for neg in _HARD_NEGATIVES:
        if neg.search(t):
            return False

    return any(p.search(t) for p in _PATTERNS)


# Optional: quick smoke test when running this file directly
if __name__ == "__main__":
    tests_true = [
        "I want to speak to a person",
        "Can I talk to a human please?",
        "connect me to a manager",
        "live agent",
        "customer service",
        "Book a call with someone",
        "phone me about pricing",
    ]
    tests_false = [
        "Is human anatomy covered in class?",
        "Talk me through HIIT vs low impact",
        "What time are classes tonight?",
    ]
    for s in tests_true:
        assert wants_handoff(s), f"Should be True: {s}"
    for s in tests_false:
        assert not wants_handoff(s), f"Should be False: {s}"
    print("intents.py self-test passed.")
