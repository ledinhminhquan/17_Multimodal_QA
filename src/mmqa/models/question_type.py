"""Question-type classification + per-type answer constraints.

A lightweight keyword classifier maps a question to one of a small taxonomy
(yes_no, number, color, object, location, person, reason, other). The type both
routes the agent (D2) and constrains the answer (D5): a yes/no question must be
answered yes/no, a counting question with a number, a colour question with a colour
word. This is the agent's value-add over a raw argmax.
"""

from __future__ import annotations

import re
from typing import List, Optional

COLORS = {
    "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown",
    "orange", "purple", "pink", "violet", "cyan", "magenta", "beige", "tan", "gold",
    "silver", "maroon", "navy", "teal", "turquoise",
}

_NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                 "nine", "ten", "eleven", "twelve", "none", "no", "many"}

_YESNO_STARTERS = ("is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
                   "can", "could", "will", "would", "should", "may", "might", "am")


def classify_question(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "other"
    if q.startswith(("how many", "how much")) or "what number" in q or q.startswith("count"):
        return "number"
    if "what color" in q or "what colour" in q or "which color" in q or "which colour" in q:
        return "color"
    if q.startswith("where"):
        return "location"
    if q.startswith("who"):
        return "person"
    if q.startswith("why"):
        return "reason"
    first = q.split()[0] if q.split() else ""
    if first in _YESNO_STARTERS:
        return "yes_no"
    if q.startswith(("what", "which", "name")):
        return "object"
    return "other"


def _is_number(ans: str) -> bool:
    a = (ans or "").strip().lower()
    return bool(re.fullmatch(r"\d+", a)) or a in _NUMBER_WORDS


def answer_matches_type(answer: str, qtype: str) -> bool:
    """Hard type-consistency check used by D5."""
    a = (answer or "").strip().lower()
    if not a:
        return False
    if qtype == "yes_no":
        return a in ("yes", "no")
    if qtype == "number":
        return _is_number(a)
    if qtype == "color":
        return any(c in a.split() or c == a for c in COLORS)
    return True   # object/location/person/reason/other: no hard constraint


def constrain_candidates(candidates: List[str], qtype: str) -> List[str]:
    """Filter top-k candidate answers to those consistent with the question type."""
    keep = [c for c in candidates if answer_matches_type(c, qtype)]
    return keep or candidates    # never empty: fall back to the raw list


def type_default(qtype: str) -> Optional[str]:
    """A safe default answer for a type (used when no candidate is consistent)."""
    return {"yes_no": "yes", "number": "1", "color": "white"}.get(qtype)


__all__ = ["COLORS", "classify_question", "answer_matches_type", "constrain_candidates",
           "type_default"]
