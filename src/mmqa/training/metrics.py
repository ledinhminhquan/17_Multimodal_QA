"""VQA evaluation metrics: the official VQA accuracy (soft, 10-annotator) + answer
normalization + per-type breakdown. Pure-python, no heavy deps.

The official VQA accuracy for a prediction against the 10 human answers is the
leave-one-out average of ``min(1, matches/3)`` over the annotators, after the standard
answer normalization (lowercase, punctuation handling, article removal, number-word and
contraction mapping). See https://visualqa.org/evaluation.html .
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hes": "he's",
    "im": "i'm", "isnt": "isn't", "its": "it's", "lets": "let's", "shes": "she's",
    "thats": "that's", "theres": "there's", "theyre": "they're", "wasnt": "wasn't",
    "werent": "weren't", "whats": "what's", "wheres": "where's", "whos": "who's",
    "wont": "won't", "wouldnt": "wouldn't", "youre": "you're", "youve": "you've",
}
_MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
               "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def _process_punctuation(s: str) -> str:
    out = s
    for p in _PUNCT:
        if (p + " " in s or " " + p in s) or (re.search(_COMMA_STRIP, s) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def _process_digit_article(s: str) -> str:
    out = []
    for w in s.lower().split():
        w = _MANUAL_MAP.get(w, w)
        if w in _ARTICLES:
            continue
        out.append(w)
    for i, w in enumerate(out):
        if w in _CONTRACTIONS:
            out[i] = _CONTRACTIONS[w]
    return " ".join(out)


def normalize_answer(ans: str) -> str:
    a = (ans or "").replace("\n", " ").replace("\t", " ").strip().lower()
    a = _process_punctuation(a)
    a = _process_digit_article(a)
    return a.strip()


def vqa_accuracy_single(pred: str, gt_answers: Sequence[str]) -> float:
    """Official soft accuracy of one prediction vs the annotator answers."""
    p = normalize_answer(pred)
    gts = [normalize_answer(g) for g in gt_answers if g is not None]
    if not gts:
        return 0.0
    if len(gts) < 3:
        return 1.0 if any(p == g for g in gts) else 0.0
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        matches = sum(1 for g in others if g == p)
        accs.append(min(1.0, matches / 3.0))
    return sum(accs) / len(accs)


def vqa_accuracy(preds: Sequence[str], gts_list: Sequence[Sequence[str]]) -> float:
    if not preds:
        return 0.0
    return round(sum(vqa_accuracy_single(p, g) for p, g in zip(preds, gts_list)) / len(preds), 4)


def per_type_accuracy(preds: Sequence[str], gts_list: Sequence[Sequence[str]],
                      types: Sequence[str]) -> Dict[str, Any]:
    buckets: Dict[str, List[float]] = {}
    for p, g, t in zip(preds, gts_list, types):
        buckets.setdefault(t or "other", []).append(vqa_accuracy_single(p, g))
    return {t: {"accuracy": round(sum(v) / len(v), 4), "n": len(v)} for t, v in sorted(buckets.items())}


def exact_match(preds: Sequence[str], golds: Sequence[str]) -> float:
    if not preds:
        return 0.0
    n = sum(1 for p, g in zip(preds, golds) if normalize_answer(p) == normalize_answer(g))
    return round(n / len(preds), 4)


def answer_distribution(answers: Sequence[str], top: int = 10) -> List[Any]:
    c = Counter(normalize_answer(a) for a in answers if a)
    return c.most_common(top)


def vqa_metrics(preds: Sequence[str], gts_list: Sequence[Sequence[str]],
                types: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"accuracy": vqa_accuracy(preds, gts_list), "n": len(preds)}
    if types is not None:
        out["per_type"] = per_type_accuracy(preds, gts_list, types)
    return out


__all__ = ["normalize_answer", "vqa_accuracy_single", "vqa_accuracy", "per_type_accuracy",
           "exact_match", "answer_distribution", "vqa_metrics"]
