"""Built-in seed dataset (offline fallback + tests).

A handful of fixed synthetic **scenes** (colored shapes) with templated questions and gold
answers, plus a small common-answer vocabulary. Everything is plain dicts (no PIL needed),
so tests / eval / the agent run with no torch, no image rendering and no network. The
``SceneStubVQA`` model answers each question from the scene spec, giving honest offline
numbers. On Colab the real VQAv2 images + a ViLT/BLIP model replace this.
"""

from __future__ import annotations

from typing import Dict, List

from . import synth_scene as S

# A few fixed, hand-set scenes (deterministic, no rendering required).
SEED_SCENES: List[Dict] = [
    {"width": 384, "height": 384, "shapes": [
        {"shape": "square", "color": "red", "bbox": [30, 40, 90, 90]},
        {"shape": "circle", "color": "blue", "bbox": [200, 60, 100, 100]},
        {"shape": "square", "color": "green", "bbox": [120, 240, 80, 80]},
    ]},
    {"width": 384, "height": 384, "shapes": [
        {"shape": "circle", "color": "yellow", "bbox": [50, 50, 120, 120]},
        {"shape": "triangle", "color": "purple", "bbox": [220, 200, 110, 110]},
    ]},
    {"width": 384, "height": 384, "shapes": [
        {"shape": "triangle", "color": "orange", "bbox": [140, 120, 110, 110]},
    ]},
    {"width": 384, "height": 384, "shapes": [
        {"shape": "square", "color": "blue", "bbox": [20, 30, 80, 80]},
        {"shape": "square", "color": "blue", "bbox": [180, 40, 90, 90]},
        {"shape": "circle", "color": "red", "bbox": [120, 220, 100, 100]},
        {"shape": "triangle", "color": "green", "bbox": [260, 230, 90, 90]},
    ]},
]

# Common-answer vocabulary (the prior baseline + an offline label space).
ANSWER_VOCAB: List[str] = (
    ["yes", "no"]
    + [str(i) for i in range(0, 11)]
    + sorted(S.PALETTE.keys())
    + ["square", "circle", "triangle", "shape", "left", "right", "top", "bottom", "none"]
)

MOST_COMMON_ANSWER = "yes"   # the global VQAv2 prior


def scenes() -> List[Dict]:
    import copy
    return copy.deepcopy(SEED_SCENES)


def seed_examples() -> List[Dict]:
    """(scene, question, answers, gold, qtype, answer_type) tuples for the seed scenes."""
    out: List[Dict] = []
    for i, scene in enumerate(SEED_SCENES):
        for q in S.scene_questions(scene, seed=100 + i):
            out.append({"scene": scene, **q})
    return out


def answer_vocab() -> List[str]:
    return list(ANSWER_VOCAB)


__all__ = ["SEED_SCENES", "ANSWER_VOCAB", "MOST_COMMON_ANSWER", "scenes", "seed_examples",
           "answer_vocab"]
