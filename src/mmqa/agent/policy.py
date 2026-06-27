"""Decision-point logic for the VQA agent (pure, testable).

Five explicit decision points on the model's intermediate outputs:
* **D1** input gate (a valid image + a non-empty question).
* **D2** question-type classification (routes the answer constraint) - in ``models/question_type``.
* **D3** answer (run the VQA model -> top-k candidates + softmax confidence).
* **D4** calibrated abstention gate (low max-prob / small margin / high entropy -> abstain;
  VQA models are overconfident, so this is the safety valve).
* **D5** type-consistency gate (constrain the answer to the question type; re-rank within top-k).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..config import AgentConfig


def input_gate(has_image: bool, question: str, image_quality: Optional[float], cfg: AgentConfig) -> Dict[str, Any]:
    q = (question or "").strip()
    if not has_image:
        return {"ok": False, "branch": "no_image"}
    if len(q) < cfg.min_question_chars:
        return {"ok": False, "branch": "no_question"}
    if image_quality is not None and image_quality < 0.02:
        return {"ok": True, "branch": "blank_image_warn", "quality": image_quality}
    return {"ok": True, "branch": "ok", "quality": image_quality}


def abstain_gate(confidence: Optional[float], margin: Optional[float], entropy: Optional[float],
                 cfg: AgentConfig) -> Dict[str, Any]:
    if not cfg.abstain_enabled:
        return {"ok": True, "branch": "disabled"}
    if confidence is not None and confidence < cfg.confidence_min:
        return {"ok": False, "branch": "low_confidence", "confidence": confidence}
    if margin is not None and margin < cfg.margin_min:
        return {"ok": False, "branch": "low_margin", "margin": margin}
    if entropy is not None and entropy > cfg.entropy_max:
        return {"ok": False, "branch": "high_entropy", "entropy": entropy}
    return {"ok": True, "branch": "ok", "confidence": confidence}


def type_consistency_gate(candidates: List[Tuple[str, float]], qtype: str,
                          cfg: AgentConfig) -> Dict[str, Any]:
    """Pick the highest-scoring candidate consistent with the question type."""
    from ..models.question_type import answer_matches_type, type_default
    if not cfg.type_consistency_enabled or not candidates:
        top = candidates[0][0] if candidates else ""
        return {"answer": top, "constrained": False, "branch": "disabled" if candidates else "empty"}
    top = candidates[0][0]
    if answer_matches_type(top, qtype):
        return {"answer": top, "constrained": False, "branch": "consistent"}
    for ans, _ in candidates:
        if answer_matches_type(ans, qtype):
            return {"answer": ans, "constrained": True, "branch": "reranked"}
    default = type_default(qtype)
    if default is not None:
        return {"answer": default, "constrained": True, "branch": "type_default"}
    return {"answer": top, "constrained": False, "branch": "no_consistent_candidate"}


__all__ = ["input_gate", "abstain_gate", "type_consistency_gate"]
