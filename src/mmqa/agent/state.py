"""Shared state types for the VQA agent (deterministic FSM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class JobStatus(str, Enum):
    PENDING = "pending"
    CLASSIFIED = "classified"
    ANSWERED = "answered"
    COMPLETED = "completed"
    ABSTAINED = "abstained"            # the agent declined to answer (low confidence)
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str               # D1..D5
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class JobState:
    # ---- inputs --------------------------------------------------------------
    question: str = ""
    has_image: bool = False
    # ---- derived -------------------------------------------------------------
    status: JobStatus = JobStatus.PENDING
    qtype: str = "other"
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    raw_top1: str = ""
    confidence: Optional[float] = None
    margin: Optional[float] = None
    entropy: Optional[float] = None
    image_quality: Optional[float] = None
    # ---- outputs -------------------------------------------------------------
    answer: str = ""
    abstained: bool = False
    type_constrained: bool = False
    needs_review: bool = False
    rationale: str = ""
    # ---- audit ---------------------------------------------------------------
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question, "has_image": self.has_image, "status": self.status.value,
            "qtype": self.qtype, "candidates": [list(c) for c in self.candidates], "raw_top1": self.raw_top1,
            "confidence": self.confidence, "margin": self.margin, "entropy": self.entropy,
            "image_quality": self.image_quality, "answer": self.answer, "abstained": self.abstained,
            "type_constrained": self.type_constrained, "needs_review": self.needs_review,
            "rationale": self.rationale,
            "decisions": [d.to_dict() for d in self.decisions],
            "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics, "model_versions": self.model_versions,
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "JobState"]
