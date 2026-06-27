"""Pydantic request/response schemas for the VQA API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskSceneRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    scene: Dict[str, Any] = Field(..., description="A synthetic scene spec (shapes) to answer about")


class AskResponse(BaseModel):
    question: str = ""
    answer: str = ""
    qtype: str = ""
    confidence: Optional[float] = None
    abstained: bool = False
    type_constrained: bool = False
    needs_review: bool = False
    candidates: List[List[Any]] = []
    model_version: str = ""
    status: str = ""
    disclaimer: str = ("Visual question answering is assistive: low-confidence answers are returned as "
                       "'unsure' and flagged for human review, never asserted as fact.")


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


__all__ = ["AskSceneRequest", "AskResponse", "HealthResponse"]
