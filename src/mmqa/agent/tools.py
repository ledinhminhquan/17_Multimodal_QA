"""Agent tools - each operates on the JobState and returns it.

Tools wrap the VQA model + the question-type router + the D1-D5 policy. They run against
the offline stack (SceneStubVQA reading the embedded scene + the keyword question-type
classifier) so the whole pipeline runs offline for tests/CI, and upgrade to a real
ViLT/BLIP model when present. The orchestrator wraps each call with timing/trace; tools
never raise past it.
"""

from __future__ import annotations

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models import question_type as QT
from ..vision.image_utils import image_quality, is_valid_image
from . import policy
from .state import Decision, JobState, JobStatus

logger = get_logger(__name__)


def tool_ingest(job: JobState, cfg: AppConfig) -> JobState:
    """D1 - input gate: a valid image + a non-empty question."""
    image = getattr(job, "_image", None)
    job.has_image = is_valid_image(image) if image is not None else False
    if job.has_image:
        try:
            job.image_quality = round(image_quality(image), 4)
        except Exception:
            job.image_quality = None
    gate = policy.input_gate(job.has_image, job.question, job.image_quality, cfg.agent)
    if not gate["ok"]:
        job.status = JobStatus.FAILED
        job.answer = ""
        job.rationale = f"Cannot answer: {gate['branch']}."
    job.add_decision(Decision("D1", "input_gate", gate["branch"], score=job.image_quality,
                              detail=f"has_image={job.has_image}, quality={job.image_quality}"))
    return job


def tool_classify(job: JobState, cfg: AppConfig) -> JobState:
    """D2 - question-type classification (routes the answer constraint)."""
    job.qtype = QT.classify_question(job.question)
    job.add_decision(Decision("D2", "question_type", job.qtype, detail=f"type={job.qtype}"))
    job.status = JobStatus.CLASSIFIED
    return job


def tool_answer(job: JobState, cfg: AppConfig, *, model) -> JobState:
    """D3 - run the VQA model -> top-k candidates + confidence."""
    image = getattr(job, "_image", None)
    res = model.answer(image, job.question, top_k=cfg.model.top_k)
    job.candidates = [(a, round(float(s), 4)) for a, s in res.answers]
    job.raw_top1 = res.top1
    job.confidence = round(res.confidence, 4)
    job.margin = res.margin
    job.entropy = res.entropy
    job.model_versions["vqa"] = f"{getattr(model, 'name', '?')}:{getattr(model, 'version', '?')}"
    job.add_decision(Decision("D3", "answer", "ok" if job.candidates else "empty",
                              score=job.confidence,
                              detail=f"top1='{job.raw_top1}' conf={job.confidence} margin={job.margin}"))
    job.status = JobStatus.ANSWERED
    return job


def tool_calibrate(job: JobState, cfg: AppConfig) -> JobState:
    """D4 - calibrated abstention gate (VQA models are overconfident)."""
    gate = policy.abstain_gate(job.confidence, job.margin, job.entropy, cfg.agent)
    if not gate["ok"]:
        job.abstained = True
        job.needs_review = True
        job.answer = "unsure"
        job.status = JobStatus.ABSTAINED
    job.add_decision(Decision("D4", "abstain_gate", gate["branch"], score=job.confidence,
                              detail=f"abstained={job.abstained} (conf={job.confidence}, "
                                     f"margin={job.margin}, entropy={job.entropy})"))
    return job


def tool_constrain(job: JobState, cfg: AppConfig) -> JobState:
    """D5 - type-consistency gate: constrain the answer to the question type."""
    if job.abstained:
        job.add_decision(Decision("D5", "type_consistency", "abstained_skip",
                                  detail="abstained -> answer='unsure'"))
        if not job.rationale:
            job.rationale = (f"Abstained on a {job.qtype} question (low confidence "
                             f"{job.confidence}); flagged for human review.")
        return job
    gate = policy.type_consistency_gate(job.candidates, job.qtype, cfg.agent)
    job.answer = gate["answer"]
    job.type_constrained = bool(gate["constrained"])
    job.add_decision(Decision("D5", "type_consistency", gate["branch"],
                              detail=f"answer='{job.answer}' constrained={job.type_constrained}"))
    job.status = JobStatus.NEEDS_REVIEW if job.needs_review else JobStatus.COMPLETED
    if not job.rationale:
        job.rationale = (f"Answered '{job.answer}' to a {job.qtype} question via "
                         f"{job.model_versions.get('vqa', 'VQA')} (confidence {job.confidence}"
                         + (", type-constrained" if job.type_constrained else "") + ").")
    return job


__all__ = ["tool_ingest", "tool_classify", "tool_answer", "tool_calibrate", "tool_constrain"]
