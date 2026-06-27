"""The VQA agent - a deterministic FSM over the visual-question-answering pipeline.

    ingest (D1 input gate) -> classify (D2 question type) -> answer (D3 run VQA)
        -> calibrate (D4 abstention gate) -> constrain (D5 type-consistency gate)

Holds the VQA model (loaded once). Runs fully offline (SceneStubVQA reading the embedded
synthetic-scene spec + the keyword question-type classifier) and upgrades to a fine-tuned/
pretrained ViLT/BLIP model when present. When the model is uncertain the agent **abstains**
(answers "unsure" and flags for review) rather than emitting an overconfident guess, and it
**constrains** the answer to the question type (yes/no -> yes/no, count -> a number). Every
step is timed and traced; same input + same model + brain disabled => identical output.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import JsonlLogger, get_logger
from . import tools
from .llm_orchestrator import LLMBrain
from .state import JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


class VqaAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True, model=None):
        self.cfg = cfg or AppConfig()
        if model is None:
            from ..models.vqa_model import load_vqa_model
            model = load_vqa_model(self.cfg.model, prefer="model" if load_model else "stub")
        self.model = model
        self.brain = LLMBrain(self.cfg.agent)
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.job_log_path) if self.cfg.serving.log_jobs else None

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def run(self, *, image=None, scene: dict = None, image_path: str = "",
            question: str = "", save: bool = True) -> JobState:
        job = JobState(question=question or "")
        # resolve the image source
        img = None
        if image is not None:
            img = image
        elif scene is not None:
            from ..vision.image_utils import SceneImage
            img = SceneImage(scene)
        elif image_path:
            try:
                from ..vision.image_utils import load_image
                img = load_image(image_path)
            except Exception as exc:
                logger.info("could not load image %s (%s)", image_path, exc)
        job._image = img  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        job = self._step(job, "ingest", lambda: tools.tool_ingest(job, self.cfg), summary="input gate (D1)")
        if job.status is not JobStatus.FAILED:
            job = self._step(job, "classify", lambda: tools.tool_classify(job, self.cfg),
                             summary="question type (D2)")
            job = self._step(job, "answer", lambda: tools.tool_answer(job, self.cfg, model=self.model),
                             summary="run VQA (D3)")
            job = self._step(job, "calibrate", lambda: tools.tool_calibrate(job, self.cfg),
                             summary="abstention gate (D4)")
            job = self._step(job, "constrain", lambda: tools.tool_constrain(job, self.cfg),
                             summary="type-consistency (D5)")

        if self.brain.available() and (job.abstained or (job.confidence or 1.0) < 0.5):
            note = self.brain.review_note(job.question, job.answer, job.confidence)
            if note:
                job.metrics["brain_note"] = note
                job.metrics["brain_used"] = True

        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if hasattr(job, "_image"):
            delattr(job, "_image")
        if save and self._log is not None:
            try:
                self._log.log("vqa", question=job.question, qtype=job.qtype, answer=job.answer,
                              confidence=job.confidence, abstained=job.abstained,
                              needs_review=job.needs_review, status=job.status.value, metrics=job.metrics)
            except Exception:
                pass
        return job

    def ask(self, question: str, *, image=None, scene: dict = None, image_path: str = "") -> dict:
        job = self.run(image=image, scene=scene, image_path=image_path, question=question, save=False)
        return {"answer": job.answer, "qtype": job.qtype, "confidence": job.confidence,
                "abstained": job.abstained, "type_constrained": job.type_constrained,
                "needs_review": job.needs_review, "candidates": [list(c) for c in job.candidates],
                "model_version": job.model_versions.get("vqa", "?"), "status": job.status.value}


_AGENT: Optional[VqaAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> VqaAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = VqaAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["VqaAgent", "get_agent"]
