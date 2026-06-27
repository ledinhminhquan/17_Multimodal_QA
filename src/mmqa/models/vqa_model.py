"""The VQA core - the trained model + a dependency-light offline stub.

* ``SceneStubVQA`` - answers from the synthetic scene spec embedded in the image
  (no torch, no real model): the offline floor + the fallback that lets the whole
  pipeline run for tests/CI, and that produces honest numbers on the synthetic set.
* ``ViltVQA`` - wraps a fine-tuned/pretrained ViLT classification model
  (``dandelin/vilt-b32-finetuned-vqa``): returns top-k (answer, probability).
* ``BlipVQA`` - wraps a generative BLIP VQA model (single answer).

All expose ``answer(image, question, top_k) -> AnswerResult`` plus ``name`` / ``version``.
``load_vqa_model`` picks the best available (fine-tuned > pretrained > stub).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import ModelConfig
from ..logging_utils import get_logger
from ..models.model_registry import resolve_latest
from . import question_type as QT

logger = get_logger(__name__)


@dataclass
class AnswerResult:
    answers: List[Tuple[str, float]] = field(default_factory=list)   # (answer, score) desc
    model: str = "stub"
    version: str = "stub-1.0"

    @property
    def top1(self) -> str:
        return self.answers[0][0] if self.answers else ""

    @property
    def confidence(self) -> float:
        return float(self.answers[0][1]) if self.answers else 0.0

    @property
    def margin(self) -> float:
        if len(self.answers) < 2:
            return self.confidence
        return round(float(self.answers[0][1] - self.answers[1][1]), 4)

    @property
    def entropy(self) -> float:
        ps = [max(1e-9, float(s)) for _, s in self.answers]
        z = sum(ps) or 1.0
        ps = [p / z for p in ps]
        return round(-sum(p * math.log(p) for p in ps), 4)


class SceneStubVQA:
    name = "scene_stub"
    version = "stub-1.0"

    def __init__(self, cfg: Optional[ModelConfig] = None):
        self.cfg = cfg

    def answer(self, image, question: str, top_k: int = 5) -> AnswerResult:
        from ..data.synth_scene import scene_answer
        from ..vision.image_utils import read_scene
        scene = read_scene(image)
        qtype = QT.classify_question(question)
        if scene is not None:
            ans = scene_answer(scene, question)
            if ans:
                cands = [(ans, 0.95)]
                # add a plausible distractor for margin/entropy realism
                if qtype == "yes_no":
                    cands.append(("no" if ans == "yes" else "yes", 0.05))
                elif qtype == "number":
                    cands.append((str(max(0, int(ans) - 1)) if ans.isdigit() else "1", 0.04))
                return AnswerResult(cands, self.name, self.version)
        # unparseable -> a low-confidence type-default so the agent abstains
        default = QT.type_default(qtype) or "yes"
        return AnswerResult([(default, 0.30), ("no", 0.25)], self.name, self.version)


class ViltVQA:
    name = "vilt"

    def __init__(self, model, processor, cfg: ModelConfig, version: str = "vilt-1.0"):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.version = version
        self._device = self._to_device()
        self.id2label = getattr(model.config, "id2label", {}) or {}

    def _to_device(self):
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(dev)
            return dev
        except Exception:
            return "cpu"

    @classmethod
    def from_pretrained(cls, model_path: str, cfg: ModelConfig) -> "ViltVQA":
        from transformers import ViltForQuestionAnswering, ViltProcessor  # lazy
        processor = ViltProcessor.from_pretrained(model_path)
        model = ViltForQuestionAnswering.from_pretrained(model_path).eval()
        return cls(model, processor, cfg, version=_read_version(model_path))

    def answer(self, image, question: str, top_k: int = 5) -> AnswerResult:
        import torch
        enc = self.processor(image, question, return_tensors="pt",
                             truncation=True, max_length=self.cfg.max_question_length).to(self._device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.shape[-1])
        top = torch.topk(probs, k)
        answers = [(self.id2label.get(int(i), str(int(i))), float(p))
                   for p, i in zip(top.values.tolist(), top.indices.tolist())]
        return AnswerResult(answers, self.name, self.version)


class BlipVQA:
    name = "blip"

    def __init__(self, model, processor, cfg: ModelConfig, version: str = "blip-1.0"):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.version = version
        self._device = self._to_device()

    def _to_device(self):
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(dev)
            return dev
        except Exception:
            return "cpu"

    @classmethod
    def from_pretrained(cls, model_path: str, cfg: ModelConfig) -> "BlipVQA":
        from transformers import BlipForQuestionAnswering, BlipProcessor  # lazy
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForQuestionAnswering.from_pretrained(model_path).eval()
        return cls(model, processor, cfg, version=_read_version(model_path))

    def answer(self, image, question: str, top_k: int = 5) -> AnswerResult:
        import torch
        enc = self.processor(image, question, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=10)
        text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return AnswerResult([(text, 0.9)], self.name, self.version)


def _read_version(model_path: str) -> str:
    meta = Path(model_path) / "model_meta.json"
    if meta.exists():
        try:
            import json
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "vqa-1.0")
        except Exception:
            pass
    return "vqa-base"


def load_vqa_model(cfg: ModelConfig, *, prefer: str = "model"):
    """Fine-tuned ViLT > pretrained ViLT/BLIP > scene stub."""
    if prefer == "stub":
        return SceneStubVQA(cfg)
    wrapper = ViltVQA if cfg.model_type != "blip" else BlipVQA
    latest = resolve_latest(cfg.output_dir)
    if latest is not None:
        try:
            return wrapper.from_pretrained(str(latest), cfg)
        except Exception as exc:
            logger.info("fine-tuned VQA unavailable (%s); trying pretrained base.", exc)
    try:
        return wrapper.from_pretrained(cfg.base_model, cfg)
    except Exception as exc:
        logger.info("pretrained VQA unavailable (%s); using scene stub.", exc)
    return SceneStubVQA(cfg)


__all__ = ["AnswerResult", "SceneStubVQA", "ViltVQA", "BlipVQA", "load_vqa_model"]
