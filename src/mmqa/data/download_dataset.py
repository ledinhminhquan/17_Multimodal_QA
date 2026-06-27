"""Prefetch + sanity-check the datasets (no large files committed).

Streaming probes confirm the VQAv2 train/eval mirrors are reachable and report their
schema WITHOUT downloading them in full. Degrades gracefully: the synthetic seed always
works. Optionally renders the synthetic eval scenes.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _probe(loader) -> Dict[str, Any]:
    try:
        return {"ok": True, **loader()}
    except Exception as exc:  # pragma: no cover - network dependent
        return {"ok": False, "error": str(exc)}


def download_all(cfg: AppConfig, render_synthetic: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"train": {}, "eval": {}, "vocab": {}, "seed": {}, "synthetic": {}}
    dc = cfg.data

    def train_probe():
        from datasets import load_dataset
        kw = {"split": "train", "streaming": True}
        if dc.trust_remote_code:
            kw["trust_remote_code"] = True
        ds = load_dataset(dc.vqa_dataset, **kw)
        first = next(iter(ds))
        return {"dataset": dc.vqa_dataset, "reachable": True, "columns": list(first.keys())}

    def eval_probe():
        from datasets import load_dataset
        ds = load_dataset(dc.vqa_eval_dataset, split=dc.vqa_eval_split, streaming=True)
        first = next(iter(ds))
        return {"dataset": dc.vqa_eval_dataset, "split": dc.vqa_eval_split, "reachable": True,
                "columns": list(first.keys())}

    def vocab_probe():
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(cfg.model.base_model)
        return {"model": cfg.model.base_model, "num_answers": len(getattr(c, "id2label", {}) or {})}

    out["train"] = _probe(train_probe)
    out["eval"] = _probe(eval_probe)
    out["vocab"] = _probe(vocab_probe)

    from . import samples
    out["seed"] = {"ok": True, "scenes": len(samples.scenes()), "examples": len(samples.seed_examples()),
                   "answer_vocab": len(samples.answer_vocab())}

    if render_synthetic:
        try:
            from .dataset import build_synthetic_eval
            out["synthetic"] = {"ok": True, **build_synthetic_eval(cfg)}
        except Exception as exc:
            out["synthetic"] = {"ok": False, "error": str(exc)}

    logger.info("download_all: train=%s eval=%s vocab=%s seed=%d examples",
                out["train"].get("ok"), out["eval"].get("ok"), out["vocab"].get("ok"),
                out["seed"]["examples"])
    return out


__all__ = ["download_all"]
