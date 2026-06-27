"""Lightweight abstention-threshold sweep (the precision/coverage trade-off).

Sweeps the agent's confidence threshold and reports coverage + accuracy-on-answered on the
seed scenes - a cheap, offline proxy for tuning the calibrated-abstention gate. Higher
thresholds answer fewer questions but more accurately.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..data.dataset import load_eval_examples
from ..logging_utils import get_logger
from . import metrics as M

logger = get_logger(__name__)


def tune(cfg: AppConfig, thresholds: Optional[List[float]] = None, save: bool = True,
         load_model: bool = True) -> Dict:
    thresholds = thresholds or [0.1, 0.2, 0.35, 0.5]
    from ..agent.vqa_agent import VqaAgent
    from ..models.vqa_model import load_vqa_model
    model = load_vqa_model(cfg.model, prefer="model" if load_model else "stub")
    examples = load_eval_examples(cfg)

    trials: List[Dict[str, Any]] = []
    for thr in thresholds:
        acfg = replace(cfg.agent, confidence_min=thr)
        tcfg = replace(cfg, agent=acfg)
        agent = VqaAgent(tcfg, load_model=False, model=model)
        ans_preds, ans_answers = [], []
        n_abstain = 0
        for ex in examples:
            if ex.image is None and ex.scene is None:
                continue
            job = agent.run(image=ex.image, scene=ex.scene if ex.image is None else None,
                            question=ex.question, save=False)
            if job.abstained:
                n_abstain += 1
            else:
                ans_preds.append(job.answer)
                ans_answers.append(ex.answers or [ex.gold])
        n = len(examples)
        trials.append({"confidence_min": thr, "coverage": round((n - n_abstain) / max(1, n), 4),
                       "accuracy_on_answered": M.vqa_accuracy(ans_preds, ans_answers) if ans_preds else 0.0})

    best = max(trials, key=lambda t: t["accuracy_on_answered"] * t["coverage"]) if trials else {}
    result = {"trials": trials, "best": best}
    if save:
        out = run_dir() / "tune"
        out.mkdir(parents=True, exist_ok=True)
        (out / "tune.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("tune: best confidence_min=%s (cover=%s acc=%s)", best.get("confidence_min"),
                    best.get("coverage"), best.get("accuracy_on_answered"))
    return result


__all__ = ["tune"]
