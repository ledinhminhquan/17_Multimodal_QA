"""Evaluation: VQA accuracy (model vs baselines) + per-type + the agent's abstention story.

Three measurements:
* **Model VQA accuracy** (the trainable core) vs the **blind question-only prior** (language
  bias) and the **most-common-answer** floor - the official soft accuracy + per-answer-type.
* **The agent**: coverage (fraction answered, not abstained), accuracy-on-answered, and the
  overall accuracy (abstentions count as wrong) - showing the precision/coverage trade-off
  the calibrated-abstention gate buys.

Runs offline on the synthetic seed scenes (no torch/model) via the SceneStubVQA; on Colab the
same code scores the fine-tuned ViLT on real VQAv2. Results -> ``run_dir/eval.json``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..data.dataset import Example, load_eval_examples
from ..logging_utils import get_logger
from . import metrics as M

logger = get_logger(__name__)


def _image_for(ex: Example):
    if ex.image is not None:
        return ex.image
    if ex.scene is not None:
        from ..vision.image_utils import SceneImage
        return SceneImage(ex.scene)
    return None


def _system_scores(system, examples: List[Example]) -> Dict[str, Any]:
    preds, answers, types = [], [], []
    for ex in examples:
        img = _image_for(ex)
        try:
            res = system.answer(img, ex.question, top_k=5)
            preds.append(res.top1)
        except Exception:
            preds.append("")
        answers.append(ex.answers or [ex.gold])
        types.append(ex.answer_type or "other")
    return {"accuracy": M.vqa_accuracy(preds, answers),
            "per_type": M.per_type_accuracy(preds, answers, types)}


def evaluate(cfg: AppConfig, *, limit: Optional[int] = None, load_model: bool = True,
             save: bool = True) -> Dict[str, Any]:
    from ..agent.vqa_agent import VqaAgent
    from ..models.baseline import MostCommonVQA, PriorVQA
    from ..models.vqa_model import load_vqa_model

    examples = load_eval_examples(cfg, limit=limit)
    if limit:
        examples = examples[:limit]
    model = load_vqa_model(cfg.model, prefer="model" if load_model else "stub")

    systems = {
        "most_common": _system_scores(MostCommonVQA(), examples),
        "blind_prior": _system_scores(PriorVQA(), examples),
        getattr(model, "name", "model"): _system_scores(model, examples),
    }

    # agent eval: coverage + accuracy-on-answered + overall (abstain = wrong)
    agent = VqaAgent(cfg, load_model=False, model=model)
    ans_preds, ans_answers, overall_preds = [], [], []
    n_abstain = 0
    for ex in examples:
        img = _image_for(ex)
        job = agent.run(image=ex.image, scene=ex.scene if ex.image is None else None,
                        question=ex.question, save=False) if img is not None else None
        gold = ex.answers or [ex.gold]
        if job is None or job.abstained:
            n_abstain += 1
            overall_preds.append("__abstain__")
        else:
            ans_preds.append(job.answer)
            ans_answers.append(gold)
            overall_preds.append(job.answer)
    n = len(examples)
    overall_answers = [ex.answers or [ex.gold] for ex in examples]
    agent_eval = {
        "n": n, "coverage": round((n - n_abstain) / max(1, n), 4), "abstain_rate": round(n_abstain / max(1, n), 4),
        "accuracy_on_answered": M.vqa_accuracy(ans_preds, ans_answers) if ans_preds else 0.0,
        "overall_accuracy": M.vqa_accuracy(overall_preds, overall_answers),
    }

    model_name = getattr(model, "name", "model")
    report = {"n": n, "systems": systems, "agent": agent_eval,
              "model_version": getattr(model, "version", "?"),
              "headline": {"model_vqa_accuracy": systems[model_name]["accuracy"],
                           "blind_prior_accuracy": systems["blind_prior"]["accuracy"],
                           "agent_accuracy_on_answered": agent_eval["accuracy_on_answered"],
                           "agent_coverage": agent_eval["coverage"]}}
    if save:
        out = run_dir() / "eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("eval -> %s (model acc=%s, blind=%s, agent cover=%s)", out,
                    report["headline"]["model_vqa_accuracy"], report["headline"]["blind_prior_accuracy"],
                    agent_eval["coverage"])
    return report


__all__ = ["evaluate"]
