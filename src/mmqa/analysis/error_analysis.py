"""Error analysis (offline): correct / abstained / wrong buckets + worst examples.

Runs the agent on the seed scenes, compares each answer to the gold, and buckets the
outcomes (correct, abstained, wrong). The short keys ``correct``/``abstained``/``wrong``
feed the charts.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..training.metrics import vqa_accuracy_single

logger = get_logger(__name__)


def error_analysis(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..agent.vqa_agent import VqaAgent
        from ..data import samples
        agent = VqaAgent(cfg, load_model=False)
        examples = samples.seed_examples()
    except Exception as exc:
        return _stub(str(exc), save)
    if limit:
        examples = examples[:limit]

    correct = abstained = wrong = 0
    worst: List[Dict] = []
    for ex in examples:
        try:
            job = agent.run(scene=ex["scene"], question=ex["question"], save=False)
        except Exception:
            continue
        if job.abstained:
            abstained += 1
            continue
        acc = vqa_accuracy_single(job.answer, ex["answers"])
        if acc >= 0.6:
            correct += 1
        else:
            wrong += 1
            if len(worst) < 8:
                worst.append({"question": ex["question"], "answer": job.answer,
                              "gold": ex["gold"], "qtype": job.qtype})
    n = max(1, len(examples))
    result = {"n": len(examples), "correct": correct, "abstained": abstained, "wrong": wrong,
              "accuracy": round(correct / n, 4), "abstain_rate": round(abstained / n, 4),
              "worst_examples": worst}
    if save:
        _save(result)
    logger.info("error analysis: correct=%d abstained=%d wrong=%d", correct, abstained, wrong)
    return result


def _stub(error: str, save: bool) -> Dict:
    result = {"n": 0, "correct": 0, "abstained": 0, "wrong": 0, "accuracy": 0.0,
              "abstain_rate": 0.0, "worst_examples": [], "error": error}
    if save:
        _save(result)
    return result


def _save(result: Dict) -> None:
    try:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("error_analysis: could not save (%s)", exc)


__all__ = ["error_analysis"]
