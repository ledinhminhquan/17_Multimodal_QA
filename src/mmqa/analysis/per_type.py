"""Per-question-type accuracy + abstention report (the P17 special quality analysis).

Runs the agent on the seed scenes and breaks accuracy + abstention down by question type
(yes_no, number, color, ...). This surfaces WHERE the system is strong/weak and where it
(correctly) declines to answer - more informative than a single accuracy number.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..training.metrics import vqa_accuracy_single

logger = get_logger(__name__)


def per_type_report(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
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

    buckets: Dict[str, Dict[str, float]] = {}
    for ex in examples:
        job = agent.run(scene=ex["scene"], question=ex["question"], save=False)
        t = job.qtype or "other"
        b = buckets.setdefault(t, {"n": 0, "abstained": 0, "acc_sum": 0.0, "answered": 0})
        b["n"] += 1
        if job.abstained:
            b["abstained"] += 1
        else:
            b["answered"] += 1
            b["acc_sum"] += vqa_accuracy_single(job.answer, ex["answers"])
    report = {t: {"n": int(b["n"]), "abstain_rate": round(b["abstained"] / max(1, b["n"]), 4),
                  "accuracy_on_answered": round(b["acc_sum"] / max(1, b["answered"]), 4)}
              for t, b in sorted(buckets.items())}
    out = {"per_type": report, "n": len(examples)}
    if save:
        _save(out)
    logger.info("per-type: %s", {t: v["accuracy_on_answered"] for t, v in report.items()})
    return out


def _stub(error: str, save: bool) -> Dict:
    out = {"per_type": {}, "n": 0, "error": error}
    if save:
        _save(out)
    return out


def _save(out: Dict) -> None:
    try:
        d = run_dir() / "per_type"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"per_type-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.info("per_type: could not save (%s)", exc)


__all__ = ["per_type_report"]
