"""Persist the prior-baseline artifact (the offline floor + a reference).

Records the global most-common answer and the per-type blind defaults that the
``PriorVQA`` / ``MostCommonVQA`` baselines use, so the registry, the report and the API
can reference the floor the trained model must beat.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..data import samples
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_baseline(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    out_path = cfg.model.baseline_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "prior", "version": "prior-1.0",
        "most_common_answer": samples.MOST_COMMON_ANSWER,
        "blind_defaults": {"yes_no": "yes", "number": "2", "color": "white"},
        "answer_vocab_size": len(samples.answer_vocab()),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("prior baseline -> %s", out_path)
    return {"baseline_path": str(out_path), "most_common_answer": samples.MOST_COMMON_ANSWER}


__all__ = ["build_baseline"]
