"""Data loading: real VQAv2 (train/eval) + the synthetic scene set (offline).

* ``load_examples`` - VQAv2 examples (image, question, the 10 annotator answers, the
  consensus answer, question/answer types). Train = ``HuggingFaceM4/VQAv2``
  (``trust_remote_code=True``); eval = ``lmms-lab/VQAv2`` validation (clean parquet).
* ``load_seed_examples`` / ``load_eval_scenes`` - the synthetic scene set for offline runs.

``datasets`` is imported lazily; everything falls back to the synthetic seed offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, data_dir
from ..logging_utils import get_logger
from . import samples

logger = get_logger(__name__)


@dataclass
class Example:
    question: str
    answers: List[str] = field(default_factory=list)     # the 10 annotator answers
    gold: str = ""                                        # consensus (multiple_choice_answer)
    answer_type: str = "other"                            # yes/no | number | other
    question_type: str = ""
    image: Any = None                                    # PIL image (real) or None
    scene: Optional[Dict] = None                          # synthetic scene spec (offline)

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "answers": self.answers, "gold": self.gold,
                "answer_type": self.answer_type, "question_type": self.question_type}


def _extract(row: Dict[str, Any]) -> Optional[Example]:
    q = str(row.get("question", "") or "").strip()
    if not q:
        return None
    raw = row.get("answers")
    answers: List[str] = []
    if isinstance(raw, list):
        for a in raw:
            if isinstance(a, dict):
                answers.append(str(a.get("answer", "")))
            else:
                answers.append(str(a))
    gold = str(row.get("multiple_choice_answer", "") or (answers[0] if answers else ""))
    if not answers and gold:
        answers = [gold]
    return Example(question=q, answers=answers, gold=gold,
                   answer_type=str(row.get("answer_type", "other") or "other"),
                   question_type=str(row.get("question_type", "") or ""),
                   image=row.get("image"))


def load_examples(cfg: AppConfig, split: str = "train", limit: Optional[int] = None,
                  eval_mode: bool = False) -> List[Example]:
    dc = cfg.data
    cap = limit or (dc.max_eval_samples if eval_mode else dc.max_train_samples)
    if dc.use_hf:
        ds_id = dc.vqa_eval_dataset if eval_mode else dc.vqa_dataset
        use_split = dc.vqa_eval_split if eval_mode else split
        try:
            from datasets import load_dataset  # lazy
            kwargs = {"split": use_split, "streaming": True}
            if not eval_mode and dc.trust_remote_code:
                kwargs["trust_remote_code"] = True
            ds = load_dataset(ds_id, **kwargs)
            out: List[Example] = []
            for r in ds:
                if len(out) >= cap:
                    break
                ex = _extract(r)
                if ex is not None:
                    out.append(ex)
            if len(out) > 2:
                logger.info("Loaded %d %s examples from %s", len(out), use_split, ds_id)
                return out
        except Exception as exc:
            logger.warning("Could not load %s (%s); using synthetic seed.", ds_id, exc)
    return load_seed_examples()


def load_seed_examples() -> List[Example]:
    out: List[Example] = []
    for e in samples.seed_examples():
        out.append(Example(question=e["question"], answers=e["answers"], gold=e["gold"],
                           answer_type=e.get("answer_type", "other"), question_type=e.get("qtype", ""),
                           scene=e["scene"]))
    return out


def load_eval_examples(cfg: AppConfig, limit: Optional[int] = None) -> List[Example]:
    if cfg.data.use_hf:
        try:
            out = load_examples(cfg, limit=limit, eval_mode=True)
            if out and out[0].image is not None:
                return out
        except Exception:
            pass
    return load_seed_examples()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic scenes (offline eval)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_dir(split: str = "eval") -> Path:
    return data_dir() / "synthetic" / split


def load_manifest(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def load_eval_scenes(cfg: AppConfig, split: str = "eval") -> List[Dict]:
    """Synthetic scene QA rows (scene + question + gold) for offline eval; prefer a rendered
    manifest, else the built-in seed scenes."""
    rows = load_manifest(synthetic_dir(split) / "manifest.jsonl")
    if rows:
        logger.info("Loaded %d synthetic eval rows", len(rows))
        return rows
    return samples.seed_examples()


def build_synthetic_eval(cfg: AppConfig, n_scenes: Optional[int] = None, split: str = "eval") -> Dict:
    from .synth_scene import generate_dataset
    return generate_dataset(str(synthetic_dir(split)), n_scenes=n_scenes or cfg.data.synth_eval_scenes,
                            max_shapes=cfg.data.max_shapes, image_size=cfg.data.image_size, seed=cfg.data.seed)


__all__ = ["Example", "load_examples", "load_seed_examples", "load_eval_examples",
           "synthetic_dir", "load_manifest", "load_eval_scenes", "build_synthetic_eval"]
