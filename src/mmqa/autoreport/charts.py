"""Matplotlib charts for the VQA report/slides.

  * an **accuracy** bar chart - VQA accuracy for most-common vs blind-prior vs model;
  * an **agent** chart - coverage, accuracy-on-answered, overall accuracy (the abstention trade-off);
  * an **outcome bucket** chart (correct / abstained / wrong) from error analysis.

Returns saved PNG paths under ``run_dir()/report``; matplotlib lazy-imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger
from . import artifact_loader as AL

logger = get_logger(__name__)

_COMMON = "#cbd5e0"
_BLIND = "#9aa7b4"
_MODEL = "#2b6cb0"
_GOOD = "#2f855a"
_MED = "#dd6b20"
_POOR = "#c53030"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def accuracy_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not AL.has_eval(arts):
        return None
    model_name = AL.model_system_name(arts) or "model"
    series = [("most_common", AL.sys_accuracy(arts, "most_common"), _COMMON),
              ("blind_prior", AL.sys_accuracy(arts, "blind_prior"), _BLIND),
              (model_name, AL.sys_accuracy(arts, model_name), _MODEL)]
    series = [(n, (v or 0.0) * 100, c) for n, v, c in series]
    try:
        plt = _mpl()
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.bar([n for n, _, _ in series], [v for _, v, _ in series], color=[c for _, _, c in series])
        for i, (_, v, _) in enumerate(series):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("VQA accuracy (%)")
        ax.set_title("VQA accuracy: model vs blind prior vs most-common floor")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("accuracy_chart skipped (%s)", exc)
        return None


def agent_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    cov = AL.agent_metric(arts, "coverage")
    acc = AL.agent_metric(arts, "accuracy_on_answered")
    overall = AL.agent_metric(arts, "overall_accuracy")
    if cov is None and acc is None:
        return None
    try:
        plt = _mpl()
        labels = ["coverage", "acc (answered)", "overall acc"]
        vals = [(cov or 0.0) * 100, (acc or 0.0) * 100, (overall or 0.0) * 100]
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        ax.bar(labels, vals, color=[_MED, _GOOD, _MODEL])
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("%")
        ax.set_title("Agent: the abstention trade-off (coverage vs accuracy)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("agent_chart skipped (%s)", exc)
        return None


def buckets_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    b = AL.buckets(arts)
    vals = [b.get("correct"), b.get("abstained"), b.get("wrong")]
    if not any(isinstance(v, (int, float)) for v in vals):
        return None
    try:
        plt = _mpl()
        labels = ["correct", "abstained", "wrong"]
        nums = [float(v) if isinstance(v, (int, float)) else 0.0 for v in vals]
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.bar(labels, nums, color=[_GOOD, _MED, _POOR])
        for i, v in enumerate(nums):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("# questions"); ax.set_title("Agent outcomes (correct / abstained / wrong)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("buckets_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []
    charts: List[Tuple[str, Path]] = []
    jobs = [("accuracy", lambda p: accuracy_chart(arts, p)),
            ("agent", lambda p: agent_chart(arts, p)),
            ("buckets", lambda p: buckets_chart(arts, p))]
    for name, fn in jobs:
        try:
            p = fn(out_dir / f"{name}.png")
        except Exception as exc:
            logger.info("chart %s skipped (%s)", name, exc)
            p = None
        if p:
            charts.append((name, p))
    return charts


__all__ = ["accuracy_chart", "agent_chart", "buckets_chart", "build_all"]
