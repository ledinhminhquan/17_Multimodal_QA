"""Typed configuration + YAML loader for the mmqa Visual-Question-Answering system.

Single source of truth for the trainable VQA core, the datasets (VQAv2 + a synthetic
scene generator for offline), the agent decision thresholds (D1-D5), and serving. Paths
come from environment variables so nothing is hard-coded (required by the assignment).

Pipeline: image + question -> (preprocess + tokenize) -> VQA model -> answer post-process.
The VQA model is the only trained component; image preprocessing, the question-type router
and the answer constraints are algorithmic.

Environment overrides
---------------------
* ``MMQA_ARTIFACTS_DIR`` - base for data/models/runs (Drive on Colab)
* ``MMQA_DATA_DIR``      - dataset cache / generated synthetic scenes
* ``MMQA_MODEL_DIR``     - trained models (the fine-tuned VQA core)
* ``MMQA_RUN_DIR``       - eval/benchmark/analysis JSON
* ``HF_HOME``            - HuggingFace cache
* ``MMQA_LLM_API_KEY``   - optional key for the LLM agent brain

Verified ids (confirmed on the HF Hub during research - keep exact):
  vqa model    dandelin/vilt-b32-finetuned-vqa (Apache-2.0, classification over ~3129 answers, default) ·
               dandelin/vilt-b32-mlm (pretrain checkpoint to fine-tune) · Salesforce/blip-vqa-base (BSD, generative) ·
               Salesforce/blip2-opt-2.7b (H100) — see docs/DESIGN_BRIEF.md
  data         HuggingFaceM4/VQAv2 (COCO images CC-BY) ; small/demo merve/vqav2-small
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("MMQA_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("MMQA_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("MMQA_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def run_dir() -> Path:
    return Path(_env("MMQA_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """VQAv2 (real) + the synthetic scene generator (offline / tests).

    VQAv2 is the canonical benchmark (COCO images, CC-BY); each question has the answers
    of 10 human annotators (the soft-accuracy target). For offline tests there is a
    reproducible SYNTHETIC scene generator (``data/synth_scene.py``) that draws colored
    shapes and embeds a scene spec in the PNG, with templated (question, gold answer)
    pairs - so the agent, eval and tests run with no torch and no real VQA model.
    """
    # TRAIN: the only common VQAv2 mirror with a train split + 10-annotator schema
    # (license FLAG: undeclared on hub, COCO/VQA upstream is CC-BY-4.0; needs trust_remote_code=True).
    vqa_dataset: str = "HuggingFaceM4/VQAv2"
    vqa_config: str = ""
    trust_remote_code: bool = True
    # EVAL: clean parquet mirror (cc-by-4.0) with the 10-annotator answers + answer_type.
    vqa_eval_dataset: str = "lmms-lab/VQAv2"
    vqa_eval_split: str = "validation"
    small_dataset: str = "merve/vqav2-small"  # fast iteration / demo (no annotators -> demo only)
    use_hf: bool = True
    max_train_samples: int = 40000
    max_eval_samples: int = 4000
    # synthetic scene generator (offline backbone + a controllable eval slice)
    synth_train_scenes: int = 400
    synth_eval_scenes: int = 120
    image_size: int = 384
    max_shapes: int = 4
    seed: int = 42


@dataclass
class ModelConfig:
    """The TRAINABLE VQA core (image+question -> answer)."""
    base_model: str = "dandelin/vilt-b32-finetuned-vqa"   # Apache, classification head (~3129 answers)
    pretrain_model: str = "dandelin/vilt-b32-mlm"          # to fine-tune the classifier from scratch
    model_type: str = "vilt"                # "vilt" (classification) | "blip" (generative) | "git"
    image_size: int = 384
    max_question_length: int = 40
    num_answers: int = 3129                 # ViLT VQA label space
    top_k: int = 5                          # answers returned for the agent to inspect
    # training (HF Trainer)
    num_train_epochs: int = 4
    learning_rate: float = 5.0e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = False
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    seed: int = 42
    output_subdir: str = "vqa"
    baseline_filename: str = "prior_baseline.json"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir

    @property
    def baseline_path(self) -> Path:
        return self.output_dir / self.baseline_filename


@dataclass
class AgentConfig:
    """VQA agent decision thresholds (D1-D5) + optional LLM brain."""
    # D1 - input gate
    min_question_chars: int = 3
    # D2 - question-type classification handled by models/question_type.py
    # D4 - calibrated abstention gate (VQA models are overconfident)
    abstain_enabled: bool = True
    confidence_min: float = 0.20            # max softmax prob below this -> abstain
    entropy_max: float = 2.5                # top-k entropy above this -> low confidence
    margin_min: float = 0.03               # (top1 - top2) below this -> uncertain
    # D5 - type-consistency gate (constrain the answer to the question type)
    type_consistency_enabled: bool = True
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "MMQA_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Multimodal Question Answering (VQA) API"
    api_version: str = "1.0.0"
    log_jobs: bool = True
    job_log_subdir: str = "job_logs"
    max_file_mb: int = 15

    @property
    def job_log_path(self) -> Path:
        return run_dir() / self.job_log_subdir / "jobs.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Multimodal Question Answering (VQA) System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "ModelConfig", "AgentConfig", "ServingConfig", "AppConfig",
           "load_config", "save_config", "ensure_dirs",
           "artifacts_dir", "data_dir", "model_dir", "run_dir"]
