"""Fine-tune the VQA core (ViLT) on VQAv2 - HF Trainer, soft-target BCE, VQA accuracy.

The official ViLT VQA recipe: the label space is the ~3129-answer vocabulary (from the
model config); each example's target is a SOFT vector where each annotator answer in the
vocab gets score ``min(1, count/3)`` (the same min(1, n/3) the metric uses), and
``ViltForQuestionAnswering`` optimizes BCE-with-logits against it. A custom collator runs
the ``ViltProcessor`` on each batch (images vary in size). Resume-safe; bf16/tf32 on
Ampere+/H100, fp16 on T4. Selected on VQA accuracy.

Only ViLT (classification) is implemented as the trainable default; BLIP/GIT are generative
(used zero-shot via the agent). Everything is lazy-imported.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional

from ..config import AppConfig
from ..data.dataset import load_examples
from ..logging_utils import get_logger
from ..models import model_registry as reg
from . import metrics as M

logger = get_logger(__name__)


def _score(count: int) -> float:
    return min(1.0, count / 3.0)


def train_vqa(cfg: AppConfig, limit: Optional[int] = None, resume: bool = True,
              base_model: Optional[str] = None) -> Dict:
    import torch
    from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                              ViltForQuestionAnswering, ViltProcessor)
    from transformers.trainer_utils import get_last_checkpoint

    mc = cfg.model
    if mc.model_type != "vilt":
        raise NotImplementedError("train_vqa implements ViLT (classification); BLIP/GIT are used zero-shot.")
    model_id = base_model or mc.base_model
    torch.backends.cuda.matmul.allow_tf32 = bool(mc.tf32)
    cap = limit or cfg.data.max_train_samples

    processor = ViltProcessor.from_pretrained(model_id)
    model = ViltForQuestionAnswering.from_pretrained(model_id)
    if bool(mc.gradient_checkpointing):
        model.gradient_checkpointing_enable()
    label2id = {v: k for k, v in (model.config.id2label or {}).items()}
    num_labels = len(model.config.id2label or {})
    logger.info("ViLT VQA fine-tune: %s, %d labels", model_id, num_labels)

    def to_target(answers: List[str]):
        from ..training.metrics import normalize_answer
        vec = [0.0] * num_labels
        counter = Counter(normalize_answer(a) for a in answers if a)
        for ans, cnt in counter.items():
            idx = label2id.get(ans)
            if idx is not None:
                vec[idx] = _score(cnt)
        return vec

    def build(split_examples):
        rows = []
        for ex in split_examples:
            if ex.image is None:
                continue
            tgt = to_target(ex.answers or [ex.gold])
            if sum(tgt) <= 0:        # skip examples with no in-vocab answer (standard for ViLT)
                continue
            rows.append({"image": ex.image, "question": ex.question, "labels": tgt,
                         "answers": ex.answers or [ex.gold], "answer_type": ex.answer_type})
        return rows

    train_rows = build(load_examples(cfg, split="train", limit=cap))
    eval_rows = build(load_examples(cfg, limit=cfg.data.max_eval_samples, eval_mode=True))
    if len(eval_rows) < 4:
        eval_rows = train_rows[: max(4, len(train_rows) // 10)]
    logger.info("train=%d eval=%d (in-vocab)", len(train_rows), len(eval_rows))

    class _DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    id2label = model.config.id2label or {}

    def collate(batch):
        images = [b["image"].convert("RGB") for b in batch]
        questions = [b["question"] for b in batch]
        enc = processor(images, questions, return_tensors="pt", padding=True,
                        truncation=True, max_length=mc.max_question_length)
        enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.float)
        return enc

    eval_answers = [r["answers"] for r in eval_rows]
    eval_types = [r["answer_type"] for r in eval_rows]

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        import numpy as np
        ids = np.asarray(preds).argmax(-1)
        hyp = [id2label.get(int(i), str(int(i))) for i in ids]
        return {"vqa_accuracy": M.vqa_accuracy(hyp, eval_answers),
                **{f"acc_{k}": v["accuracy"] for k, v in M.per_type_accuracy(hyp, eval_answers, eval_types).items()}}

    out_dir = mc.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=mc.num_train_epochs, learning_rate=mc.learning_rate,
        per_device_train_batch_size=mc.per_device_train_batch_size,
        per_device_eval_batch_size=mc.per_device_eval_batch_size,
        gradient_accumulation_steps=mc.gradient_accumulation_steps,
        weight_decay=mc.weight_decay, warmup_ratio=mc.warmup_ratio, max_grad_norm=mc.max_grad_norm,
        bf16=bool(mc.bf16), fp16=bool(mc.fp16),
        eval_strategy="steps", save_strategy="steps", eval_steps=mc.eval_steps, save_steps=mc.save_steps,
        save_total_limit=2, logging_steps=mc.logging_steps, seed=mc.seed, report_to=[],
        remove_unused_columns=False, load_best_model_at_end=True,
        metric_for_best_model="vqa_accuracy", greater_is_better=True)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=mc.early_stopping_patience)] \
        if mc.early_stopping_patience > 0 else []
    trainer = Trainer(model=model, args=args, train_dataset=_DS(train_rows), eval_dataset=_DS(eval_rows),
                      data_collator=collate, compute_metrics=compute_metrics, callbacks=callbacks)
    last = get_last_checkpoint(str(out_dir)) if resume and out_dir.exists() else None
    if last:
        logger.info("Resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    metrics = {}
    try:
        metrics = {k: float(v) for k, v in trainer.evaluate().items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.info("final eval failed (%s)", exc)

    version = reg.make_version(model_id)
    final_dir = out_dir / version
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    reg.write_metadata(final_dir, version=version, base_model=model_id,
                       dataset_signature={"train": len(train_rows), "dataset": cfg.data.vqa_dataset,
                                          "num_labels": num_labels, "seed": mc.seed},
                       metrics=metrics, extra={"model_type": "vilt"})
    reg.update_latest_pointer(out_dir, final_dir)
    (out_dir / "last_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("VQA training done -> %s", final_dir)
    return {"version": version, "model_dir": str(final_dir), "base_model": model_id,
            "n_train": len(train_rows), "metrics": metrics}


__all__ = ["train_vqa"]
