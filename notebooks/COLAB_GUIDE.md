# Colab Guide — training mmqa (VQA) on an H100 (auto-adapts A100/L4/T4)

Run `MMQA_Colab_Training_H100_AUTOPILOT.ipynb`: upload the repo, set a few controls, **Run all**.

## 0. What gets trained
- **Only the ViLT VQA core** is fine-tuned — HF Trainer with **soft-target BCE** over the
  ~3129-answer vocabulary (each annotator answer gets score min(1, count/3)), selected on **VQA
  accuracy**, resume-safe.
- **The question-type router, the calibrated-abstention gate and the answer constraints** are
  algorithmic, not trained. The notebook reports VQA accuracy + per-answer-type + the agent's
  abstention/coverage.

## 1. Put the repo where Colab can see it (pick ONE)
- **GitHub (recommended):** push this folder to `https://github.com/<you>/mmqa`, set `GIT_REPO_URL`.
- **Drive:** upload `17_Multimodal_QA/` to `MyDrive/mmqa/mmqa` (repo root = `.../mmqa/mmqa`); leave
  `GIT_REPO_URL` as the placeholder.

```
MyDrive/mmqa/
├── mmqa/           <- the repo, if using Drive
└── artifacts/      <- created automatically; the VQA model + reports persist here
```

## 2. Runtime
`Runtime -> Change runtime type -> GPU`. ViLT (~113M) fine-tunes even on a **free T4**; cell 6
auto-profiles batch/precision for H100/A100/L4/T4 (T4 has no bf16 -> fp16).

## 3. Controls (cell 0)
- `VQA_BASE` — `dandelin/vilt-b32-finetuned-vqa` (Apache, default; continues from the VQA head) or
  `dandelin/vilt-b32-mlm` (train a fresh head).
- `VQA_DATASET` — the train mirror (`HuggingFaceM4/VQAv2`, needs `trust_remote_code=True`).
- `EVAL_DATASET` — `lmms-lab/VQAv2` (clean, CC-BY-4.0).
- `MAX_TRAIN_SAMPLES`, `EPOCHS` — training budget.

## 4. Run all
The **autopilot** (cell 9) does everything: baseline -> fine-tune ViLT -> evaluate (VQA accuracy +
per-type + agent coverage) -> analysis -> **report.pdf + slides.pptx + grading + submission_bundle.zip**.
Resume-safe: re-run cell 9 after a disconnect.

## 5. Read the results (cell 11)
Look for the fine-tuned ViLT's **VQA accuracy** beating the **blind question-only prior** (the
language-bias floor), a reasonable per-answer-type breakdown, and a sane abstention rate.

## 6. Test the trained model (cell 12)
Cell 12 asks "how many shapes are there?" about the sample image and prints the answer + confidence.
Try your own image by uploading it and changing `--image` / `--question`.

## 7. Deliverables (cell 13)
`report.pdf`, `slides.pptx`, `submission_bundle.zip` under
`artifacts/submission/submission-<stamp>/` (on Drive).

## Troubleshooting
- **"Set GIT_REPO_URL..."** — neither a repo URL nor a Drive copy was found; do step 1.
- **VQAv2 train load error** — the `HuggingFaceM4/VQAv2` mirror is a loading-script dataset; the
  notebook passes `trust_remote_code=True`. If it still fails on a new `datasets` version, train on
  `Multimodal-Fatima/VQAv2_sample_train` (small, full schema) instead.
- **bf16 error on T4** — Turing has no bf16; cell 6 falls back to fp16.
- **OOM** — lower the batch in cell 6 / `MAX_TRAIN_SAMPLES`.
- **License** — the shipped stack is ViLT (Apache) + VQAv2 (COCO/VQA CC-BY-4.0 upstream); LLaVA and
  Qwen2.5-VL-3B are non-commercial (flagged, not shipped).
