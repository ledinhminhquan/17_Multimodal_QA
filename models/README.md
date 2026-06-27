# models/

Trained-model artifacts land here at runtime and are **not committed** (see `.gitignore`).

- `mmqa train-vqa` writes the fine-tuned ViLT VQA core to `$MMQA_MODEL_DIR/vqa/<version>/`
  (a `latest` pointer + `model_meta.json` track the active version).
- `mmqa train-baseline` writes `prior_baseline.json` (the offline prior floor).

The default base model `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0) is downloaded from the
Hugging Face Hub on first use into `HF_HOME` — it is not stored here.
