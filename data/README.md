# data/

Dataset caches and generated synthetic scenes live here at runtime — none of it is committed
(see `.gitignore`).

- VQAv2 is streamed by `mmqa data` / `mmqa train-vqa` (train `HuggingFaceM4/VQAv2` with
  `trust_remote_code=True`; eval `lmms-lab/VQAv2` validation) and cached under `HF_HOME`.
- Synthetic scene images are rendered by `mmqa gen-synthetic` into
  `$MMQA_DATA_DIR/synthetic/<split>/` (PNGs + `manifest.jsonl`), each PNG embedding its scene spec.
- The **offline backbone** (seed scenes + the answer vocabulary) is code, not data —
  [`src/mmqa/data/samples.py`](../src/mmqa/data/samples.py) — so the pipeline runs with no network.

Nothing here is required to import the package or run the tests.
