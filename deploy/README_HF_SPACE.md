# Deploying mmqa as a Hugging Face Space

Two options: a **Gradio** Space (simplest) or a **Docker** Space (full API + UI).

## Option A — Gradio Space (recommended for a demo)

1. Create a new Space → SDK: **Gradio**.
2. Add these files to the Space repo:
   - `app.py` → copy `app/gradio_app.py` (or `from mmqa.api.ui import build_demo; build_demo().launch()`).
   - the `src/mmqa/` package (or `pip install` it from your GitHub repo in `requirements.txt`).
   - `requirements.txt` → at least: `transformers datasets accelerate sentencepiece Pillow gradio reportlab python-pptx matplotlib`.
   - a `packages.txt` with the image system dep:
     ```
     libgl1
     ```
3. The first request lazily downloads `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0) into the
   Space cache. With no torch at all the Space falls back to the SceneStubVQA so the demo on the
   synthetic sample scene never hard-fails.

## Option B — Docker Space (full FastAPI + Gradio)

1. Create a new Space → SDK: **Docker**.
2. Add the repo's `Dockerfile` (it installs `libgl1` + the Python deps).
3. The container serves `mmqa.api.app_combined:app` on port 8000:
   - REST: `POST /ask` (upload an image + a question → answer + confidence + abstain flag),
     `POST /ask-scene` (JSON synthetic path), `GET /healthz`, `GET /version`.
   - UI: the Gradio demo is mounted at **`/ui`**.

## Notes
- Set `MMQA_INFER_CONFIG=/app/configs/infer.yaml` to change the model / thresholds.
- The optional LLM brain is OFF by default; set `MMQA_LLM_API_KEY` + `agent.llm_fallback_enabled: true`.
- User photos can be sensitive (faces, homes, documents) — the Space processes uploads transiently
  and logs metadata only; low-confidence answers are returned as "unsure".
