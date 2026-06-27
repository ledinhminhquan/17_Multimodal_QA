# Multimodal Question Answering (VQA) — serving image.
# libGL/libglib for Pillow; the ViLT model downloads from the HF Hub on first use.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    MMQA_ARTIFACTS_DIR=/app/artifacts

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

EXPOSE 8000
# Serve the FastAPI app with the Gradio UI mounted at /ui.
CMD ["uvicorn", "mmqa.api.app_combined:app", "--host", "0.0.0.0", "--port", "8000"]
