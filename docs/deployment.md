# P17 Multimodal Question Answering (VQA) — Deployment

> Author: Le Dinh Minh Quan (23127460). Package `mmqa`, folder `17_Multimodal_QA`.
> This document covers how the VQA system is served: the FastAPI surface, the Gradio UI,
> Docker packaging, a Hugging Face Space, GPU-vs-CPU serving tiers, latency, environment
> configuration, and a concrete request/response example.

This is the **first multimodal-vision project** in the series (P02–P16 were text / document /
OCR). The deployment story therefore has to handle a new ingredient that none of the earlier
projects did: a **binary image payload** alongside the text question. Everything else — the
config object, structured logging, registry, autoreport, CLI/API wrappers — is reused from the
sibling templates (P15 `imgtrans`, P14 `doctrans`) and is not re-invented here.

The thing being served is **not the raw model** — it is the deterministic agent
(`src/mmqa/agent/`) wrapping the VQA model. Every HTTP and UI response is the agent's output
object: `{answer, status, question_type, confidence, topk}`, where `status ∈ {ok, ok:reranked,
abstained:low_confidence, abstained:type_mismatch, error:bad_image, error:bad_question,
error:model_failure}`. The deployment surface exists to expose the agent's two value-adds —
**calibrated abstention** (the system says `"unsure"` instead of hallucinating) and the
**type-aware answer constraint** — over the network, never just a raw argmax.

---

## 1. Serving topology

```
                         ┌───────────────────────────────────────────────┐
   image + question ───► │  FastAPI  (src/mmqa/api/)                      │
   (multipart or JSON)   │   /healthz   GET   liveness/readiness         │
                         │   /ask       POST  multipart image + question │ ──► agent.predict() ──► {answer,
                         │   /ask-scene POST  JSON synthetic scene path  │                          status,
                         │   /info      GET   model id, license, device  │                          question_type,
                         └───────────────────────────────────────────────┘                          confidence,
                         ┌───────────────────────────────────────────────┐                          topk}
   browser upload ─────► │  Gradio UI  (src/mmqa/ui/)                     │ ──► agent.predict() ──►
                         │   image upload + question textbox             │
                         └───────────────────────────────────────────────┘
```

Both front ends call the **same** `agent.predict(image, question)` entry point. There is exactly
one inference code path; the API and the UI are thin adapters. This is what guarantees the
demo and the production endpoint cannot diverge.

The agent runs against either backend with **no code change**:

- **Real backend** — `ViltForQuestionAnswering` + `ViltProcessor` over
  `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0, ~113M, classification head over the canonical
  3129-answer vocab). The agent needs the full softmax distribution (top-k, `p_max`, entropy,
  margin), which is why the explicit processor+model path is used rather than the convenience
  `transformers.pipeline`.
- **Stub backend** — `SceneStubVQA`, which reads the scene spec embedded in the PNG metadata and
  returns a realistic (not one-hot) distribution. **No torch, no model download, no network.**
  This is what powers `/ask-scene`, CI, and any CPU-only / air-gapped deployment.

Selecting between them is a single env var (`MMQA_BACKEND`, see §7). Swapping `SceneStubVQA`
for the real ViLT wrapper "flips the system to production with no agent changes."

---

## 2. FastAPI endpoints

The API lives in `src/mmqa/api/` and wraps the agent. Launch:

```bash
uvicorn mmqa.api.app:app --host 0.0.0.0 --port 8000
# or via the project CLI
python -m mmqa.cli serve --host 0.0.0.0 --port 8000
```

### 2.1 `GET /healthz` — liveness / readiness

Cheap, dependency-free probe for container orchestration (Docker `HEALTHCHECK`, k8s
liveness/readiness, HF Space warmup). It does **not** run inference. It reports whether the
configured backend is loaded and which device it is on.

```json
{
  "status": "ok",
  "backend": "vilt",
  "model_id": "dandelin/vilt-b32-finetuned-vqa",
  "device": "cuda:0",
  "model_loaded": true,
  "version": "0.1.0"
}
```

When `MMQA_BACKEND=stub`, `backend` is `"stub"`, `model_id` is `null`, `device` is `"cpu"`, and
`model_loaded` is `true` immediately (the stub has nothing to download). `/healthz` returns
`200` once the process can serve; if the real model is still loading it returns `503` with
`"status": "loading"` so the orchestrator does not route traffic prematurely.

### 2.2 `POST /ask` — image upload + question (the real VQA path)

The primary endpoint. Accepts a **`multipart/form-data`** body: a binary image file plus the
question string. Returns the agent's answer, its confidence, and the abstain flag.

This route is **gated on `python-multipart`**. FastAPI/Starlette cannot parse `multipart/form-data`
(which is how file uploads arrive) unless `python-multipart` is installed. If it is missing, the
app does **not** crash on import — instead `/ask` is registered to return a clear `503`:

```json
{
  "detail": "image-upload endpoint unavailable: install 'python-multipart' to enable multipart/form-data file uploads"
}
```

This keeps `/healthz`, `/info`, and `/ask-scene` (all JSON, no upload) fully functional even in a
minimal install, and surfaces a precise, actionable error instead of an opaque 500. `python-multipart`
is in the default deployment requirements; the gate is a defensive guard for trimmed environments.

**Form fields**

| field      | type           | required | notes                                                          |
|------------|----------------|----------|----------------------------------------------------------------|
| `image`    | file (binary)  | yes      | PNG/JPEG/WebP; decoded to RGB by the model image processor      |
| `question` | string (form)  | yes      | natural-language question; validated at the agent's D1 gate     |
| `top_k`    | int (form)     | no       | candidates to return (default 5)                               |

**Processing.** The uploaded bytes are decoded with Pillow to an RGB image and handed, together
with the question, to `agent.predict()`. The agent runs its 5-point FSM:

1. **D1 input gate** — rejects a corrupt/blank image (`error:bad_image`) or an empty/non-question
   string (`error:bad_question`) **before** any model call. These return HTTP `422` (the input
   was structurally invalid), with the error status in the body.
2. **D2 question-type router** → `yes_no / count / color / object-other`.
3. **D3 run VQA** → top-k + `p_max` + entropy + margin (a model exception surfaces as
   `error:model_failure`, HTTP `500`).
4. **D4 calibrated abstention** → if under-confident, `answer="unsure"`,
   `status="abstained:low_confidence"`, `needs_review=true` (HTTP `200` — abstaining is a valid,
   successful outcome, not an error).
5. **D5 type-consistency + re-rank** → constrains/re-ranks within top-k; `ok` or `ok:reranked`,
   or `abstained:type_mismatch`.

**Response body** (HTTP `200`):

```json
{
  "answer": "2",
  "status": "ok:reranked",
  "question_type": "count",
  "answer_type": "number",
  "confidence": 0.71,
  "abstain": false,
  "needs_review": false,
  "topk": [
    {"answer": "cat", "score": 0.40},
    {"answer": "2",   "score": 0.31},
    {"answer": "3",   "score": 0.12},
    {"answer": "1",   "score": 0.09},
    {"answer": "dog", "score": 0.08}
  ],
  "model_id": "dandelin/vilt-b32-finetuned-vqa",
  "latency_ms": 142
}
```

`abstain` is the convenience boolean (`true` when `status` starts with `abstained:`); `status`
carries the precise reason and is the field the agent actually emits. `confidence` is `p_max`
after temperature scaling. The example above shows the re-rank value-add: raw argmax was `cat`
(type-wrong for "how many"), and D5 promoted the highest-prob type-consistent candidate `2`.

**HTTP status mapping**

| agent `status`                | HTTP | meaning                                              |
|-------------------------------|------|------------------------------------------------------|
| `ok`, `ok:reranked`           | 200  | answered                                             |
| `abstained:low_confidence`    | 200  | answered with `"unsure"`, `needs_review=true`        |
| `abstained:type_mismatch`     | 200  | no type-consistent candidate; `"unsure"`             |
| `error:bad_image`             | 422  | image failed D1 validation                           |
| `error:bad_question`          | 422  | question failed D1 validation                        |
| `error:model_failure`         | 500  | model raised at inference                            |
| (`python-multipart` missing)  | 503  | upload support not installed                         |

### 2.3 `POST /ask-scene` — JSON synthetic-scene path

The torch-free, network-free path for the synthetic scene generator (`data/synth_scene.py`). It
takes **JSON only** — no file upload, so it needs no `python-multipart` — and is the endpoint used
for CI smoke tests, offline demos, and CPU-only deployments where downloading ViLT is undesirable.

Two ways to specify the scene:

- a `seed` (the server calls `make_scene(seed)` to deterministically build the scene + spec), or
- an explicit `scene_spec` (the same dict that would be embedded in PNG metadata: objects with
  `shape`/`color`/`bbox`, `counts_by_color_shape`, `colors_present`, `shapes_present`).

**Request**

```json
{
  "seed": 7,
  "question": "how many red squares?",
  "top_k": 5
}
```

**Response** — identical schema to `/ask` so clients are interchangeable:

```json
{
  "answer": "2",
  "status": "ok",
  "question_type": "count",
  "answer_type": "number",
  "confidence": 0.83,
  "abstain": false,
  "needs_review": false,
  "topk": [
    {"answer": "2", "score": 0.83},
    {"answer": "1", "score": 0.07},
    {"answer": "3", "score": 0.06},
    {"answer": "0", "score": 0.04}
  ],
  "model_id": null,
  "latency_ms": 3
}
```

Because `SceneStubVQA` reads the gold answer from the scene spec and spreads the remaining mass
over plausible type-consistent distractors, the full agent — including D4 abstention and D5
re-rank — runs and is exercised exactly as in production, with single-digit-millisecond latency
and zero external dependencies.

### 2.4 `GET /info` — model & license metadata

Returns the registry entry for the active backend: HF id, **license (with any flag)**, parameter
count, processor/model classes, the answer-vocab size (3129 for ViLT), and the configured
thresholds (`tau_conf`, `tau_ent`, `tau_margin`). This makes the licensing posture inspectable at
runtime — important because several upstream datasets/models in this project carry **FLAG**ged or
non-commercial licenses (see §8) and the served default must be the clean Apache-2.0 ViLT.

```json
{
  "backend": "vilt",
  "model_id": "dandelin/vilt-b32-finetuned-vqa",
  "license": "apache-2.0",
  "license_flag": null,
  "params": "~113M",
  "answer_vocab_size": 3129,
  "thresholds": {"tau_conf": 0.30, "tau_ent": 1.5, "tau_margin": 0.10}
}
```

### 2.5 OpenAPI / docs

FastAPI auto-serves interactive docs at `/docs` (Swagger UI) and `/redoc`. The multipart form for
`/ask` renders a file-picker there, making it usable for manual testing without writing a client.

---

## 3. Gradio UI

A single-screen Gradio app (`src/mmqa/ui/app.py`) for human use and the HF Space:

- an **image upload** component (drag-drop or file picker; also webcam-capture optional),
- a **question textbox**,
- a **Submit** button,
- outputs: the **answer** (large), a **confidence** bar, an **abstain / needs-review** badge that
  lights up when `status` starts with `abstained:`, the detected **question_type**, and a
  collapsible **top-k table** so a user can see the candidate distribution behind the answer.

Launch:

```bash
python -m mmqa.cli demo            # or
python -m mmqa.ui.app
```

The UI calls the **same `agent.predict()`** as the API — it does not call the HTTP layer, so it
works standalone. When `MMQA_BACKEND=stub`, the UI offers a **"generate synthetic scene"** button
(seed slider) so the demo runs with no model and no GPU — useful for the public Space where a
real ViLT download/GPU may be undesirable.

The UI deliberately surfaces the abstention/needs-review state prominently. For an assistive
use case (e.g. a blind user, in the spirit of VizWiz), the product promise is that the tool
**assists and abstains** rather than asserting a confident wrong answer; the badge is the
front-end embodiment of that contract.

---

## 4. Docker

A single image serves both API and UI; the entrypoint selects which to run.

### 4.1 System dependency: libGL

Pillow's image handling and the transformers vision stack pull in OpenCV-style native libraries
that require **`libGL`** (`libgl1`) at runtime. This is the classic
`ImportError: libGL.so.1: cannot open shared object file` that bites image projects on slim base
images. It must be installed in the image:

```dockerfile
FROM python:3.11-slim

# System deps for the vision stack (Pillow / OpenCV need libGL; libglib for some image ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# python-multipart is in requirements -> enables POST /ask (image upload)
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV MMQA_BACKEND=stub \
    MMQA_DEVICE=cpu \
    MMQA_API_HOST=0.0.0.0 \
    MMQA_API_PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "mmqa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Two image flavours

- **CPU / stub image (default, small).** `MMQA_BACKEND=stub`, no torch/transformers wheels needed
  beyond Pillow — a few hundred MB. This is the image CI builds and the public Space can run for
  the synthetic demo. Fast to build, no model download.
- **GPU / real image (large).** Built on a CUDA-enabled base (e.g. an NVIDIA PyTorch base image),
  installs torch + transformers, sets `MMQA_BACKEND=vilt` and `MMQA_DEVICE=cuda`. ViLT weights
  (~113M) are baked in at build time or mounted/cached at `MMQA_HF_CACHE` to avoid a cold-start
  download. Run with `--gpus all`.

```bash
# CPU / stub
docker build -t mmqa:stub .
docker run -p 8000:8000 mmqa:stub

# GPU / real ViLT
docker build -t mmqa:gpu -f Dockerfile.gpu .
docker run --gpus all -p 8000:8000 -e MMQA_BACKEND=vilt -e MMQA_DEVICE=cuda mmqa:gpu
```

The HF model cache should be mounted (`-v $HOME/.cache/huggingface:/root/.cache/huggingface`) so a
restart does not re-download weights.

---

## 5. Hugging Face Space

The public demo is a **Gradio Space** wrapping the same UI.

- **Default Space = stub backend on CPU.** `MMQA_BACKEND=stub`, `MMQA_DEVICE=cpu`. It runs the
  synthetic-scene generator + `SceneStubVQA`, so the Space needs no GPU, no large weights, and no
  flagged dataset — it stays comfortably inside the free CPU tier and starts instantly. This is the
  recommended public default because it demonstrates the **full agent** (type routing, abstention,
  re-rank) without shipping any model the Space would have to download or any image data with an
  unconfirmed license.
- **Real-model Space (optional).** Set `MMQA_BACKEND=vilt` and select a GPU-backed Space hardware
  tier (T4 is sufficient for ViLT inference). ViLT is Apache-2.0, so it is clean to host. Add
  `HF_TOKEN` as a Space secret only if pulling private/gated assets — the default ViLT and
  `lmms-lab/VQAv2` eval set do not require it.
- Space config (`README.md` front-matter): `sdk: gradio`, an `app_file` pointing at the UI entry,
  and the env vars above. `/healthz` is not used by Spaces, but the same readiness logic guards the
  UI so it does not accept input before the backend is ready.

**License note for the Space:** the served model must be the Apache-2.0 ViLT (or BSD-3 BLIP / MIT
GIT / MIT BLIP-2 / Apache Qwen2-VL-2B). Do **not** host `llava-hf/llava-1.5-7b-hf` (llama2,
non-commercial-ish) or `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen Research, empty HF license) on a public
Space — they are flagged in the brief and reserved as offline upper-bound comparisons only.

---

## 6. GPU vs CPU serving

There are two distinct serving regimes, and the abstention/type-constraint logic is identical in
both — only D3 (the model call) differs.

| Aspect            | GPU (real ViLT)                              | CPU (stub) / CPU (real)                         |
|-------------------|----------------------------------------------|-------------------------------------------------|
| Backend           | `MMQA_BACKEND=vilt`, `MMQA_DEVICE=cuda`       | `MMQA_BACKEND=stub` (CPU) or `vilt` on CPU      |
| Model             | `dandelin/vilt-b32-finetuned-vqa` (~113M)    | `SceneStubVQA` (no model) / ViLT on CPU         |
| Hardware tier     | T4 (16GB) is plenty for inference; A100/L4 faster | any CPU; stub is essentially free          |
| Dependencies      | torch + transformers + vision stack          | Pillow only (stub); torch (real-on-CPU)         |
| Use               | production accuracy, low latency             | CI, offline demo, air-gapped, public Space      |

**Recommendations**

- **Production:** real ViLT on a single GPU. ViLT is small (~113M); a T4 serves it with low
  latency and the classification head gives the clean softmax the agent's calibration needs. No
  need for A100/H100 to *serve* — those tiers are for *training* / for the heavier generative
  upgrades (BLIP-2, Qwen2-VL) if you swap the backend.
- **CI / offline / Space default:** the stub. Torch-free, deterministic from a seed, exercises all
  5 decisions. This is the same SeedEngine pattern as P15's OCR project.
- **Real ViLT on CPU** is viable for low-throughput / cost-sensitive deployments (ViLT is small
  enough to run on CPU), but expect substantially higher per-request latency than GPU.

The agent never assumes a GPU. If `MMQA_DEVICE=cuda` is requested but CUDA is unavailable, the
backend logs a warning and falls back to CPU rather than failing — and `/healthz` / `/info` report
the actual device in use.

---

## 7. Configuration via environment variables

All deployment knobs are env vars read by the reused config object (defaults shown). Thresholds are
the **calibrated abstention** parameters from D4 and matter as much as the model id.

| Env var               | Default                               | Purpose                                                            |
|-----------------------|---------------------------------------|--------------------------------------------------------------------|
| `MMQA_BACKEND`        | `stub`                                | `stub` (SceneStubVQA) or `vilt` (real ViLT). Selects D3 backend.   |
| `MMQA_MODEL_ID`       | `dandelin/vilt-b32-finetuned-vqa`     | HF model id for the real backend.                                  |
| `MMQA_DEVICE`         | `cpu`                                 | `cpu` or `cuda` (falls back to CPU if CUDA missing).               |
| `MMQA_HF_CACHE`       | `~/.cache/huggingface`                | model/weights cache dir (mount this in Docker).                    |
| `MMQA_TOP_K`          | `5`                                   | candidates returned and scanned at D5.                             |
| `MMQA_TAU_CONF`       | `0.30`                                | D4 min `p_max` to answer.                                          |
| `MMQA_TAU_ENT`        | `1.5`                                 | D4 max entropy (nats) to answer.                                   |
| `MMQA_TAU_MARGIN`     | `0.10`                                | D4 min top1−top2 margin to answer.                                 |
| `MMQA_TEMPERATURE`    | `1.0`                                 | softmax temperature for calibration (tuned on held-out split).     |
| `MMQA_MAX_IMAGE_MB`   | `10`                                  | reject uploads larger than this (DoS / accidental-huge guard).     |
| `MMQA_API_HOST`       | `0.0.0.0`                             | bind host.                                                         |
| `MMQA_API_PORT`       | `8000`                                | bind port.                                                         |
| `MMQA_LOG_LEVEL`      | `INFO`                                | structured-logger level; per-decision (D1..D5) trace lines.        |
| `MMQA_RETAIN_IMAGES`  | `false`                               | if `false`, uploaded image bytes are processed in-memory and **not** persisted (privacy default). |
| `MMQA_ENABLE_LLM`     | `false`                               | optional advisory LLM brain (anthropic); OFF by default, never overrides the FSM. |
| `ANTHROPIC_API_KEY`   | (unset)                               | only read when `MMQA_ENABLE_LLM=true`.                             |

Per-question-type threshold overrides (e.g. stricter `tau` for `count`, looser for `yes_no`) are
also configurable; the abstention gate is a **tuned** gate, not a hard-coded constant, because VQA
models are overconfident.

---

## 8. Privacy, robustness, and operational notes

- **Image privacy is a first-class concern.** VQA runs on user photos that can contain faces,
  homes, documents, or medical images, and the assistive use case (blind users) is exactly the
  population least able to vet what is uploaded. The deployment default is **local/in-memory
  processing with `MMQA_RETAIN_IMAGES=false`** — raw image bytes are decoded, fed to the model, and
  discarded; nothing is written to disk by default. Any logging of requests records the
  question + answer + status, never the raw image. Consent and a clear notice belong in the UI
  copy.
- **Abstain, never assert false certainty.** The whole point of the agent is that low-confidence
  answers come back as `"unsure"` / `needs_review` rather than confident hallucinations. This is
  non-negotiable for accessibility and any medical-adjacent image. The API exposes `abstain` and
  `needs_review` precisely so a downstream caller can withhold or escalate.
- **Language-prior bias.** VQA models answer "yes" / "2" / "white" from the question alone,
  ignoring the image. This is a *model* property, mitigated by reporting the blind-prior baseline
  in evaluation (see the evaluation docs), not by the serving layer — but operators should know a
  fluent answer is not proof the image was used.
- **License posture at serve time.** The served default is the clean **Apache-2.0** ViLT. Flagged
  assets (`HuggingFaceM4/VQAv2`, `merve/vqav2-small`, VizWiz, etc. — undeclared licenses;
  `llava-1.5-7b`, `Qwen2.5-VL-3B` — non-commercial / empty-license) must **not** be baked into a
  public deployment. `/info` exposes the active license + flag so this is auditable at runtime.
- **Robustness guards built into the path:** D1 rejects blank/corrupt images and non-questions
  before the model; `MMQA_MAX_IMAGE_MB` caps upload size; out-of-vocab gold answers are an inherent
  classification ceiling (3129 answers) and surface as abstentions rather than wrong confident
  answers; image quality/blur and adversarial questions are handled by the same calibrated
  abstention gate.

---

## 9. Quick start (copy-paste)

```bash
# 1. Offline stub server — no torch, no GPU, no downloads
MMQA_BACKEND=stub uvicorn mmqa.api.app:app --port 8000

# 2. Health + synthetic scene answer (JSON, no upload needed)
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/ask-scene \
  -H 'content-type: application/json' \
  -d '{"seed": 7, "question": "how many red squares?"}'

# 3. Real ViLT on GPU, then ask about an uploaded image (multipart)
MMQA_BACKEND=vilt MMQA_DEVICE=cuda uvicorn mmqa.api.app:app --port 8000
curl -s -X POST localhost:8000/ask \
  -F 'image=@kitchen.jpg' \
  -F 'question=how many chairs are there?'
```

Expected `/ask` response shape (abbreviated):

```json
{"answer": "2", "status": "ok", "question_type": "count", "answer_type": "number",
 "confidence": 0.74, "abstain": false, "needs_review": false,
 "topk": [{"answer": "2", "score": 0.74}, {"answer": "3", "score": 0.11}],
 "model_id": "dandelin/vilt-b32-finetuned-vqa", "latency_ms": 138}
```

If the model is unsure, the same call returns
`{"answer": "unsure", "status": "abstained:low_confidence", "abstain": true, "needs_review": true, ...}`
with HTTP `200` — an abstention is a correct, successful response, not an error.
