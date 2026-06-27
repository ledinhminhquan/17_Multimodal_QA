# Multimodal Question Answering — VQA (`mmqa`)

> Answer a natural-language **question** about an **image** — with a trainable VQA core, calibrated
> abstention, and answer constraints, as a debuggable, license-clean, production-grade pipeline.
>
> **NLP in Industry — Final Assignment, Project #17.** Author: **Le Dinh Minh Quan** (Student 23127460).

`mmqa` runs a **trainable VQA core** (`dandelin/vilt-b32-finetuned-vqa`, fine-tuned) behind a
deterministic **agent (D1–D5)** that classifies the question type, runs the model, **abstains** when
the model is uncertain (VQA models are overconfident), and **constrains** the answer to the question
type (a yes/no question gets a yes/no answer, a counting question a number). Only the VQA model is
trained; the routing, the abstention gate, and the constraints are algorithmic.

```
image + question ──► classify question ──► VQA model (top-k + confidence)
                                                  │
            answer ◄── constrain to type ◄── abstain if unsure ◄┘   (else "unsure" + review)
```

---

## How this repo meets each assignment requirement

| Requirement | Where it is delivered |
|---|---|
| **Problem definition** | [docs/problem_definition.md](docs/problem_definition.md) |
| **Data description + data card** | [docs/data_description.md](docs/data_description.md), [docs/data_card.md](docs/data_card.md); synthetic scenes [src/mmqa/data/synth_scene.py](src/mmqa/data/synth_scene.py) |
| **Model selection + baseline** | [docs/model_selection.md](docs/model_selection.md); VQA core [src/mmqa/models/vqa_model.py](src/mmqa/models/vqa_model.py); baselines [baseline.py](src/mmqa/models/baseline.py) |
| **Training + evaluation** | [src/mmqa/training/train_vqa.py](src/mmqa/training/train_vqa.py), [evaluate.py](src/mmqa/training/evaluate.py); [docs/vqa_evaluation.md](docs/vqa_evaluation.md) |
| **Agentic AI component** | [src/mmqa/agent/](src/mmqa/agent/) — 5 decision points; [docs/agent_architecture.md](docs/agent_architecture.md) |
| **Deployment / serving** | FastAPI [src/mmqa/api/main.py](src/mmqa/api/main.py) + Gradio [ui.py](src/mmqa/api/ui.py); [docs/deployment.md](docs/deployment.md) |
| **Continual learning + monitoring** | [src/mmqa/monitoring/drift_report.py](src/mmqa/monitoring/drift_report.py); [docs/continual_learning_monitoring.md](docs/continual_learning_monitoring.md) |
| **Privacy + robustness** | [docs/privacy_robustness.md](docs/privacy_robustness.md) |
| **Project plan** | [docs/project_plan.md](docs/project_plan.md) |
| **Ethics** | [docs/ethics_statement.md](docs/ethics_statement.md) |
| **Report + slides (auto-generated)** | [src/mmqa/autoreport/](src/mmqa/autoreport/) → `report.pdf` + `slides.pptx` |
| **Reproducible training** | H100 notebook [notebooks/MMQA_Colab_Training_H100_AUTOPILOT.ipynb](notebooks/MMQA_Colab_Training_H100_AUTOPILOT.ipynb) + [COLAB_GUIDE.md](notebooks/COLAB_GUIDE.md) |

---

## Repository layout

```
17_Multimodal_QA/
├── src/mmqa/
│   ├── config.py  cli.py  logging_utils.py
│   ├── data/        # samples (seed scenes), synth_scene (generator + answerer), dataset, download
│   ├── models/      # vqa_model (ViLT/BLIP/SceneStub), question_type, baseline, model_registry
│   ├── vision/      # image_utils (load, read embedded scene, SceneImage no-PIL carrier)
│   ├── training/    # train_vqa, train_baseline, evaluate, tune, metrics (VQA accuracy + normalization)
│   ├── agent/       # state, policy (D1-D5), tools, llm_orchestrator, vqa_agent
│   ├── api/         # schemas, dependencies, main (FastAPI), ui (Gradio), app_combined
│   ├── analysis/    # error_analysis, latency, per_type
│   ├── autoreport/  # artifact_loader, charts, report_pdf, slides_pptx
│   ├── monitoring/  # drift_report (job-log monitor)
│   ├── automation/  # autopilot (one button)
│   └── grading/     # checklist (rubric self-check)
├── configs/  docs/ (14 + DESIGN_BRIEF)  notebooks/  app/  deploy/  sample_data/  scripts/  tests/
├── Dockerfile  docker-compose.yml  Makefile  pyproject.toml  requirements*.txt  .github/workflows/ci.yml
└── LICENSE (MIT)  README.md
```

---

## Models & data (all verified on the HF Hub)

| Slot | Default (shipped) | License | Alternatives |
|---|---|---|---|
| **VQA core (trained)** | [`dandelin/vilt-b32-finetuned-vqa`](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) (classification, 3129 answers) | **Apache-2.0** | `Salesforce/blip-vqa-base` (BSD, generative); `microsoft/git-base-vqav2` (MIT); H100: `Salesforce/blip2-flan-t5-xl` (MIT) |
| **Train data** | [`HuggingFaceM4/VQAv2`](https://huggingface.co/datasets/HuggingFaceM4/VQAv2) (10 annotators, `trust_remote_code`) | undeclared → flag (COCO/VQA CC-BY-4.0 upstream) | `Multimodal-Fatima/VQAv2_sample_train` (1K, full schema) |
| **Eval data** | [`lmms-lab/VQAv2`](https://huggingface.co/datasets/lmms-lab/VQAv2) validation | **CC-BY-4.0** | `facebook/textvqa` (CC-BY), `lmms-lab/GQA` (MIT) |
| **Answer vocab** | ViLT config `id2label` (3129) | Apache | — |
| **Primary offline data** | **synthetic scene generator** (`data/synth_scene.py`) | — | shapes + embedded spec → SceneStubVQA |

**Avoided** (flagged): `google/pix2struct-vqav2-base` (does not exist / 404), `llava-hf/llava-1.5-7b-hf`
(Llama-2 non-commercial), `Qwen/Qwen2.5-VL-3B` (Qwen research license).

**Why a synthetic scene generator?** VQA needs both an image and a model, both heavy. The generator
draws colored shapes, embeds the scene spec in the PNG, and templates `(question, gold answer, type)`
triples; the offline `SceneStubVQA` reads the embedded scene to answer — so the agent, the evaluation
and the tests run with **no torch, no model, no network** (mirrors the OCR SeedEngine in P15).

---

## Quickstart

```bash
pip install -e .                 # core: runs offline (SceneStubVQA + synthetic scenes)
pip install -e .[all]            # + torch/transformers (ViLT), FastAPI/Gradio, reportlab

mmqa demo-agent --fast                                          # the 5-decision agent on seed scenes
mmqa ask-scene --question "what color is the circle?" --scene 0 --fast
mmqa ask --image sample_data/sample_scene.png --question "how many shapes are there?"   # real ViLT
mmqa evaluate --fast                                           # VQA accuracy vs baselines + per-type
mmqa autopilot --no-train                                      # report.pdf + slides.pptx + grade + bundle
bash scripts/smoke.sh                                          # full offline smoke
```

`--fast` uses the SceneStubVQA (no download). Drop it to use the fine-tuned ViLT.

## Train on Colab (H100, auto-adapts A100/L4/T4 — ViLT fine-tunes even on a free T4)

Push this folder to GitHub (or Drive), open `notebooks/MMQA_Colab_Training_H100_AUTOPILOT.ipynb`,
set the controls in cell 0, and **Runtime → Run all**. It fine-tunes the ViLT core (resume-safe),
runs the full evaluation, and writes `report.pdf` + `slides.pptx` + the bundle to Drive. See
[notebooks/COLAB_GUIDE.md](notebooks/COLAB_GUIDE.md).

## The agent (the mandatory agentic component)

A deterministic FSM `ingest → classify → answer → calibrate → constrain` with **five decision points**:

| # | Decision | Gates on | Branches |
|---|---|---|---|
| **D1** | input gate | valid image + non-empty question | ok / no_image / no_question |
| **D2** | question-type | keyword classifier | yes_no / number / color / object / ... |
| **D3** | answer | run VQA → top-k + confidence | ok / empty |
| **D4** | **calibrated abstention** | max-prob / margin / entropy | answer / **abstain → "unsure"** |
| **D5** | **type-consistency** | answer vs question type | consistent / **reranked** / type_default |

The value-add over a raw argmax: the model says **"unsure"** instead of hallucinating, and a yes/no
question never gets a colour as its answer. An optional LLM brain (`anthropic`) is **off by default**.

## Serving

```bash
mmqa serve --ui              # FastAPI on :8000 + Gradio demo at /ui
# POST /ask        (upload image + question -> answer + confidence + abstain flag)
# POST /ask-scene  (JSON {question, scene})        GET /healthz  /version
docker compose up --build    # containerized (libGL baked in)
```

## Verified offline results

On the synthetic seed scenes (offline, SceneStubVQA): **model VQA accuracy 1.0 vs blind question-only
prior 0.26** (the gap proves the system uses the image, not just language priors), per-type all 1.0,
agent coverage 1.0, **grade 1.0**, all 5 decision points fire, 27 tests pass. The offline number is the
perfect-scene-reading upper bound; on real VQAv2 the fine-tuned ViLT gives a realistic accuracy with
calibrated abstention — the honest comparison happens on Colab.

## Tests

```bash
pytest -q        # CPU-only, no downloads (HF_HUB_OFFLINE); the scene stub stands in for the model
```

## Documentation

[Problem](docs/problem_definition.md) · [Data](docs/data_description.md) · [Data card](docs/data_card.md) ·
[Models](docs/model_selection.md) · [Architecture](docs/architecture.md) · [Agent](docs/agent_architecture.md) ·
[Evaluation](docs/vqa_evaluation.md) · [Deployment](docs/deployment.md) ·
[Continual learning & monitoring](docs/continual_learning_monitoring.md) ·
[Privacy & robustness](docs/privacy_robustness.md) · [Ethics](docs/ethics_statement.md) ·
[Project plan](docs/project_plan.md) · [Model card](docs/model_card.md) · [Slides outline](docs/slide_deck_outline.md) ·
[Design brief](docs/DESIGN_BRIEF.md)

## Ethics & license

VQA on user photos is **sensitive** (faces, homes, documents; assistive use for blind users like
VizWiz): the default path processes images transiently, logs metadata only, and the LLM brain is off.
The tool **assists** and **abstains** on low-confidence answers — it never asserts certainty on
high-stakes questions. VQA models carry strong **language priors** and COCO demographic bias; the
**blind-prior baseline** and **per-type accuracy** are reported to surface this. Code is **MIT**
([LICENSE](LICENSE)); the shipped model is permissive (ViLT Apache); non-commercial options are flagged.
