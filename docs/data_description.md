# P17 Multimodal Question Answering (VQA) — Data Description

> **Package:** `mmqa` · **Folder:** `17_Multimodal_QA` · **Author:** Le Dinh Minh Quan (student 23127460)
>
> This is the project's first multimodal-vision system (P02–P15 were text/document/OCR; P16 done). The data that feeds it is therefore a mix of **image+text benchmark corpora** (VQAv2 and friends), a **fixed answer vocabulary** pulled from a model config, and a **synthetic, self-describing scene generator** that lets the whole agent and scorer run offline. This document describes every data source: what it is, its schema, its split sizes, its license, and why it is here.

---

## 1. The task and what "data" means here

The trainable core is a **Visual Question Answering (VQA)** model: given an `(image, question)` pair it emits a short answer (`"how many red squares?" -> "2"`, `"what color is the circle?" -> "blue"`, `"is there a triangle?" -> "yes"`). Consequently every data item is a triple-plus:

```
(image, question) -> answer(s)
```

and for proper scoring it carries **ten human annotator answers** rather than one gold label, plus type tags. The VQA-specific quirks that shape all data handling:

- Answers are **short free text** (`"yes"`, `"2"`, `"white"`, `"baseball"`), normalized before comparison (§5 of the design brief).
- Ground truth is **10 annotators per question**, scored with the official soft VQA accuracy `min(1, matches/3)` — so any dataset used for *scoring* must preserve all 10 annotator dicts.
- Each question is tagged with an **`answer_type ∈ {yes/no, number, other}`** and a finer **`question_type`** (the canonical leading phrase, e.g. `"how many"`, `"what color is the"`), which drive both per-type reporting and the agent's type-constraint logic.

Four data roles are kept strictly separate: **TRAIN**, **EVAL**, **DEMO/smoke**, and **OFFLINE/CI** (the synthetic generator). A fifth pseudo-source — the **3129-answer vocabulary** — comes from a model config, not a dataset.

---

## 2. Source-of-truth table

All ids below are verified on the Hugging Face Hub. **Any repository that declares no license on the repo is treated as "license unconfirmed" and FLAGGED**, even though VQAv2's upstream sources (COCO 2014/2015 images and the VQA annotations) are both **CC-BY-4.0** and commercially usable in principle.

| id | Role | License | Schema / notes |
|---|---|---|---|
| `lmms-lab/VQAv2` | **PRIMARY EVAL** (canonical benchmark) | **cc-by-4.0 (clean)** | 8 columns, clean parquet, embedded PIL images, full 10-annotator structure. **No train split.** Soft-VQA-accuracy-ready. |
| `HuggingFaceM4/VQAv2` | **PRIMARY TRAIN** (only common mirror with a train split + full annotator schema) | **unspecified-on-hub — FLAG** (annotations + COCO images CC-BY-4.0 upstream) | Full VQAv2 fields incl. 10-annotator `answers`. Loading-script dataset; needs `trust_remote_code=True`; fragile. |
| `merve/vqav2-small` | DEMO / fast smoke loop | **unspecified — FLAG** | Only 3 columns (`image`, `question`, `multiple_choice_answer`). **Not valid for official scoring or type routing.** Demo only. |
| `Multimodal-Fatima/VQAv2_sample_train` | **TINY TRAIN with FULL schema** (offline-friendly) | **unspecified — FLAG** | 1.0K rows, 13 columns including the 10 annotator dicts; works offline once cached; best small set that keeps proper scoring. |
| `dandelin/vilt-b32-finetuned-vqa` | **ANSWER-VOCAB SOURCE** (3129 labels) | **apache-2.0 (clean)** | A **MODEL, not a dataset**. Its `config.json` `id2label`/`label2id` is the canonical 3129-answer VQA label space. |

**Side / alternative benchmarks** (not the main loop, used for stress tests and ablations):

| id | Role | License | Why it is here |
|---|---|---|---|
| `lmms-lab/GQA` | Compositional / structured-scene reasoning | **mit (clean)** | Tests multi-step relational questions. |
| `facebook/textvqa` | Text-in-image / OCR-reasoning ("read the text") | **cc-by-4.0 (clean)** | The one side benchmark that targets text-in-scene; *not* the primary task. |
| `lmms-lab/OK-VQA` | Knowledge-based VQA | **unspecified — FLAG** | Good abstention stress test (answers need outside knowledge). |
| `lmms-lab/VizWiz-VQA` | Real blind-user photos with an explicit **unanswerable** label | **unspecified — FLAG** | **Ideal calibrated-abstention test** — has a native "unanswerable" class to evaluate the agent's "unsure" gate. |
| `Luo-wj/DAQUAR` | Small classic indoor-scene VQA | **unspecified — FLAG** | Only viable DAQUAR mirror with real data. |

**Explicitly avoided:**
- `thangduong0509/daquar_vqa` — **cc-by-nc-sa-4.0 (NON-COMMERCIAL). Do not use.** (Use `Luo-wj/DAQUAR` if a DAQUAR mirror is needed, but treat its license as unconfirmed.)
- `jp1924/VisualQuestionAnswering` — gated + Korean, not VQAv2.

---

## 3. Primary EVAL — `lmms-lab/VQAv2`

This is the **canonical evaluation benchmark** and the cleanest source we have: a **CC-BY-4.0** parquet dataset with embedded images and the full 10-annotator structure intact, so the official soft VQA accuracy can be computed locally.

### 3.1 The 8-column schema

| Column | Type | Description |
|---|---|---|
| `question` | `string` | The natural-language question, e.g. `"What color is the bus?"` |
| `image` | `PIL.Image` (RGB) | The COCO scene image, embedded in the parquet. |
| `answers` | `List[dict]` (length 10) | **The 10 annotator answers.** Each dict = `{answer: str, answer_confidence ∈ {yes, maybe, no}, answer_id ∈ 1..10}`. |
| `multiple_choice_answer` | `string` | The consensus / most-frequent annotator answer (a convenience field; the soft metric still uses all 10). |
| `question_type` | `string` | Canonical leading phrase, one of ~65 buckets (`"how many"`, `"what color is the"`, `"is the"`, `"are there"`, ...). Drives per-question-type reporting. |
| `answer_type` | `string` | One of **`{yes/no, number, other}`** — the headline 3-bucket breakdown. |
| `image_id` | `int` | COCO image id (links back to the source photo). |
| `question_id` | `int` | Unique question id (used for EvalAI submission on held-out splits). |

### 3.2 The 10-annotator structure (why it matters)

The `answers` field is **the heart of the metric**. The official VQA accuracy samples over the 10 humans:

```
acc(prediction) = (1/10) * Σ over the 10 leave-one-out subsets  min(1, matches_in_subset / 3)
```

A prediction matching **≥3** of the 10 annotators scores **1.0**; 2 → 0.667; 1 → 0.333; 0 → 0.0. Both the prediction and all 10 GT answers pass through the **same** canonical normalization first. **A dataset without all 10 annotators cannot produce this score** — which is exactly why `merve/vqav2-small` is demo-only.

### 3.3 Splits and sizes

| Split | Rows | GT available? |
|---|---|---|
| `validation` | **214.4K** | **Yes — score locally.** This is our working eval split. |
| `testdev` | 107.4K | No (held out — EvalAI server) |
| `test` | 447.8K | No (held out — EvalAI server) |
| **Total** | **~769.5K** | — |
| **train** | **— (absent)** | This mirror has **NO train split.** |

Because the parquet ships pre-decoded images and is license-clean, **`lmms-lab/VQAv2` validation is the default evaluation set** for every reported number (overall, per-answer-type, per-question-type, abstention rate).

---

## 4. Primary TRAIN — `HuggingFaceM4/VQAv2`

To **fine-tune** the ViLT core on real VQAv2 we need a train split that still carries 10 annotators. The only common mirror that has both is `HuggingFaceM4/VQAv2`.

### 4.1 Schema and splits

It exposes the **full VQAv2 fields**, including the 10-annotator `answers` list, `question_type`, `answer_type`, `image`, `question`, `image_id`, and `question_id` — i.e. the same shape as the eval mirror, so train/eval handling is symmetric.

| Split | Approx. rows |
|---|---|
| `train` | **~443K** |
| `validation` | ~214K |
| `test` | held out |

### 4.2 License — FLAG

**The repo declares no license on the Hub** → treated as **"license unconfirmed"** and **FLAGGED**. Upstream the data is COCO 2014/2015 images (CC-BY-4.0) + VQA annotations (CC-BY-4.0), so it is commercially usable *in principle* — but verify before any commercial use.

### 4.3 Loading caveats (important, fragile)

This is a **loading-script dataset**, not parquet, and it has known traps:

- Requires **`load_dataset("HuggingFaceM4/VQAv2", trust_remote_code=True)`** — it runs a remote Python loading script.
- The **Dataset Viewer returns 501** (no preview on the Hub page).
- It is **fragile on newer `datasets` versions**; pin the `datasets` version when using it.
- It **may pull COCO images by URL** at load time, so it is network-heavy and flaky in CI.

**Decision:** use `HuggingFaceM4/VQAv2` **only** when you must fine-tune on real VQAv2 train. For evaluation always prefer the clean `lmms-lab/VQAv2` parquet. For everything offline/CI, use the synthetic generator (§7).

---

## 5. The 3129-answer vocabulary — from a model config, not a dataset

The **DEFAULT trainable core is classification VQA**: a linear head over a **fixed answer vocabulary** produces logits → softmax → top-k answers with clean per-answer probabilities. That fixed vocabulary is the **canonical 3129-answer VQAv2 label space**, and we source it directly from the default model's config:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
id2label = cfg.id2label            # len == 3129
answer_vocab = list(id2label.values())   # the 3129 canonical answers
```

- `dandelin/vilt-b32-finetuned-vqa` is **apache-2.0 (clean)** and is a **model**, used here purely as the authoritative source of `id2label` / `label2id`.
- This same 3129-answer list is the **single source of truth** for three things that must agree: (a) the classification head's label space, (b) the agent's type-constraint answer sets at D5 (yes/no, color, count subsets are slices of this vocab), and (c) the report buckets.
- **Hard ceiling:** any gold answer **outside** these 3129 answers is unreachable by the classification model. If you instead fine-tune from `dandelin/vilt-b32-mlm` with a custom vocabulary, the metric, the constraint sets, and the report buckets **must all use that same custom vocab.** Generative models (BLIP/GIT) avoid this ceiling but trade away the clean softmax confidence signal.

---

## 6. DEMO / smoke-test sets

Two small sets exist for fast iteration. They are **not** for headline numbers.

### 6.1 `merve/vqav2-small` — quick smoke loop (FLAG)

- **3 columns only:** `image`, `question`, `multiple_choice_answer`.
- `validation` **21.4K rows**, parquet, embedded images.
- **No 10 annotators** → **invalid for official soft VQA accuracy**. **No `question_type` / `answer_type`** → cannot drive type routing or per-type reporting.
- License **unspecified — FLAG**.
- **Use only** as a fast Colab smoke loop ("does the pipeline run end-to-end?"), never for reported accuracy.

### 6.2 `Multimodal-Fatima/VQAv2_sample_train` — tiny train with full schema (FLAG)

- **train 1.0K rows (~155MB)**, **13 columns**, including **`answers_original`** (the 10 annotator dicts), `question_type`, `answer_type`, `question_id`, embedded `image`, plus extras (`blip_caption`, `clip_tags`, `DETA_detections`).
- License **unspecified — FLAG**.
- **The best small set that keeps the 10-annotator structure**, so it supports *proper soft scoring offline once cached*. Use it when you need a real (small) accuracy number without network access but still want correct VQA-accuracy semantics.

---

## 7. SYNTHETIC scene generator — the primary OFFLINE / CI data source (REQUIRED)

### 7.1 Why it exists

The real VQA stack is heavy and flaky: `torch` + `transformers` + a vision stack + multi-GB parquet/COCO images. `HuggingFaceM4/VQAv2` runs a remote loading script and may fetch images by URL; the model is 100M+ parameters. None of that survives a clean CI run or a network-free Colab session.

So — mirroring **P15's OCR `SeedEngine`** — P17 ships a **synthetic, self-describing scene generator** (`data/synth_scene.py`) plus a torch-free `SceneStubVQA` model. Together they let **the entire 5-decision agent and the full scorer run with NO torch, NO model download, and NO network**, deterministically and reproducibly from a seed. It produces a real accuracy / abstention-rate / re-rank-rate report in CI. Swapping `SceneStubVQA` for a real `ViltForQuestionAnswering` wrapper (same `predict()` returning top-k + probs) flips the system to production with **no agent changes**.

### 7.2 How a scene is generated — `make_scene(seed)` (PIL only)

1. **Sample objects.** Draw N objects (2–6), each with a `shape ∈ {square, circle, triangle}`, a `color` from a fixed 6-color lexicon `{red, blue, green, yellow, purple, orange}`, and a **non-overlapping** bounding box on a WxH (e.g. 224×224) white canvas, rendered with `ImageDraw` (`rectangle` / `ellipse` / `polygon`).
2. **Build a SCENE SPEC dict:**
   ```python
   spec = {
     "objects": [{"shape": ..., "color": ..., "bbox": [...]}, ...],
     "counts_by_color_shape": {...},   # e.g. {"red square": 2, "blue circle": 1}
     "colors_present": [...],
     "shapes_present": [...],
   }
   ```
3. **EMBED the spec in the PNG** so the image is **self-describing** and evaluation is fully deterministic:
   ```python
   from PIL import PngImagePlugin
   meta = PngImagePlugin.PngInfo()
   meta.add_text("scene_spec", json.dumps(spec))
   img.save(path, pnginfo=meta)
   ```

### 7.3 Templated QA derivation (one generator per type — exercises D2 routing and D5 constraints)

| Type | Question template | Gold answer source |
|---|---|---|
| `yes_no` | `"is there a {shape}?"` / `"is there a {color} {shape}?"` | `yes`/`no` from spec membership |
| `count` | `"how many {color} {shape}s?"` | integer from `counts_by_color_shape` |
| `color` | `"what color is the {shape}?"` | the color — **only emitted when that shape is unique**, so gold is unambiguous |
| `object` | `"what shape is the {color} object?"` | the shape |

Each generated item is `{image_path, question, gold_answer, question_type}`, and a **10-answer list is synthesized from the gold** so the official soft VQA-accuracy metric runs unchanged. The result mirrors the `lmms-lab/VQAv2` 8-column schema (`{PIL image, question, answers (List of 10), multiple_choice_answer, question_type, answer_type}`), so offline data is a drop-in for the real eval shape.

### 7.4 The `SceneStubVQA` model (no torch, no network)

The stub has the **same `predict(image, question)` signature** as the real wrapper and returns `topk + probs`:

- **(a)** Reads the spec back out of the PNG: `json.loads(Image.open(path).text["scene_spec"])`.
- **(b)** Computes the TRUE answer from spec + question via the same template logic.
- **(c)** Returns a **realistic distribution, not one-hot**: ~0.7–0.9 mass on the correct answer, the remainder spread over plausible **type-consistent distractors** (other colors, adjacent counts) — so `p_max` / entropy / margin at D3/D4 are meaningful and the **abstention gate is actually exercised**.
- **(d)** A `noise` / `difficulty` knob deliberately lowers `p_max` or injects a type-wrong top-1 on a fraction of items, so **D4 (abstain) and D5 (re-rank) trigger** on the eval set and can be unit-tested.

**Offline verification (confirmed):** the stub answers seed scenes correctly and **all 5 agent decisions fire** — the agent, eval, and tests run with no torch/model/network.

---

## 8. Splits, sizes, and roles at a glance

| Source | Split(s) | Size | Has 10 annotators? | Has type tags? | Role |
|---|---|---|---|---|---|
| `lmms-lab/VQAv2` | validation | 214.4K | ✅ | ✅ | **EVAL (headline numbers)** |
| `lmms-lab/VQAv2` | testdev / test | 107.4K / 447.8K | held out | held out | EvalAI-only |
| `HuggingFaceM4/VQAv2` | train | ~443K | ✅ | ✅ | **TRAIN (fine-tune)** |
| `Multimodal-Fatima/VQAv2_sample_train` | train | 1.0K | ✅ | ✅ | tiny offline train / real scoring |
| `merve/vqav2-small` | validation | 21.4K | ❌ | ❌ | DEMO smoke only |
| Synthetic generator | seeded, in-memory | arbitrary | ✅ (synthesized) | ✅ | **OFFLINE / CI (primary)** |
| `dandelin/vilt-b32-finetuned-vqa` config | n/a | 3129 labels | n/a | n/a | **ANSWER VOCAB** |

---

## 9. Licensing summary and decisions

**Commercially safe (license declared):**
- `lmms-lab/VQAv2` — **cc-by-4.0**
- `facebook/textvqa` — **cc-by-4.0**
- `lmms-lab/GQA` — **mit**
- `dandelin/vilt-b32-finetuned-vqa` — **apache-2.0**

**FLAG — license unconfirmed** (undeclared on the repo; CC-BY-4.0 upstream for the VQAv2/COCO content; verify before commercial use):
- `HuggingFaceM4/VQAv2`, `merve/vqav2-small`, `Multimodal-Fatima/VQAv2_sample_train`, `lmms-lab/OK-VQA`, `lmms-lab/VizWiz-VQA`, `Luo-wj/DAQUAR`.

**AVOID — non-commercial:**
- `thangduong0509/daquar_vqa` — **cc-by-nc-sa-4.0. Do not use.**

**Upstream provenance note.** VQAv2 images are **COCO 2014/2015 (CC-BY-4.0)** and the VQA annotations are **CC-BY-4.0**. This makes the underlying content commercially usable in principle, but **several practical HF mirrors declare no license on the repo** — so the *repo*, not the upstream content, is what gets flagged. For anything that must be commercially defensible, prefer the declared-clean sets (`lmms-lab/VQAv2`, `facebook/textvqa`, `lmms-lab/GQA`) and the clean model config vocab.

**Default data decision.**
- **TRAIN** = `HuggingFaceM4/VQAv2` (only train mirror with 10 annotators; FLAG license; needs `trust_remote_code=True`).
- **EVAL** = `lmms-lab/VQAv2` validation (cc-by-4.0, clean parquet).
- **DEMO** = `merve/vqav2-small` (quick) or `Multimodal-Fatima/VQAv2_sample_train` (when you need real scoring offline).
- **VOCAB** = `dandelin/vilt-b32-finetuned-vqa` config (3129 answers).
- **CI / OFFLINE** = the synthetic scene generator + `SceneStubVQA` (§7).

---

## 10. Data-handling notes and gotchas

- **Preserve all 10 annotators** in any pipeline that scores. Dropping `answers` down to `multiple_choice_answer` (as `merve/vqav2-small` does) silently breaks the official metric — the score will be wrong, not just noisier.
- **One normalization function, applied identically** to predictions and all 10 references (lowercase; punctuation removal that keeps decimals like `2.5` and merges digit groups like `100,000` → `100000`; number-word → digit; drop articles `a/an/the`; canonicalize contractions). Generative answers especially must pass through it or correct answers score as misses.
- **Type tags drive both reporting and the agent.** `answer_type ∈ {yes/no, number, other}` is the headline 3-bucket breakdown; `question_type` (≈65 buckets) is the finer appendix. Both come straight from the annotations in `lmms-lab/VQAv2` and `HuggingFaceM4/VQAv2`; the synthetic generator and the keyword classifier reproduce the same tags so offline reports have the same shape.
- **Answer-vocab mismatch is a hard ceiling** for the classification core: gold answers outside the 3129 labels are unreachable. Keep the vocab consistent across head, constraint sets, and report buckets.
- **`HuggingFaceM4/VQAv2` is the fragile one** — loading script, Dataset-Viewer 501, `trust_remote_code=True`, possible image-by-URL fetches, version-sensitive. Reserve it strictly for the train split and pin versions; never put it on the CI path.
- **Language-prior baselines need the type tags too.** The blind/question-only and most-common-answer-per-type baselines (the bias ablations) all read `question_type` / `answer_type`, so they only run on the schema-complete sources (`lmms-lab/VQAv2`, `HuggingFaceM4/VQAv2`, `Multimodal-Fatima/VQAv2_sample_train`, and the synthetic generator).
