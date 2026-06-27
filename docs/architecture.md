# P17 Multimodal Question Answering (VQA) — System Architecture

Author: Le Dinh Minh Quan (student 23127460) · Package `mmqa` · Folder `17_Multimodal_QA`

This document describes the end-to-end system architecture of P17: the inference pipeline (image preprocess + question tokenize → VQA model → top-k answers → agent gates), the module map under `src/mmqa`, the data-flow between stages, and the offline / degradation design that lets the entire agent and scorer run with no torch, no model download, and no network.

P17 is the **first multimodal-vision** project in this series. P02–P15 operated on text, documents, and OCR; P16 is done. Where those projects consumed strings, P17 consumes an **`(image, question)` pair** and emits a short natural-language answer. The vision modality is genuinely new, but the *agentic shape* — a deterministic finite-state machine wrapping a single trained core, degrading to a torch-free stub for offline runs — is deliberately inherited from P15's OCR `SeedEngine` pattern so the proven plumbing is reused rather than reinvented.

---

## 1. What the system does

Given an **image** and a **natural-language question about it**, produce a **short answer**. Example: a scene of colored shapes + "how many red squares?" → "2"; "what color is the circle?" → "blue"; "is there a triangle?" → "yes".

The single trainable artifact is the **VQA model** — the multimodal transformer that consumes image patches + question tokens and emits an answer. Everything else (image decode/resize, question tokenization, answer normalization, the agent's decision logic, scoring) is pretrained or deterministic and is **not** trained.

The system supports two answer regimes and defaults to the first:

- **Classification VQA (DEFAULT).** A fixed answer vocabulary — the canonical **3129-answer** VQAv2 label space — with a linear head → logits → softmax over the vocab → top-k answers with calibrated-ish probabilities. Clean metric, clean confidence signal, easy to train and evaluate. **This is the regime the agent is designed against**, because a per-answer softmax distribution is the cleanest possible input for a calibrated abstention gate. Default core: `dandelin/vilt-b32-finetuned-vqa` (apache-2.0, ~113M, ViLT single-stream VL transformer).
- **Generative VQA (alternative).** Autoregressive decoding of free-form answer text (open vocabulary) — `Salesforce/blip-vqa-base` (bsd-3) or `microsoft/git-base-vqav2` (mit). Handles answers outside the fixed vocab; confidence becomes the sequence/token logprob. Harder to evaluate (decode → normalize → exact-match), used as a complement and as an upper-bound comparison.

Wrapping the model is a **deterministic finite-state agent** (5 decision points) that classifies the question type, runs the model, **abstains when uncertain**, and **constrains the answer to the question type**. This self-checking wrapper — not a bigger model — is the agentic value-add: a stock ViLT/BLIP only does raw argmax; the agent adds calibrated abstention and a type-aware answer constraint with **no extra training**.

---

## 2. High-level pipeline

Six logical stages, each tagged **PRETRAINED**, **TRAINED**, or **ALGORITHMIC**. Only the multimodal encode + answer head are trained — the rest is the model's own processor or deterministic agent logic.

| # | Stage | Status | Component |
|---|-------|--------|-----------|
| 1 | Image preprocess | PRETRAINED / algorithmic | RGB-convert, resize to model resolution (ViLT/BLIP 384px), normalize — done by the model's own image processor (`ViltImageProcessor`). Input validation (degenerate-blank detection) lives in `vision/image_utils`. |
| 2 | Question tokenize | PRETRAINED | WordPiece (`BertTokenizerFast` for ViLT/BLIP) or BPE/T5; fused as `[CLS] question [SEP]` with image patches. |
| 3 | Multimodal encode | **TRAINED (the core)** | Single-stream transformer over concatenated image-patch + text tokens (ViLT), or image encoder + Q-Former / cross-attn decoder (BLIP / BLIP-2 / GIT). |
| 4 | Answer head | **TRAINED (the core)** | Classification: 3129-way linear head → softmax. Generation: autoregressive decoder. |
| 5 | Answer post-process | ALGORITHMIC | Classification: argmax / top-k label lookup over the vocab. Generation: decode string → apply canonical VQAv2 answer-normalization. |
| 6 | Agent wrapper + scoring | ALGORITHMIC (no training) | Deterministic 5-point state machine over the model's top-k + softmax (§4); soft VQA accuracy over 10 annotators. |

**Three production entry points** the notebook demonstrates, in increasing control:

1. `transformers.pipeline('visual-question-answering', model='dandelin/vilt-b32-finetuned-vqa')` → `[{answer, score}, ...]`. The quick classification path.
2. Explicit `ViltProcessor` + `ViltForQuestionAnswering` → raw `outputs.logits` → softmax → top-k + confidence. **This is the path the agent uses** — it needs the full distribution, not just argmax.
3. `BlipProcessor` + `BlipForQuestionAnswering` with `.generate()` — the open-vocab generative path.

**Architectural key point:** classification VQA gives a per-answer softmax distribution that is the cleanest input for the calibrated abstention gate (D4); generative VQA must fall back to logprob/beam-score as the confidence proxy. The agent targets regimes (1)/(2).

---

## 3. Data-flow diagram

```
                          ┌──────────────────────────────────────────────┐
   (image, question) ────▶│  AGENT FSM  (src/mmqa/agent)                  │
   or (scene_png, q)      │  deterministic router, 5 decision points     │
                          └──────────────────────────────────────────────┘
                                              │
   ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
   ▼                                           ▼                                           ▼
┌─────────────┐  D1 INPUT GATE        ┌────────────────────┐  D2 QUESTION-TYPE ROUTER
│   INGEST    │  valid RGB image,     │     CLASSIFY       │  keyword/regex →
│ vision/     │  non-blank (var>eps), │  models/           │  {yes_no, count, color,
│ image_utils │  non-empty question,  │  question_type     │   object/other}
│             │  interrogative        │                    │  → defines allowed answer set
└─────────────┘                       └────────────────────┘
   │ bad image  → halt error:bad_image    │
   │ bad ques.  → halt error:bad_question  │  question_type forwarded to D5
   └──────────────────────────────────────┤
                                          ▼
                               ┌────────────────────┐  D3 RUN VQA  (only step that
                               │      ANSWER        │  touches the model)
                               │  models/vqa_model  │  real: ViltForQuestionAnswering
                               │  | SceneStubVQA    │        → logits → softmax
                               │  (offline)         │  stub: read embedded scene_spec
                               └────────────────────┘  → topk(k=5), p_max, margin, entropy
                                          │  exception → halt error:model_failure
                                          ▼
                               ┌────────────────────┐  D4 CALIBRATED ABSTENTION GATE
                               │     CALIBRATE      │  p_max ≥ tau_conf AND
                               │  agent/calibrate   │  entropy H ≤ tau_ent AND
                               │                    │  margin (p1-p2) ≥ tau_margin ?
                               └────────────────────┘
                                          │ no  → answer='unsure'
                                          │       status='abstained:low_confidence'
                                          │ yes
                                          ▼
                               ┌────────────────────┐  D5 TYPE-CONSISTENCY GATE + RE-RANK
                               │     CONSTRAIN      │  top1 in D2 allowed set?
                               │  agent/constrain   │  else scan top-k for best
                               │                    │  type-consistent candidate
                               └────────────────────┘
                                          │  no top-k match → status='abstained:type_mismatch'
                                          ▼
                  {answer, status∈{ok, ok:reranked, abstained:*, error:*},
                   question_type, confidence, topk}  → VQA-accuracy scoring (analysis)
```

The five decision points (D1–D5) live in the agent; each consults the stage it gates. The agent never re-implements model inference — D3 is the only step that touches the model — it only routes over the model's own top-k + softmax.

---

## 4. The agent: 5 decision points

A **deterministic finite-state machine** wrapping the frozen VQA model. Each gate is auditable and offline-testable. Mapping question-types to VQA's 3 reported answer-types: `yes_no → yes/no`, `count → number`, everything else `→ other`. The final emitted answer is run through the canonical VQA normalization (§6 / `analysis`) before scoring.

| ID | State / module | Gates on (signal) | Branches |
|----|----------------|-------------------|----------|
| **D1** | ingest (`vision/image_utils`) | Validity of `(image, question)` **before** any model call. Image decodes to a valid nonzero RGB array and is not degenerate-blank (pixel-variance > eps); question is non-empty after strip, has ≥1 alphabetic token, length in [1,128] tokens, and is interrogative (ends in '?' OR starts with wh/aux). PIL + regex, **no torch**. | valid → D2 · corrupt/blank image → halt `error:bad_image` · empty/non-question → halt `error:bad_question`. Hard fail; never reaches the model. |
| **D2** | classify (`models/question_type`) | Deterministic lexical mapping to `{yes_no, count, color, object/other}`. 'is/are/does/do/can/has/was' or 'is there' → yes_no; 'how many'/'number of'/'count' → count; 'what color'/'which color' → color; else → object/other. Pure regex/keyword table. **The value-add anchor** — the chosen type defines the allowed answer set used at D5. | yes_no → `{yes,no}` · count → number words/digits `{0..20,'none'}` · color → closed color lexicon `{red,blue,green,yellow,...}` · object/other → open vocab (D5 becomes a pass-through no-op). |
| **D3** | answer (`models/vqa_model` \| `SceneStubVQA`) | Execute the model and capture the **full distribution**, not just argmax. Real: `ViltForQuestionAnswering → logits`. Offline: `SceneStubVQA` reads the embedded scene spec. Produces top-k (k=5) `(answer, prob)`, `p_max = p1`, Shannon entropy `H = -Σ pᵢ log pᵢ` over the truncated/renormalized dist, top1-top2 margin, and regime. Only step that touches the model. | always emits `{topk, p_max, entropy, margin, regime}`; runtime exception → halt `error:model_failure`; normal → D4. |
| **D4** | calibrate (`agent/calibrate`) | Whether the model is confident enough to answer at all. **VQA classifiers are overconfident**, so thresholds are calibrated on a held-out split (temperature-scaled softmax), not hard-coded. Signals: `p_max` vs `tau_conf` (e.g. 0.30) AND entropy `H` vs `tau_ent` (e.g. 1.5 nats) AND margin `(p1-p2)` vs `tau_margin`. Optionally per-question-type thresholds (count is harder → stricter). | all pass → CONFIDENT → D5 · fails any → ABSTAIN: answer='unsure', `status='abstained:low_confidence'` (or route to needs_review). |
| **D5** | constrain (`agent/constrain`) | Force the answer consistent with the D2 type by re-ranking **within** top-k. Is `topk[0]` in the allowed set? If not, scan top-k for the highest-prob entry that IS. Normalizes form (digit↔word for counts, canonical color spelling). Example: count question, top1='cat'(0.4), top2='3'(0.3) → re-rank to '3'. | top1 already consistent → `status='ok'` · a lower-ranked top-k entry is consistent → `status='ok:reranked'` · NO top-k entry matches → abstain `status='abstained:type_mismatch'`. For object/other the set is open → pass-through. |

**Value-add (no extra training).** A stock ViLT/BLIP model does raw argmax only. The agent adds, over the model's own top-k + softmax: (1) **calibrated abstention (D4)** — a temperature-scaled max-prob + entropy + margin gate lets the system say "unsure"/needs_review instead of emitting a confident wrong answer; the headline metric is *answer-when-confident accuracy and abstention rate vs raw argmax*; (2) **type-aware answer constraint + re-rank (D2→D5)** — restrict/re-rank to a type-consistent candidate, fixing the common failure where the argmax is fluent-but-type-wrong (answering 'cat' to 'how many?'). The same frozen model yields higher constrained accuracy.

An optional LLM brain (`anthropic`) is **OFF by default, advisory only, never overrides**. The agent runs fully offline on `SceneStubVQA` + the keyword classifier.

---

## 5. Module map (`src/mmqa`)

The package mirrors the proven layout of P15 imgtrans / P14 doctrans. The config / logging / registry / autoreport / monitoring / automation / grading / cli / api templates are reused; the multimodal model wrapper, vision handling, the synthetic-scene generator, the VQA-accuracy metric, the question-type classifier, and the type-aware abstaining agent are net-new for P17.

### `config`
Single source of truth for paths, model id, dataset ids, thresholds, seed, resolution, and batch size. Holds the decision-point constants (`tau_conf≈0.30`, `tau_ent≈1.5`, `tau_margin`, optional per-type overrides), the top-k value (`k=5`), the model registry defaults (`dandelin/vilt-b32-finetuned-vqa`), and the `MMQA_OFFLINE` flag. Typed dataclass config reused from P14/P15 so every module reads one object.

### `data`
Corpus loading and the **synthetic generator** — the primary offline data source.

- **`data/samples`** — tiny committed fixtures (fixed-seed scene PNGs + QA items) that drive unit tests and the torch-free smoke loop.
- **`data/synth_scene.py`** (NEW, the offline backbone) — `make_scene(seed)`: draws 2–6 non-overlapping colored shapes (`shape ∈ {square, circle, triangle}` × `color ∈ {red, blue, green, yellow, purple, orange}`) on a ~224×224 white PIL canvas with `ImageDraw`, builds a SCENE SPEC dict (`objects`, `counts_by_color_shape`, `colors_present`, `shapes_present`), and **embeds the spec in the PNG** via `PngImagePlugin.PngInfo().add_text('scene_spec', json.dumps(spec))` so the image is self-describing and eval is fully deterministic. Templated QA derivation (one generator per type, to exercise D2/D5): yes_no ("is there a {color} {shape}?"), count ("how many {color} {shape}s?"), color ("what color is the {shape}?" — only when that shape is unique so gold is unambiguous), object ("what shape is the {color} object?"). Each item = `{image_path, question, gold_answer, question_type}` plus a 10-answer list built from gold for soft scoring.
- **`data/dataset`** — loaders for the real splits, mapping each to the canonical `lmms-lab/VQAv2` 8-col schema (`question, image, answers (List of 10 {answer, answer_confidence∈{yes,maybe,no}, answer_id 1..10}), multiple_choice_answer, question_type, answer_type∈{yes/no,number,other}, image_id, question_id`). EVAL = `lmms-lab/VQAv2` validation (cc-by-4.0, clean parquet). TRAIN = `HuggingFaceM4/VQAv2` (only train mirror with 10 annotators; **FLAG: license undeclared on hub** — COCO/VQA upstream cc-by-4.0; needs `trust_remote_code=True`; loading-script, Dataset Viewer 501, fragile). DEMO = `merve/vqav2-small` (3 cols, no annotators → demo only, **FLAG license**) or `Multimodal-Fatima/VQAv2_sample_train` (1K rows, FULL 10-annotator schema, offline-friendly, **FLAG license**).
- **`data/download`** — vocab + dataset fetch helpers. The **answer vocabulary (3129)** comes from the model config, not a dataset: `AutoConfig.from_pretrained('dandelin/vilt-b32-finetuned-vqa').id2label` (len 3129); answer list = `list(config.id2label.values())`.

### `models/vqa_model`
**NEW** — the multimodal VQA model wrapper exposing a uniform `predict(image, question) → {topk:[(ans,prob)], p_max, entropy, margin, regime}` over: `ViltForQuestionAnswering` (classification, DEFAULT) via `ViltProcessor`; `BlipForQuestionAnswering` / `microsoft/git-base-vqav2` (generative alternative) via `.generate()` with logprob-as-confidence; and `SceneStubVQA` (offline, no torch). A capability probe selects real-vs-stub at runtime — **same `predict()` signature**, so swapping the stub for a real ViLT wrapper flips the system to production with NO agent changes. Documented upgrades: H100 tier `Salesforce/blip2-flan-t5-xl` (mit, ~3.9B, encoder-decoder → clean constrained decoding that fits the agent's type constraints) / `Salesforce/blip2-opt-2.7b` (mit); modern Apache VLM `Qwen/Qwen2-VL-2B-Instruct` (apache). **AVOID / FLAG:** `google/pix2struct-vqav2-base` does NOT exist (404; pix2struct only ships doc/infographic variants); `llava-hf/llava-1.5-7b-hf` (llama2, non-commercial-ish — FLAG); `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen Research, empty HF license — FLAG).

### `models/question_type`
**NEW** — the question-type keyword classifier (the D2 router). Lowercase + strip, take leading tokens, evaluate rules **top-down, first match wins** (order is critical: 'how many' must beat 'how'; 'what color' must beat generic 'what'). The 10-rule reference table — yes_no / number / color / where / who / why / when / which / what-object / default-other — maps onto the agent's 4 coarse buckets and onto the VQAv2 65 question-type buckets for reporting. Pure regex/keyword; no torch.

### `models/baseline`
**NEW** — the bias-ablation baselines (report all): (1) **prior 'yes'** — always answer "yes"; (2) **most-common-answer per-type prior** — overall 'yes', count→'2', color→'white'; (3) **blind / question-only** — classify from the QUESTION text alone, NO image (measures dataset language-prior bias; the full-model gap over it is the real signal the image is used); (4) **zero-shot pretrained ViLT** — `dandelin/vilt-b32-finetuned-vqa` with no fine-tuning on the target split. These quantify the language-prior shortcut that VQA models are prone to.

### `models/model_registry`
Registry mapping role → verified HF id + license + class/processor (populated from the §3 model tier table). Single lookup for "DEFAULT trainable core", "generative alternative", "H100 upgrade", flagging non-commercial / undeclared licenses at lookup time. Template reused from siblings.

### `vision/image_utils`
**NEW** — the image-handling layer: RGB decode/convert, resize/normalize delegated to the model's image processor, and **input validation** (degenerate-blank detection: pixel-variance > eps) that drives **D1**. Also defines **`SceneImage`** — a no-PIL carrier object that holds an image path + the parsed `scene_spec` so the offline path can pass "an image" through the agent without importing PIL/torch at all (the stub reads the spec straight off the carrier). Pillow-only when PIL is present; pure-stdlib via `SceneImage` when it is not.

### `training`
HF `Trainer` harness to fine-tune the **one** trainable stage — the VQA core. Default: fine-tune `ViltForQuestionAnswering` (classification head over 3129 answers) from `dandelin/vilt-b32-finetuned-vqa` or pretrain-base `dandelin/vilt-b32-mlm`. GPU tiers: **T4** fine-tunes ViLT (batch 16–32 @384px); **A100/L4** full speed; **H100** for BLIP-2 / Qwen via LoRA + bf16. Corpus loading via `data`; checkpoints registered through `model_registry`. Template reused from P14/P15.

### `agent`
**NEW** — the mandatory agentic component: the deterministic FSM of §4 in `src/mmqa/agent/` with states `ingest → classify → answer → calibrate → constrain` and the 5 decision points. `agent/calibrate` implements D4 (temperature-scaled `p_max` + entropy + margin gate); `agent/constrain` implements D5 (type-consistency check + re-rank within top-k). Optional `anthropic` LLM brain OFF by default, advisory only, never overrides. Runs fully offline on `SceneStubVQA` + the keyword classifier.

### `api`
**FastAPI** service: `POST /ask` (image upload + question → answer + confidence + abstain flag; route gated on `python-multipart`) and `POST /ask-scene` (JSON for the synthetic `SceneImage` path, torch-free). A **Gradio** UI (upload an image + type a question) is mounted for interactive demo. Packaged with **Docker** (needs `libGL` for Pillow) and an HF Space. Request/response wrappers around `agent.predict()`.

### `analysis`
**NEW metric core** — the **official VQA accuracy** (soft) plus the bias-ablation report. Because **there is no `evaluate-metric/vqa` space on HF Hub**, the canonical `VQAEval` is re-implemented (~40 lines): answer normalization (lowercase; strip punctuation but keep decimal periods and merge digit-group commas; number-word→digit map; drop articles a/an/the; ~80-entry contractions dict) applied IDENTICALLY to prediction and all 10 GT answers, then the **10× leave-one-out** average of `min(1, matches/3)` per question, ×100. Reports overall + **per-answer-type** (yes/no, number, other) + **per-question-type** (65 buckets) accuracy, plus **abstention rate / coverage** and re-rank rate. `evaluate-metric/exact_match` and `evaluate-metric/accuracy` are used only as auxiliary sanity checks, never as the VQA soft accuracy.

### `autoreport`
Auto-generates the run report (config + overall/per-answer-type/per-question-type accuracy + baselines + abstention rate + re-rank rate + sample scenes) from a single command. Template reused from P14/P15.

### `monitoring`
Run-time signal capture: per-stage timings/latency, decision-point branch counts (how many items hit ok / ok:reranked / abstained:low_confidence / abstained:type_mismatch / error:*), confidence (`p_max` / entropy / margin) distributions, and abstention/coverage tracking. Template reused from siblings.

### `automation`
Autopilot driver that chains generate-scenes → run-agent → score → report end-to-end for reproducible benchmark runs, online or offline. Reused template.

### `grading`
Self-grading harness that scores a run against the project rubric (metrics present, baselines beaten, offline path green, all 5 decision branches exercised) for the assignment deliverable. Wraps the `analysis` VQA-accuracy implementation. Reused template.

---

## 6. Offline & degradation design

The defining engineering property of P17 is that **the entire 5-point agent and the scorer run with zero torch and no network** — deterministic and reproducible from a seed, mirroring P15's `SeedEngine`. Tests pass in CI / Colab-free mode with nothing downloaded. Four mechanisms make this work.

### 6.1 Lazy imports + capability probes (one code path)
Heavy dependencies (`torch`, `transformers`, `PIL`, `anthropic`) are imported lazily inside the functions that need them, never at module top level. Each stage runs a capability probe (`try import`) and selects the real component when present, the stub when absent. The env flag `MMQA_OFFLINE=1` pins stub mode for reproducible tests. Crucially, **the probe upgrades each stage in place; the surrounding code and the tests are identical online and offline.** Swapping `SceneStubVQA` for the real `ViltForQuestionAnswering` wrapper (same `predict()` returning topk+probs) flips the system to production with **no agent changes**.

### 6.2 SceneStubVQA (offline VQA model, no torch / no network)
The offline core, mirroring P15's OCR `SeedEngine`. Same `predict(image, question)` signature as the real wrapper. It (a) reads the spec back out of the PNG metadata — `json.loads(Image.open(path).text['scene_spec'])`, or straight off a `SceneImage` carrier; (b) computes the TRUE answer from spec + question via the same template logic as the generator; (c) returns a **realistic distribution, not one-hot** — ~0.7–0.9 mass on the correct answer, the remainder spread over plausible type-consistent distractors (other colors, adjacent counts), so `p_max` / entropy / margin at D3/D4 are meaningful and the abstention gate is actually exercised; (d) a `noise` / `difficulty` knob deliberately lowers `p_max` or injects a type-wrong top1 on a fraction of items so **D4 (abstain) and D5 (re-rank) trigger** on the eval set and can be unit-tested. Verified offline: stub answers seed scenes correctly and all 5 decisions fire.

### 6.3 SceneImage no-PIL carrier
For the strictest offline path, `vision/image_utils.SceneImage` carries an image path plus the parsed `scene_spec` so the agent can pass "an image" end-to-end **without importing PIL or torch at all** — the D1 validity check reads the carrier's metadata, and `SceneStubVQA` reads the spec straight off the carrier. PIL is upgraded in only when actually decoding pixels (real images), keeping the synthetic + CI path stdlib-only.

### 6.4 Keyword classifier + pure-Python metric
- **Question-type classifier** (D2) is pure regex/keyword (`models/question_type`) — no model, deterministic, instant, the same code online and offline.
- **VQA-accuracy metric** (`analysis`) is a pure-Python re-implementation of `VQAEval` (normalization + 10× leave-one-out soft accuracy + per-type aggregation), since no HF `evaluate-metric/vqa` exists — fully meaningful offline because it scores the deterministic stub answers against the gold 10-answer lists built by the generator.

**Net:** with the standard library plus an optional Pillow, `scene → SceneStubVQA → agent(D1..D5) → VQA-accuracy + per-type + abstention/re-rank report` runs deterministically with no torch and no network, yielding a real accuracy / abstention-rate / re-rank-rate report in CI and Colab-free mode. The agent branch coverage and the metric are fully meaningful offline because they measure the deterministic stub distribution and the gold scene spec, not real-model quality.

---

## 7. Reuse map

- **From P14 doctrans / P15 imgtrans:** the config / logging / registry / autoreport / monitoring / automation / grading / cli / api templates; the lazy-import + capability-probe pattern; and the offline-stub pattern (P15's OCR `SeedEngine` → P17's `SceneStubVQA`).
- **New for P17:** the multimodal VQA model wrapper (`models/vqa_model`: ViLT classification + BLIP generative + `SceneStubVQA`); image handling (`vision/image_utils` + the `SceneImage` no-PIL carrier); the synthetic-scene generator (`data/synth_scene.py`); the official VQA-accuracy metric + answer normalization (`analysis`); the question-type keyword classifier (`models/question_type`); and the type-aware + abstaining 5-decision agent (`agent`).

---

## 8. Ethics, privacy & robustness (architectural commitments)

VQA over user photos is sensitive — faces, homes, documents, medical images, and assistive use for blind users (the VizWiz setting). The architecture therefore: **processes locally by default**, **retains no raw image** by default, treats the tool as an **assistant that abstains/flags low-confidence answers** (D4/D5 → `unsure` / `needs_review`) and **never asserts certainty** (especially for accessibility / medical), and requires consent for any retention.

Bias is engineered against, not assumed away: VQA models carry strong **language priors** — answering "yes" / "2" / "white" while ignoring the image — plus demographic bias in COCO. The architecture surfaces this by always reporting the **blind question-only baseline** (`models/baseline`) and **per-answer-type accuracy**; the full-model gap over the blind prior is the honest evidence the image is actually used. Robustness is exercised by design: overconfidence (mitigated by the D4 calibrated gate, because raw softmax is poorly calibrated and skews high), the language-prior shortcut, out-of-vocab answers (the 3129-label classification ceiling — generative models avoid it but trade away the clean confidence signal), unanswerable questions (the `lmms-lab/VizWiz-VQA` explicit-unanswerable label is the ideal abstention test — **FLAG license**), image quality/blur (the D1 input gate), and adversarial questions. License hygiene is structural: every dataset/model id flagged non-commercial or undeclared (`HuggingFaceM4/VQAv2`, `merve/vqav2-small`, `Multimodal-Fatima/*`, VizWiz, Qwen2.5-VL, LLaVA) is FLAGGED at registry lookup, and the declared-clean sets (`lmms-lab/VQAv2` cc-by-4.0, `facebook/textvqa` cc-by-4.0, `lmms-lab/GQA` mit, `dandelin/vilt-b32-finetuned-vqa` apache-2.0) are preferred for anything that must be commercially defensible.
