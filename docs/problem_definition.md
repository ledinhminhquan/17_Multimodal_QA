# P17 — Problem Definition: Multimodal Question Answering (VQA)

> Project: **P17 Multimodal Question Answering** (Visual Question Answering, VQA) · Package `mmqa` · Folder `17_Multimodal_QA`
> Author: Le Dinh Minh Quan (student 23127460)
> Status: first **multimodal-vision** project in the series (P02–P15 were text / document / OCR; P16 done). Everything before P17 reasoned over text or rendered text; P17 reasons jointly over **pixels and language**.

---

## 1. What problem are we solving?

**Visual Question Answering (VQA)** is the task of answering a free-form natural-language question about the content of an image. The system is given an `(image, question)` pair and must return a short natural-language `answer`.

```
INPUT                                              OUTPUT
┌─────────────────────┐
│  [an image]         │  +  "how many red squares?"   ──►   "2"
│  [a kitchen photo]  │  +  "what color is the table?"  ──►   "white"
│  [a street scene]   │  +  "is there a stop sign?"     ──►   "yes"
└─────────────────────┘
```

The answer is **grounded in the image**: the same question over two different images should yield two different answers, and the same image with two different questions yields two different answers. A system that answers "yellow" to "what color is the banana?" *without looking at the image* has not solved VQA — it has merely learned a language prior. Beating that prior is the heart of the problem (see §6 and the blind baseline in §7).

Concretely for P17, an example offline item produced by the synthetic-scene generator is: an image with two red squares and one blue circle on a white canvas, plus the question "how many red squares?", with the gold answer "2". The model must read the visual scene, the agent must recognise this as a **count** question, and the final answer must be a number — not a fluent-but-type-wrong token like "cat".

### Why this is hard
- **Two modalities, one answer.** The model must align image regions/patches with question tokens and fuse them before producing an answer. This is harder than either image classification or text QA alone.
- **Open-ended questions.** "How many", "what color", "is there", "where", "why" — each implies a different *kind* of answer (a number, a color word, yes/no, a place, a reason).
- **Models are overconfident and prior-driven.** VQA models routinely emit a confident answer that ignores the image, especially on questions with a strong dataset prior. Producing a *calibrated* answer — including the option to say "unsure" — is as important as raw accuracy.

---

## 2. The trainable core vs. the deterministic wrapper

P17 cleanly separates **what is learned** from **what is engineered**:

- **Trainable core = the VQA model.** A multimodal transformer that consumes image patches + question tokens and emits an answer. This is the single trainable artifact. By default it is a **classification** model over the canonical 3129-answer VQAv2 label space (`dandelin/vilt-b32-finetuned-vqa`, Apache-2.0).
- **Deterministic agent = everything else.** Image decode/resize, question tokenization, answer normalization, the question-type classifier, the abstention logic, the type constraint, and scoring are all **pretrained or deterministic** and are **not trained**. The agent is a 5-decision finite state machine that wraps the frozen model (see §3 and §7).

This split is what makes the project both a *machine-learning* deliverable (train and evaluate a real VQA model) and a *production-engineering* deliverable (a reliable, auditable, offline-testable agent around it).

---

## 3. Inputs and outputs (the contract)

### Inputs
| Field | Type | Constraints (enforced at the agent's D1 input gate) |
|---|---|---|
| `image` | An RGB raster image (PIL / uploaded file / array) | Must decode to a valid non-zero RGB array; must not be a degenerate blank (pixel-variance above a small epsilon). COCO-style natural scenes are the target distribution. |
| `question` | A natural-language string | Non-empty after strip; ≥1 alphabetic token; length in `[1, 128]` tokens; interrogative (ends in `?` **or** starts with a wh-/auxiliary word: what/which/how/where/is/are/does/do/can). |

### Outputs
The agent returns a structured result, not a bare string:

| Field | Meaning |
|---|---|
| `answer` | The short answer string (e.g. `"2"`, `"white"`, `"yes"`), or `"unsure"` when the agent abstains. |
| `status` | One of `ok`, `ok:reranked`, `abstained:low_confidence`, `abstained:type_mismatch`, `error:bad_image`, `error:bad_question`, `error:model_failure`. |
| `question_type` | The classified type (`yes_no` / `count` / `color` / `object/other`), which drives the answer constraint. |
| `confidence` | The model's calibrated confidence (max softmax probability for the classification path; sequence log-prob for the generative path). |
| `topk` | The top-k `(answer, prob)` candidates, for auditability and re-ranking. |

The key design point: the output is **honest about uncertainty**. A raw model emits an argmax; P17 emits an answer *with* a status that can say "abstained" or "reranked", so a downstream consumer (or a human reviewer) knows when to trust it.

---

## 4. Real-world use cases

VQA is the engine behind several production scenarios, each with its own stakes:

- **Visual assistance for blind and low-vision users.** A user photographs their surroundings and asks "what is in front of me?" / "what color is this shirt?" / "is the stove on?". This is the canonical high-stakes use case (the VizWiz benchmark is built from exactly these real blind-user photos). Here a *wrong but confident* answer can mislead someone who cannot verify it — so **abstention and calibrated confidence are safety features, not nice-to-haves**.
- **E-commerce product Q&A.** "Does this jacket have a hood?", "what color options are shown?", "how many are in the pack?" answered directly from product imagery, reducing returns and support load.
- **Image search and retrieval.** Letting users query a photo library in natural language ("photos where there is a dog on a beach") by answering structured questions over each image.
- **Education and accessibility.** Generating or checking simple comprehension questions over diagrams and scene images; helping learners interrogate a picture.
- **Content moderation triage.** A fast first-pass "is there X in this image?" gate that flags items for human review — where the **abstain / needs-review** path is exactly the right behaviour for ambiguous cases rather than a forced yes/no.

Across all of these, the common thread that P17 emphasises: the tool **assists and abstains**, it never asserts certainty it does not have. That is why the agent's abstention gate is the headline value-add (see §6).

---

## 5. Scope and non-goals

### In scope
- **Natural-image VQA** over COCO-style scenes (the VQAv2 distribution).
- The **three VQAv2 answer types**: `yes/no`, `number`, `other`.
- The **agentic wrapper**: input gate → question-type router → model → calibrated abstention → type-consistency constraint.
- An **offline synthetic test harness** (the synthetic-scene generator + a torch-free stub model) so the full agent and scorer run with no network and no model download.
- Both **classification** (default) and **generative** (alternative) answer regimes — see §8.

### Out of scope (explicit non-goals)
- **Document / infographic / OCR VQA as the primary target.** TextVQA is only a *side* benchmark; reading dense rendered text inside images is not the main objective. (P15 already covered the OCR/document axis.)
- **Multi-turn visual dialog.** Each `(image, question)` is answered independently; there is no conversational memory.
- **Region grounding / bounding-box outputs.** The system outputs a short text answer, not coordinates or segmentation masks.
- **Open-world knowledge QA** ("who painted this in 1889?") as the main loop — knowledge-VQA (OK-VQA) is only a side abstention test.

---

## 6. Classification VQA vs. generative VQA

P17 supports both answer regimes and **defaults to classification**. Understanding the trade-off is part of the problem definition because it determines the confidence signal the agent relies on.

| | **Classification VQA (DEFAULT)** | **Generative VQA (alternative)** |
|---|---|---|
| Answer space | Fixed 3129-answer VQAv2 vocabulary | Open vocabulary (free-form text) |
| Mechanism | Linear head → logits → softmax over the vocab → top-k | Autoregressive decoding of the answer string |
| Confidence signal | Clean per-answer **softmax distribution** (p_max, entropy, top1–top2 margin) | Sequence / token **log-probability** (noisier proxy) |
| Default model | `dandelin/vilt-b32-finetuned-vqa` (ViLT, ~113M, Apache-2.0) | `Salesforce/blip-vqa-base` (BLIP, ~385M, BSD-3) |
| Evaluation | Direct label lookup → clean metric | Decode + canonical normalization + exact-match (harder) |
| Ceiling | Any gold answer outside the 3129 vocab is **unreachable** | No fixed-vocab ceiling |

**Why classification is the default.** The softmax distribution over a fixed vocabulary is the *cleanest input for a calibrated abstention gate*: it gives a real probability, a real entropy, and a real top1–top2 margin that the agent thresholds on. Generative models avoid the 3129-answer ceiling but trade away that clean confidence signal (they expose only log-probs). The agent (§7) is designed against the classification regime; the generative path is supported as a complement and as an upper-bound comparison, and exists on the same `predict(image, question) → {topk, p_max, entropy, margin, regime}` interface.

---

## 7. The agentic component (where the engineering value lives)

A stock ViLT/BLIP model only does raw argmax. P17 wraps the frozen model in a **deterministic 5-decision finite state machine** (`src/mmqa/agent/`) that adds two things production needs, with **no extra training**:

1. **Calibrated abstention.** VQA models are notoriously overconfident, so thresholds are *calibrated* (temperature-scaled) on a held-out split, not hard-coded. The gate combines max-prob, entropy, and top1–top2 margin; if the model is not confident enough, the answer becomes `"unsure"` with a `needs_review` flag instead of a confident wrong answer.
2. **Type-aware answer constraint + re-rank.** The question is classified (yes/no / count / color / object-other), and the answer is re-ranked within the top-k to the best **type-consistent** candidate. This fixes the classic failure where the argmax is fluent-but-type-wrong (answering `"cat"` to "how many?").

The five decision points:

| ID | Decision | Effect |
|---|---|---|
| **D1** | Input gate | Validate image + question *before* any model call; halt on bad image / bad question. |
| **D2** | Question-type router | Keyword/regex map to `{yes_no, count, color, object/other}`; defines the allowed answer set. |
| **D3** | Run VQA | The only step that touches the model; produces top-k, p_max, entropy, margin. |
| **D4** | Calibrated abstention gate | Low confidence / high entropy / small margin → abstain `"unsure"`. |
| **D5** | Type-consistency gate + re-rank | Force the answer to match the question type; re-rank within top-k or abstain on type mismatch. |

The agent runs **fully offline** using a synthetic-scene generator (which embeds a `scene_spec` in PNG metadata) and a `SceneStubVQA` model that reads that spec to answer — mirroring P15's OCR `SeedEngine`. This means the entire 5-decision pipeline, the metric, and the tests run with **no torch, no model, and no network**, deterministically from a seed. Swapping the stub for a real `ViltForQuestionAnswering` wrapper (same `predict()` returning top-k + probs) flips the system to production with **no agent changes**.

---

## 8. Success criteria

P17 is judged on two axes — **accuracy** and **abstention safety** — not accuracy alone.

### 8.1 Accuracy (the official VQA soft metric)
The headline metric is the **official VQA accuracy** (Antol et al. 2015; Goyal et al. 2017), robust to inter-annotator disagreement:

```
acc(answer) = min( 1, (# of the 10 human annotators that gave this answer) / 3 )
```

implemented as the canonical 10× leave-one-out average, after applying the single canonical answer-normalization function (lowercase; strip/standardize punctuation without splitting decimals; number-word → digit; drop articles a/an/the; contraction mapping) **identically** to the prediction and all 10 ground-truth answers. (There is no `evaluate-metric/vqa` on the HF Hub — the metric is re-implemented from the reference `VQAEval`.)

Reporting requirements:
- **Overall accuracy** plus the three **answer-type** buckets `{yes/no, number, other}` (×100).
- **Per-question-type** accuracy (the finer ~65-bucket breakdown) as a table/appendix.
- The result must **beat the baselines**: the most-common-answer prior ("yes"), the **blind / question-only** prior (the language-bias baseline — the gap above it is the real signal that the image is used), and zero-shot pretrained ViLT.

### 8.2 Abstention safety (the agentic headline)
Because the value-add is calibrated abstention, P17 also reports:
- **Answer-when-confident accuracy** (accuracy on the items the agent chose to answer) vs. a **raw-argmax baseline** — the abstention gate should trade a little coverage for a meaningfully higher precision.
- **Abstention rate / coverage** and **re-rank rate** from D4/D5.
- The system must demonstrably say **"unsure" instead of hallucinating** on low-confidence / unanswerable items (VizWiz-style unanswerable questions are the ideal abstention probe).

Put plainly: a successful P17 is **accurate when it answers, and quiet when it should be** — the constrained, abstaining agent must beat the raw model on a reliability-adjusted measure.

---

## 9. Ethics, privacy, and robustness framing

Because VQA runs on **user photos** (faces, homes, documents, medical images, and assistive use for blind users), the problem is framed with explicit safeguards:

- **Privacy:** consent, local/on-device processing where possible, and **no raw-image retention by default**.
- **Safety:** the tool **assists and abstains/flags** low-confidence answers; it never asserts certainty — especially for accessibility and medical-adjacent use.
- **Bias:** VQA models carry strong **language priors** (answering "yes" / "2" / "white" while ignoring the image) and demographic bias inherited from COCO. P17 reports the **blind-prior baseline** and **per-type accuracy** so this bias is measured, not hidden.
- **Robustness concerns** treated as first-class: overconfidence, the language-prior shortcut, out-of-vocabulary answers (the 3129-answer ceiling), unanswerable questions, image quality/blur, and adversarial questions.

---

## 10. Assignment mapping

| Assignment requirement | How P17 satisfies it |
|---|---|
| A trainable ML core | The **VQA model** — a multimodal transformer fine-tuned/evaluated as a 3129-way classifier (`dandelin/vilt-b32-finetuned-vqa`, Apache-2.0), with a generative alternative (`Salesforce/blip-vqa-base`). |
| A real, verified dataset | **Train** `HuggingFaceM4/VQAv2` (only common mirror with a train split + full 10-annotator schema; license undeclared-on-hub → **FLAG**, COCO/VQA upstream CC-BY-4.0). **Eval** `lmms-lab/VQAv2` validation (CC-BY-4.0, clean). **Demo/offline** `Multimodal-Fatima/VQAv2_sample_train` (1K rows, full schema → **FLAG**) and the synthetic generator. **Vocab** from the ViLT config (3129). |
| A principled evaluation metric | The **official VQA soft accuracy** (10-annotator leave-one-out) with per-answer-type and per-question-type breakdowns and a full baseline table. |
| A mandatory agentic component | A **deterministic 5-decision FSM** (input gate → type router → model → calibrated abstention → type-consistency re-rank) that is auditable and offline-testable. |
| A demonstrable value-add over the bare model | **Type-aware answer constraint + calibrated abstention** — higher constrained accuracy and a principled refuse-to-answer, beating raw argmax with no extra training. |
| Reproducibility / offline CI | A **synthetic-scene generator + torch-free `SceneStubVQA`** that runs the whole agent, metric, and tests with no network and no model download, deterministically from a seed (mirroring P15's `SeedEngine`). |
| Deployment | FastAPI (`POST /ask` image upload + question; `POST /ask-scene` JSON for the synthetic path), a Gradio UI, Docker (with libGL for Pillow), and a HF Space. |
| Ethics & responsible AI | Consent, no raw-image retention, abstain-not-assert on low confidence, and explicit language-prior / demographic-bias reporting via the blind baseline and per-type metrics. |

---

### One-sentence definition
> **P17 takes an image and a natural-language question about it and returns a short, type-consistent answer — or honestly abstains when the underlying VQA model is not confident enough — judged by the official VQA soft accuracy and by its abstention safety relative to a raw-argmax baseline.**
