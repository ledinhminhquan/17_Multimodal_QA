# P17 Multimodal Question Answering (VQA) — Data Card

> Package `mmqa` · Folder `17_Multimodal_QA` · Author: Le Dinh Minh Quan (student 23127460)
> Task: given an **image** + a natural-language **question** about it, produce a short **answer** (e.g. a scene + "how many red squares?" → "2").
> This is the FIRST multimodal-vision project in the series (P02–P16 were text / document / OCR).

This card documents every dataset the project consumes, the **synthetic scene generator** that lets the agent, the metric, and the tests run with no torch / no model / no network, and the intended and out-of-scope uses of all of them. It is the single source of truth for **what data is used, where it came from, what license it carries, and what biases it encodes**.

A guiding rule for this card: **any Hugging Face repo with no declared license is treated as "license unconfirmed" and FLAGGED**, even when the upstream source (COCO images + VQA annotations, both CC-BY-4.0) is commercially usable. The upstream provenance does not launder an undeclared mirror.

---

## 1. At-a-glance summary

| id | Role in P17 | License | 10-annotator schema? | Commercial use |
|---|---|---|---|---|
| `lmms-lab/VQAv2` (validation) | **PRIMARY EVAL** | cc-by-4.0 (clean) | Yes | Safe (declared) |
| `HuggingFaceM4/VQAv2` (train) | **PRIMARY TRAIN** | undeclared — **FLAG** | Yes | Verify first |
| `Multimodal-Fatima/VQAv2_sample_train` | Tiny train, full schema, offline-friendly | undeclared — **FLAG** | Yes | Verify first |
| `merve/vqav2-small` | DEMO / smoke loop only | undeclared — **FLAG** | Verify first | Verify first |
| `dandelin/vilt-b32-finetuned-vqa` (config) | **ANSWER-VOCAB SOURCE** (3129 labels) | apache-2.0 (clean) | n/a (model) | Safe (declared) |
| **Synthetic scene generator** (`data/synth_scene.py`) | **PRIMARY OFFLINE DATA + CI** | Project-owned (this repo) | Yes (synthesized) | Safe (we own it) |
| `lmms-lab/GQA` | Side benchmark (compositional) | mit (clean) | No (different schema) | Safe (declared) |
| `facebook/textvqa` | Side benchmark (text-in-image) | cc-by-4.0 (clean) | partial | Safe (declared) |
| `lmms-lab/VizWiz-VQA` | Side benchmark — abstention test | undeclared — **FLAG** | has `unanswerable` label | Verify first |
| `lmms-lab/OK-VQA` | Side benchmark — knowledge VQA | undeclared — **FLAG** | Yes | Verify first |
| `Luo-wj/DAQUAR` | Side benchmark — classic indoor VQA | undeclared — **FLAG** | small | Verify first |

> **Two repos are deliberately NOT used:** `thangduong0509/daquar_vqa` (**cc-by-nc-sa-4.0 — NON-COMMERCIAL**, excluded outright) and `jp1924/VisualQuestionAnswering` (gated + Korean, not VQAv2). See §8.

---

## 2. Shared lineage: VQAv2 and COCO

All of the core datasets in this project are mirrors or subsets of **VQA v2.0** (Goyal et al. 2017, "Making the V in VQA Matter"), which is itself built on **VQA v1.0** (Antol et al. 2015). Understanding the shared lineage once means the per-dataset sections below only have to state what differs.

**Images — COCO 2014/2015.** Every natural-image question in VQAv2 is asked about an image from Microsoft COCO (Common Objects in Context). COCO images are everyday photographs of scenes containing common objects (people, animals, furniture, food, vehicles, sports). COCO images are licensed **CC-BY-4.0**.

**Annotations — VQA project.** For each image, human workers wrote open-ended questions, and for **each question, 10 independent annotators** each supplied a short free-text answer. VQAv2 annotations are licensed **CC-BY-4.0** upstream. The 10-answer structure is the heart of the official soft-accuracy metric (§5) — it is what lets the metric tolerate legitimate human disagreement ("teal" vs "blue").

**What VQAv2 fixed vs VQAv1 (the balancing).** VQAv1 had a severe **language-prior** problem: a model could answer "what color is the banana?" → "yellow" or "is it sunny?" → "yes" from the *question text alone*, ignoring the image, and still score well. VQAv2 deliberately constructed **complementary image pairs**: for many questions it added a second, similar image for which the *same question* has a *different* answer. This depresses the blind/question-only baseline and forces models to actually use the image. **It depresses the language prior but does not eliminate it** — see §6 (biases). The project measures the residual prior directly via the blind baseline (§5).

**The 3 answer types.** Every VQAv2 question is tagged `answer_type ∈ {yes/no, number, other}`. This is the headline reporting breakdown (§5). Typical difficulty ordering: yes/no easiest (~80-90%), `other` mid, `number` hardest.

**The 65 question types.** Every question also carries a `question_type` — the canonical leading phrase ("how many", "what color is the", "is the", "are there", "what is the", ...). 65 distinct buckets; reported as a finer appendix table.

---

## 3. Core datasets (full detail)

### 3.1 `lmms-lab/VQAv2` — PRIMARY EVAL

| Field | Value |
|---|---|
| **Role** | Canonical evaluation benchmark. The headline accuracy numbers come from here. |
| **License** | **cc-by-4.0 (declared, clean)** — commercially safe. |
| **Format** | Parquet, embedded images. Clean, no loading script, no `trust_remote_code`. |
| **Size** | 769.5K rows total: **validation 214.4K**, testdev 107.4K, test 447.8K. **No train split.** |
| **Provenance** | COCO 2014/2015 images (CC-BY-4.0) + VQAv2 annotations (CC-BY-4.0). |

**Schema (8 columns) — verified via preview:**

| Column | Type | Meaning |
|---|---|---|
| `question` | string | The natural-language question. |
| `image` | PIL image | The COCO scene the question is about. |
| `answers` | list of 10 dicts | **The 10-annotator block.** Each dict = `{answer: str, answer_confidence ∈ {yes, maybe, no}, answer_id ∈ 1..10}`. |
| `multiple_choice_answer` | string | The **consensus** answer (the most agreed single answer). |
| `question_type` | string | One of the 65 canonical leading phrases. |
| `answer_type` | string | One of `{yes/no, number, other}`. |
| `image_id` | int | COCO image id. |
| `question_id` | int | Unique question id. |

**Why this is the eval set.** It is the only common mirror that is (a) declared CC-BY-4.0 clean, (b) clean parquet with no fragile loading script, and (c) carries the **full 10-annotator `answers` block** required for the official soft VQA accuracy (§5). Per-answer-type and per-question-type aggregation are both directly computable from its `answer_type` / `question_type` columns. For the validation split the ground truth is present, so accuracy is computed locally; for testdev / test the GT is held out (EvalAI server) — the project evaluates on **validation**.

---

### 3.2 `HuggingFaceM4/VQAv2` — PRIMARY TRAIN

| Field | Value |
|---|---|
| **Role** | Fine-tuning the trainable core (the VQA model) on VQAv2 train. The only common mirror that has **both** a train split **and** the full 10-annotator schema. |
| **License** | **Undeclared on the repo — FLAG.** Upstream COCO images + VQA annotations are CC-BY-4.0, but the mirror itself declares nothing. Treat as "verify before commercial use." |
| **Size** | train ~443K + val ~214K + test. |
| **Schema** | Full VQAv2 fields including the 10-annotator `answers` list (same structure as §3.1). |

**Loading caveats (this dataset is fragile — read before use):**

- It is a **loading-script dataset**, not clean parquet. Requires `load_dataset(..., trust_remote_code=True)`.
- The **HF Dataset Viewer returns 501** for it (cannot preview online).
- It may **pull COCO images by URL** at load time → network-dependent, can break if upstream URLs move.
- It is **fragile on newer `datasets` versions**. **Pin the `datasets` version** when you use it.

**Mitigation / scoping.** Use this dataset **strictly for the train split**. Evaluate on the clean `lmms-lab/VQAv2` parquet (§3.1), never on this. For any offline / CI / torch-free run, do not touch it at all — use the synthetic generator (§4).

---

### 3.3 `Multimodal-Fatima/VQAv2_sample_train` — tiny train, full schema, offline-friendly

| Field | Value |
|---|---|
| **Role** | The **best small set that keeps the 10-annotator structure** for proper soft scoring without the fragility of §3.2. Used for the offline-friendly demo path where real human scoring is wanted, and for quick fine-tune smoke loops. Works offline once cached. |
| **License** | **Undeclared — FLAG** (CC-BY-4.0 upstream). |
| **Size** | train **1.0K rows** (~155 MB), embedded images. |
| **Schema** | 13 columns incl. `answers_original` (the **10 annotator dicts**), `question_type`, `answer_type`, `question_id`, `image` (embedded), plus enrichment extras: `blip_caption`, `clip_tags`, `DETA_detections`. |

**Note on the extras.** `blip_caption`, `clip_tags`, and `DETA_detections` are **model-generated** auxiliary annotations (a BLIP caption, CLIP tags, a DETA object-detector's boxes). They are convenient but are *not* gold human labels and inherit the biases of the models that produced them — do not treat them as ground truth and do not score against them.

---

### 3.4 `merve/vqav2-small` — DEMO ONLY

| Field | Value |
|---|---|
| **Role** | Fast Colab smoke loop / quick demo of the inference path. **DEMO ONLY.** |
| **License** | **Undeclared — FLAG.** |
| **Size** | validation 21.4K rows, parquet, embedded images. |
| **Schema** | **Only 3 columns:** `image`, `question`, `multiple_choice_answer`. |

**Hard limitation — why it is demo-only.** It has **no `answers` block (no 10 annotators)** and **no `question_type` / `answer_type`**. Therefore it **cannot be used for the official soft VQA accuracy** (which requires the 10 annotators) and **cannot drive per-type routing or reporting**. It is suitable only for eyeballing predictions / a quick smoke loop. Any number computed against it is not the official metric and must not be reported as such.

---

### 3.5 `dandelin/vilt-b32-finetuned-vqa` (config) — ANSWER-VOCAB SOURCE

| Field | Value |
|---|---|
| **Role** | **Source of the canonical 3129-answer VQAv2 label space.** This is a *model*, used here as the authoritative answer vocabulary, not as a dataset of examples. |
| **License** | **apache-2.0 (clean).** |
| **How to load the vocab** | `AutoConfig.from_pretrained('dandelin/vilt-b32-finetuned-vqa').id2label` → length **3129**; the answer list = `list(config.id2label.values())`. |

**Why it matters as "data."** The classification VQA core has a **fixed 3129-way head**. That vocabulary is the closed answer space of the whole classification pipeline: it defines what answers are even *reachable*, what the metric's candidate space is, and what the agent's type-constraint sets (§7 of the design brief) are drawn from. **Any gold answer outside these 3129 is unreachable by the classification model — a hard accuracy ceiling.** If a custom vocabulary is ever trained (from `dandelin/vilt-b32-mlm`), the metric buckets, the answer-constraint sets, and the report must all use that *same* vocabulary. See §6 (answer-vocab mismatch).

---

## 4. The synthetic scene generator — PRIMARY OFFLINE DATA

`data/synth_scene.py` is a **project-owned, fully deterministic** dataset generator. It is the *primary* data path for offline/CI runs and for development without a GPU. It mirrors P15's OCR `SeedEngine` pattern: the entire 5-decision agent, the metric, and the test suite run from synthetic data with **no torch, no model download, and no network**.

It is the single most-used data source during development, because it is the only one that is free, instant, license-clean (we own it), and reproducible bit-for-bit from a seed.

### 4.1 What it produces

For a seed, `make_scene(seed)` (PIL only) draws colored shapes on a small white canvas (e.g. 224×224) and emits items that **match the `lmms-lab/VQAv2` 8-column schema** so the metric and agent are exercised exactly as on real data:

`{PIL image, question, answers (list of 10), multiple_choice_answer, question_type, answer_type}`.

**Coverage — the shape/color lexicon:**

- **Shapes:** `{square, circle, triangle}` (drawn as rectangle / ellipse / polygon via `ImageDraw`).
- **Colors:** a fixed 6-color lexicon `{red, green, blue, yellow, orange, purple}`.
- **Object count per scene:** N objects, N ∈ 2..6, placed in **non-overlapping** bounding boxes.

**Question templates (one generator per type → exercises the agent's D2 router and D5 type-constraint):**

| Type | Template | Gold answer derivation |
|---|---|---|
| `yes_no` | "is there a {shape}?" / "is there a {color} {shape}?" | yes/no from spec membership |
| `count` (number) | "how many {color} {shape}s?" | integer from `counts_by_color_shape` |
| `color` (other) | "what color is the {shape}?" | the color — **only emitted when that shape is unique** so the gold is unambiguous |
| `object` (other) | "what shape is the {color} object?" | the shape |

Each item also carries a **10-answer list synthesized from the gold** so the soft VQA accuracy (§5) runs unchanged.

### 4.2 Determinism — self-describing PNGs

The scene specification is **embedded inside the PNG metadata**, so the image is self-describing and evaluation is fully deterministic and stateless:

```python
spec = {
  'objects': [{'shape': ..., 'color': ..., 'bbox': ...}, ...],
  'counts_by_color_shape': {...},
  'colors_present': [...],
  'shapes_present': [...],
}
meta = PngImagePlugin.PngInfo()
meta.add_text('scene_spec', json.dumps(spec))
img.save(path, pnginfo=meta)
```

The torch-free **`StubVQA`** model reads the spec back out (`json.loads(Image.open(path).text['scene_spec'])`), computes the true answer from the template logic, and returns a **realistic top-k distribution** (≈0.7-0.9 mass on the correct answer, the remainder spread over plausible type-consistent distractors — other colors, adjacent counts) — *not* a one-hot — so that `p_max`, entropy, and top1-top2 margin at the agent's D3/D4 gates are meaningful. A `noise` / `difficulty` knob deliberately lowers `p_max` or injects a type-wrong top-1 on a fraction of items, so the **abstention gate (D4)** and **type-consistency re-rank (D5)** actually trigger and can be unit-tested.

**Verified offline:** stub answers seed scenes correctly, and all 5 agent decisions fire. Swapping `StubVQA` for a real `ViltForQuestionAnswering` wrapper (same `predict()` signature returning top-k + probs) flips the system to production with **no agent changes**.

### 4.3 Limitations vs real photos

The synthetic generator is a **control harness, not a model of the world**. It is intentionally narrow:

- **Closed, tiny vocabulary.** 3 shapes × 6 colors only. No real objects, people, animals, text, scenes, or backgrounds. None of the COCO-style complexity (occlusion, lighting, clutter, perspective, ambiguity).
- **Clean, unambiguous scenes.** White background, non-overlapping shapes, no noise, no blur. Real VQA images are messy; the generator never reproduces image-quality or ambiguity failure modes by default.
- **Template questions only.** A handful of fixed templates, no paraphrase variety, no compositional or knowledge or reasoning questions. It exercises the agent's *plumbing* (the 5 decisions, the metric, the type routing), **not** real visual understanding.
- **Synthesized annotators.** The 10-answer list is derived from the single gold answer, so it does **not** reproduce genuine human disagreement; the soft metric runs but its "softness" is degenerate on synthetic items.
- **No language-prior or demographic bias to measure.** Because there is no real-world content, the synthetic set cannot surface the biases described in §6. Those require the real datasets.

**Conclusion:** synthetic data **validates the system end-to-end (logic, metric, agent, CI)** and is the default offline path; it does **not** validate real-world VQA accuracy or fairness. Real accuracy and bias claims must come from `lmms-lab/VQAv2` (and the side benchmarks).

---

## 5. The metric this data feeds (brief)

The official **VQA accuracy** is computed against the 10-annotator `answers` block. Per candidate answer (after normalization): `acc(a) = min(1, matches/3)` where `matches` = number of the 10 annotators who gave `a`; the project implements the canonical **10× leave-one-out average** to match the reference `VQAEval`. **Both prediction and all 10 GT answers pass through the SAME normalization** (lowercase, strip/standardize punctuation without splitting decimals, number-word → digit, drop articles `a/an/the`, contraction canonicalization). There is **no `evaluate-metric/vqa` on HF Hub** — the metric is re-implemented in this repo.

**Reported:** overall accuracy + per-`answer_type` (`yes/no`, `number`, `other`) + per-`question_type` (65 buckets) + **abstention rate / coverage** from the agent. **Baselines:** prior-"yes"; most-common-answer-per-type prior; the **blind / question-only** baseline (the key bias probe); and zero-shot pretrained ViLT. The headline agent metric is *answer-when-confident accuracy vs raw argmax*.

> The metric is meaningful **only** on datasets that carry the 10 annotators: `lmms-lab/VQAv2`, `HuggingFaceM4/VQAv2`, `Multimodal-Fatima/VQAv2_sample_train`. It is **not** computable on `merve/vqav2-small` (3 columns, no annotators).

---

## 6. Known biases and limitations

These biases are **properties of the real datasets and the models trained on them**, and they are why the agent (calibrated abstention + type constraint) and the blind baseline exist.

- **Language priors (the headline VQA bias).** VQA models learn to answer from the *question text alone*: "what color is the banana?" → "yellow", "is it...?" → "yes", "how many...?" → "2", regardless of the image. VQAv2's complementary-pair balancing **depresses but does not eliminate** this. **Mitigation/measurement:** the project always runs the **blind question-only baseline**; the full-model gap over it is the real evidence the image is used.

- **Demographic / object-frequency bias in COCO.** COCO is a convenience sample of common Western web photos. Object frequencies are skewed (some categories are far more common than others), scenes and people are not demographically representative, and known correlations (e.g. activity↔gender) leak into the annotations. Models amplify these correlations. **Mitigation:** report **per-answer-type** and **per-question-type** accuracy rather than a single number, so skew is visible; treat demographic predictions with caution (§7 out-of-scope).

- **Overconfident, poorly-calibrated softmax.** VQA classifiers skew their probabilities high; raw `p_max` is not a reliable confidence. **Mitigation:** the agent's D4 gate uses temperature-scaled `p_max` + entropy + top1-top2 margin (calibrated on a held-out split), not a raw constant — this is the single most defensible value-add.

- **Closed answer vocabulary (3129) ceiling.** Any gold answer outside the canonical 3129 labels (§3.5) is **unreachable** by the classification core — a hard ceiling on `other`-type questions especially. Generative models avoid the ceiling but lose the clean confidence signal. The 3129 vocab, the metric, the constraint sets, and the report must all use the **same** vocabulary.

- **Undeclared licenses on practical mirrors.** The cleanest train mirrors declare no license (§8). This is a *legal* risk, not a data-quality one, but it constrains commercial use.

- **Annotator confidence noise.** The `answer_confidence ∈ {yes, maybe, no}` field reflects annotator self-reported certainty; "maybe"/"no" answers are noisier. The soft metric already tolerates this via the `/3` threshold, but it is a source of label noise.

---

## 7. Intended use and out-of-scope use

### 7.1 Intended use

- **Natural-image VQA** in the COCO/VQAv2 style: short answers to yes/no, counting, color, and object questions about everyday scenes.
- **Training / fine-tuning** the VQA core on VQAv2 train (`HuggingFaceM4/VQAv2`) and **evaluating** with the official soft accuracy on `lmms-lab/VQAv2` validation.
- **Offline development, CI, and unit testing** via the synthetic scene generator (§4) — the primary path for any torch-free / no-network run.
- **Research and education** on VQA, language priors, calibration, abstention, and type-aware answer constraints. **Assistive use** (e.g. VizWiz-style help for blind users) is an *intended research direction* — but only under the safeguards below, because the model **assists and abstains**, it never asserts certainty.

### 7.2 Out-of-scope / prohibited use

- **High-stakes decisions on people.** Do **not** use VQA outputs for medical, legal, safety, surveillance, hiring, or any decision affecting a person. The model is overconfident and biased; for accessibility and medical contexts it must **abstain or flag low confidence**, never assert.
- **Demographic inference.** Do **not** ask the model to infer race, gender, age, health, identity, or other protected attributes of people in images. COCO-driven correlations make such outputs both unreliable and harmful.
- **Privacy-sensitive imagery.** VQA on **user photos is sensitive** — faces, homes, documents, medical images, location cues. Default to **local processing, explicit consent, and no raw-image retention.** Do not build retention/indexing of user images on top of this without an explicit privacy review.
- **Document / infographic / OCR-as-primary VQA.** Out of scope for the trainable core (TextVQA is only a *side* benchmark). The `pix2struct-vqav2-base` document model is **not** used (it does not exist; pix2struct ships only doc/infographic variants — see the design brief §3).
- **Treating synthetic data as a real-world benchmark.** The synthetic generator validates *plumbing*, not visual understanding or fairness (§4.3). Never report synthetic accuracy as a real-world VQA result.
- **Adversarial / unanswerable questions answered with false confidence.** Questions about objects not in the image, or genuinely unanswerable ones, must route to abstention (`unsure` / `needs_review`), not a hallucinated answer. The VizWiz `unanswerable` label is used to test exactly this.

---

## 8. Licensing decision (definitive)

**Commercially safe — license declared & clean:**
- `lmms-lab/VQAv2` — **cc-by-4.0**
- `facebook/textvqa` — **cc-by-4.0**
- `lmms-lab/GQA` — **mit**
- `dandelin/vilt-b32-finetuned-vqa` (vocab source) — **apache-2.0**
- The **synthetic scene generator** — project-owned (this repo).

**FLAG — license unconfirmed (undeclared on repo; CC-BY-4.0 upstream). Verify before any commercial use:**
- `HuggingFaceM4/VQAv2` (train mirror)
- `merve/vqav2-small` (demo)
- `Multimodal-Fatima/VQAv2_sample_train` and other `Multimodal-Fatima/*`
- `lmms-lab/OK-VQA`, `lmms-lab/VizWiz-VQA`
- `Luo-wj/DAQUAR`

**AVOID — do not use:**
- `thangduong0509/daquar_vqa` — **cc-by-nc-sa-4.0 (NON-COMMERCIAL).** Excluded. If a DAQUAR mirror is needed, use `Luo-wj/DAQUAR` (real data, but license-unconfirmed → FLAG).
- `jp1924/VisualQuestionAnswering` — **gated + Korean, not VQAv2.** Excluded.

**Default data decision:** TRAIN = `HuggingFaceM4/VQAv2` (only train mirror with the 10 annotators; FLAG license, needs `trust_remote_code=True`, pin `datasets`). EVAL = `lmms-lab/VQAv2` validation (cc-by-4.0, clean parquet). DEMO = `merve/vqav2-small` (quick) or `Multimodal-Fatima/VQAv2_sample_train` (when real offline scoring is needed). VOCAB = `dandelin/vilt-b32-finetuned-vqa` config (3129). **CI / offline / default-dev = the synthetic generator (§4).**

---

## 9. Provenance, attribution & citation

- **Images:** Microsoft COCO 2014/2015 (Lin et al. 2014, "Microsoft COCO: Common Objects in Context"), **CC-BY-4.0**.
- **Annotations:** VQA v2.0 (Goyal et al. 2017, "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering"), building on VQA v1.0 (Antol et al. 2015, "VQA: Visual Question Answering"), **CC-BY-4.0**.
- **Answer vocabulary:** the 3129-label space from `dandelin/vilt-b32-finetuned-vqa` (ViLT, Kim et al. 2021), apache-2.0.
- **Side benchmarks:** GQA (Hudson & Manning 2019, mit), TextVQA (Singh et al. 2019, cc-by-4.0), VizWiz-VQA (Gurari et al. 2018), OK-VQA (Marino et al. 2019), DAQUAR (Malki & Fritz 2014).
- **Synthetic data:** generated by `data/synth_scene.py` in this repository; owned by the project; reproducible bit-for-bit from a seed.

When redistributing or reporting results, **attribute COCO and the VQA project** per CC-BY-4.0, and **state the FLAG status** of any undeclared mirror used.
