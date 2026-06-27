# Model Card — P17 Multimodal Question Answering (VQA) Core

> Model card for the **trainable VQA core** of the P17 "Multimodal Question Answering" system (package `mmqa`, folder `17_Multimodal_QA`). This card documents the classification VQA model that the deterministic agent wraps — the single trained artifact in the project. Everything else (image preprocessing, question tokenization, answer normalization, the 5-decision agent, scoring) is pretrained or deterministic and is NOT covered here as a trained component.

| Field | Value |
|---|---|
| Project | P17 — Multimodal Question Answering (Visual Question Answering, VQA) |
| Package / folder | `mmqa` / `17_Multimodal_QA` |
| Author | Le Dinh Minh Quan (student 23127460) |
| Model role | Trainable core — multimodal transformer, `(image, question) -> short answer` |
| Default base model | `dandelin/vilt-b32-finetuned-vqa` |
| Base license | Apache-2.0 (clean, commercially usable) |
| Base parameters | ~113M (ViLT, single-stream vision-language transformer) |
| Architecture class | `ViltForQuestionAnswering` (classification head) |
| Processor | `ViltProcessor` = `ViltImageProcessor` + `BertTokenizerFast` |
| Answer regime | Classification over a fixed 3129-answer vocabulary (softmax) |
| Pretrain-from-scratch base | `dandelin/vilt-b32-mlm` (Apache-2.0) for a custom answer vocab |
| First multimodal-vision project in the series | Yes (P02–P15 were text/doc/OCR; P16 done) |

---

## 1. Model overview

The P17 core is a **classification Visual Question Answering model**. Given a natural image and a natural-language question about that image, it produces a short answer drawn from a fixed answer vocabulary. Examples:

- image of a kitchen + "how many chairs are there?" -> `2`
- image of a scene + "what color is the table?" -> `white`
- image of a scene + "is there a stove?" -> `yes`

The default and shipped core is **`dandelin/vilt-b32-finetuned-vqa`** (Apache-2.0, ~113M parameters). ViLT is a single-stream vision-language transformer: it concatenates image-patch embeddings with WordPiece question tokens (`[CLS] question [SEP]`) and runs them through one shared transformer, with no heavy convolutional or region-proposal vision backbone. A **3129-way linear classification head** maps the fused `[CLS]` representation to logits over the canonical VQAv2 answer vocabulary; a softmax yields a per-answer probability distribution.

The classification regime is the deliberate choice because it gives the cleanest metric (exact-match over a closed label space), the cleanest confidence signal (a real softmax distribution over all answers), and is the easiest to fine-tune and evaluate with the Hugging Face Trainer. **This clean distribution is precisely what the agent's calibrated-abstention and type-consistency gates consume** — the agent needs the full top-k distribution, not just `argmax`.

### Why this base and not the alternatives

`dandelin/vilt-b32-finetuned-vqa` is the cleanest permissive, Trainer-friendly option, already a classification head over the canonical 3129 answers, and it maps directly onto the agent's type-constraint and abstention gates. The project also supports generative alternatives as complements and upper-bound comparisons (documented for completeness, not the trained core):

| Role | id | License | Params | Notes |
|---|---|---|---|---|
| **Default trainable core (classification)** | `dandelin/vilt-b32-finetuned-vqa` | Apache-2.0 | ~113M | Shipped core; clean softmax + metric |
| Pretrain base for custom vocab | `dandelin/vilt-b32-mlm` | Apache-2.0 | ~113M | Fine-tune with `num_labels=your_vocab` |
| Generative alternative (default tier) | `Salesforce/blip-vqa-base` | BSD-3-Clause | ~385M | Open-vocab free-text answers |
| Stronger generative | `Salesforce/blip-vqa-capfilt-large` | BSD-3-Clause | ~470M | Open-vocab |
| Lightweight generative contrast | `microsoft/git-base-vqav2` | MIT | ~177M | Open-vocab |
| H100 upgrade (encoder-decoder) | `Salesforce/blip2-flan-t5-xl` | MIT | ~3.9B | Clean constrained decoding; LoRA + bf16 |
| H100 alternative (decoder-only) | `Salesforce/blip2-opt-2.7b` | MIT | ~3.7B | OPT weights carry an upstream Meta research-use caveat |
| Permissive instruction-VLM | `Qwen/Qwen2-VL-2B-Instruct` | Apache-2.0 | ~2.2B | Modern VLM |

**Explicitly avoided / flagged:** `google/pix2struct-vqav2-base` does NOT exist (404 on the Hub — never reference it; Pix2Struct ships only document/infographic variants, the wrong fit for natural-scene VQA). `llava-hf/llava-1.5-7b-hf` carries the `llama2` custom license (restricted-use / non-commercial-ish — FLAG; use only as a flagged upper-bound). `Qwen/Qwen2.5-VL-3B-Instruct` is under the Qwen Research License (non-commercial) with an empty HF license field — FLAG before any commercial use.

---

## 2. Intended use

### Primary intended use
Answer short factual questions about the content of a **single natural image** (COCO-style scenes): object presence, counting, color, and other open-vocabulary attribute/identity questions, of the VQAv2 yes/no + number + other answer-type families. The model is deployed behind the deterministic P17 agent and is intended to be used **only** through that agent, which adds a calibrated abstention gate and a type-consistency constraint on top of the raw model output.

### Intended users
Developers and applications that need a VQA capability with an explicit confidence / "unsure" channel — e.g. an assistive image-description helper, a scene-inspection tool, or a demonstration/teaching system. The accompanying FastAPI service (`POST /ask` with an image upload + question; `POST /ask-scene` for the synthetic path), Gradio UI, Docker image, and HF Space are the intended deployment surfaces.

### Out-of-scope and non-intended use
- **Document / infographic / OCR VQA as the primary target.** The core does not read or reason over text rendered inside images; TextVQA is only a side benchmark. Do not deploy it as a document-understanding model.
- **Knowledge-based VQA.** Questions requiring external world knowledge not present in the image (OK-VQA style) are out of scope; the model has no knowledge-retrieval component.
- **Multi-turn dialog**, region grounding, or bounding-box outputs.
- **Any high-stakes, certainty-asserting use** — medical-image interpretation, safety decisions, identity/biometric judgments. The system is designed to ASSIST and to ABSTAIN, never to assert certainty. See §6 (Ethical considerations).

---

## 3. Inputs and outputs

**Inputs.** An RGB image (decoded, resized to the model resolution — 384px for ViLT — and normalized by `ViltImageProcessor`) and a natural-language question (WordPiece-tokenized as `[CLS] question [SEP]` by `BertTokenizerFast`).

**Outputs.** Logits over the fixed 3129-answer vocabulary -> softmax -> the agent consumes the **top-k (k=5) `(answer, prob)`** plus derived signals:
- `p_max` = top-1 probability,
- top1–top2 `margin` = `p1 − p2`,
- Shannon `entropy` `H = −Σ pᵢ log pᵢ` over the truncated/renormalized distribution.

The final user-facing answer is the agent's emitted answer (possibly `"unsure"` when the model abstains), with a `status`, the inferred `question_type`, a `confidence`, and the `topk`. The emitted answer is run through the canonical VQAv2 answer normalization before scoring.

The 3129 answer vocabulary is the single source of truth: `AutoConfig.from_pretrained('dandelin/vilt-b32-finetuned-vqa').id2label` (length 3129); the answer list is `list(config.id2label.values())`. The metric, the agent's answer-constraint sets, and the report buckets all use this same vocabulary.

---

## 4. Training data

| Split | Dataset id | License | Notes |
|---|---|---|---|
| **Train** | `HuggingFaceM4/VQAv2` | **Undeclared on Hub — FLAG** (COCO 2014/2015 images + VQA annotations CC-BY-4.0 upstream) | Only common mirror with a train split AND the full 10-annotator schema. Loading-script dataset: requires `load_dataset(..., trust_remote_code=True)`, Dataset Viewer returns 501, may fetch COCO images by URL, fragile on newer `datasets`. Reserve strictly for the train split and pin versions. |
| **Eval** | `lmms-lab/VQAv2` (validation) | CC-BY-4.0 (clean) | Clean parquet, 8 columns incl. `answers` (10 annotators), `multiple_choice_answer`, `question_type`, `answer_type ∈ {yes/no, number, other}`. Soft-VQA-accuracy ready. |
| **Demo (quick)** | `merve/vqav2-small` | **Undeclared — FLAG** | Only 3 columns; no annotators / no type fields -> NOT valid for official scoring or type routing. Demo smoke-loop only. |
| **Demo (offline, real scoring)** | `Multimodal-Fatima/VQAv2_sample_train` | **Undeclared — FLAG** | 1K rows with the full 10-annotator schema; offline-friendly once cached. |
| **Answer vocabulary (3129)** | `dandelin/vilt-b32-finetuned-vqa` config | Apache-2.0 (clean) | `id2label`/`label2id` is the canonical 3129-answer label space (a model, not a dataset). |

**Synthetic offline data (primary CI / torch-free path).** The repo ships a synthetic scene generator (`data/synth_scene.py`) that draws colored shapes (square / circle / triangle × red / blue / green / yellow / purple / orange) on a PIL canvas, **embeds the scene spec in the PNG metadata**, and templates `(question, gold answer, type)` triples ("how many red squares?", "what color is the circle?", "is there a triangle?"). This mirrors P15's OCR SeedEngine and lets the agent, metric, and tests run with no torch, no model download, and no network. See §7 (Offline verified behavior).

**Side benchmarks (not the main training/eval loop):** `lmms-lab/GQA` (MIT, compositional reasoning), `facebook/textvqa` (CC-BY-4.0, text-in-image), `lmms-lab/VizWiz-VQA` (license undeclared — FLAG; has an explicit *unanswerable* label, an ideal abstention test).

**Licensing decision.** Commercially defensible declared-clean sets: `lmms-lab/VQAv2` (CC-BY-4.0), `facebook/textvqa` (CC-BY-4.0), `lmms-lab/GQA` (MIT), `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0). Several practical VQAv2 mirrors declare NO license on the repo even though VQAv2's upstream COCO images and VQA annotations are CC-BY-4.0 — these are FLAGGED as "verify before commercial use." `thangduong0509/daquar_vqa` is CC-BY-NC-SA-4.0 (NON-COMMERCIAL) and is not used.

---

## 5. Metrics and evaluation

### Headline metric — official VQA accuracy (soft)
The reported metric is the **official VQA accuracy** (Antol et al. 2015; Goyal et al. 2017), robust to inter-annotator disagreement. There is **no `evaluate-metric/vqa` space on the HF Hub** (verified — not found), so the metric is re-implemented from the canonical `VQAEval` reference (`processPunctuation` + `processDigitArticle` + the accuracy loop).

For a predicted answer `a` (after normalization):
```
acc(a) = min( 1, (# of the 10 human annotators that gave answer a) / 3 )
```
Matching ≥3 humans -> 1.0; 2 -> 0.667; 1 -> 0.333; 0 -> 0.0. The implementation uses the canonical **10× leave-one-out average** (compare the prediction against each 9-of-10 annotator subset and average), to match the reference exactly. Both the prediction and all 10 ground-truth answers pass through the **same** normalization first (lowercase; strip; remove punctuation but keep decimal points and merge digit groups across commas; number-word -> digit map `{none/zero:0 … ten:10}`; drop articles `a/an/the`; canonicalize ~80 contractions). The dataset score is the mean per-question accuracy × 100.

### Reported breakdowns
- **Per answer-type (headline 3-bucket breakdown):** `overall`, `yes/no`, `number`, `other`. Always reported as four numbers ×100. Typical ordering: yes/no highest, other mid, number lowest.
- **Per question-type (65-bucket appendix):** accuracy grouped by the canonical leading phrase ("how many", "what color is the", "is there", …).
- **Abstention / coverage:** the fraction of questions the agent answers vs abstains on (`unsure` / `needs_review`), and the answer-when-confident accuracy. **This is the single most defensible headline of the agentic value-add** — answer-when-confident accuracy and abstention rate vs a raw-argmax baseline.
- **Re-rank rate:** the fraction of answers changed by the D5 type-consistency re-rank.

### Baselines (all reported)
1. **Prior "yes"** — always answer `yes` (lower bound for yes/no).
2. **Most-common-answer (per-type) prior** — the globally most-frequent answer per type (overall `yes`; count `2`; color `white`).
3. **Blind / question-only (language-prior) baseline** — a classifier on the question text alone, NO image. The gap between the full model and this baseline measures how much the image actually contributes. VQAv2 was specifically constructed to depress this baseline, but it is not zero.
4. **Zero-shot pretrained ViLT** — `dandelin/vilt-b32-finetuned-vqa` with no fine-tuning on the target split, the ideal zero-shot classification reference. Generative references: `Salesforce/blip-vqa-base` / `blip-vqa-capfilt-large`.

---

## 6. Limitations

These are real, structural limitations of the classification VQA core. The agent mitigates several of them but cannot remove them.

- **Language-prior bias (shortcut learning).** VQA models answer from the question text alone — "what color is the banana?" -> "yellow" regardless of the image; "how many?" -> "2"; generic -> "white". This is why the **blind question-only baseline is mandatory** in every report, and why the per-answer-type and per-question-type breakdowns are reported rather than a single accuracy number.
- **Fixed 3129-answer vocabulary (a hard ceiling).** The classification head can only ever emit one of the canonical 3129 answers. Any gold answer outside this closed vocabulary is **unreachable** — a strict accuracy ceiling. If you fine-tune from `dandelin/vilt-b32-mlm` with a custom vocab, the metric, the answer-constraint sets, and the report buckets must all use that same vocab. Generative alternatives (BLIP / GIT / BLIP-2) avoid this ceiling but trade away the clean softmax confidence signal.
- **Overconfidence / poor calibration.** ViLT/VQA softmax probabilities are poorly calibrated and skew high; a confident-looking `p_max` is not a reliable correctness probability. Raw `argmax` will emit a confident wrong answer. The agent's D4 gate therefore combines temperature-scaled `p_max` + entropy + top1–top2 margin (calibrated on a held-out split), and **never thresholds on raw `p_max` alone**.
- **Domain shift.** Trained on COCO-style natural scenes (VQAv2). Accuracy degrades on out-of-distribution imagery — diagrams, screenshots, charts, medical/scientific images, heavy blur, low light, unusual framing, or the synthetic-shape scenes used for offline testing (which the StubVQA path, not the real model, serves).
- **No OCR / no text reading.** The core does not read text inside images; "what does the sign say?" / "read the label" questions are not supported (TextVQA is a side benchmark only).
- **No external-knowledge reasoning.** Questions needing world knowledge beyond the pixels (OK-VQA style) are out of scope.
- **Counting is the weakest answer type.** Number/count accuracy is structurally lower than yes/no and other; per-type thresholds make the count abstention gate stricter to compensate.
- **Other robustness gaps:** out-of-vocabulary gold answers, genuinely unanswerable questions, image quality/blur, and adversarial or presupposition-loaded questions ("how many unicorns?") can all produce confident-wrong outputs that the abstention gate, not the model, must catch.

### How the agent constrains these limitations (value-add)
The same frozen model, wrapped by the deterministic 5-decision agent, gains two production-grade behaviors with **no extra training**:

1. **Calibrated abstention (D4)** — say `unsure` / route to `needs_review` when `p_max`, entropy, or margin fail their (tuned) thresholds, trading a little coverage for a large precision/reliability gain instead of hallucinating a confident wrong answer.
2. **Type-aware answer constraint + re-rank (D2 -> D5)** — classify the question type and re-rank within the top-k to the best type-consistent candidate (yes/no -> `{yes,no}`; "how many" -> a number; "what color" -> a color word), fixing the common fluent-but-type-wrong failure (e.g. answering "cat" to "how many?").

An optional advisory LLM "brain" (anthropic) is **OFF by default** and **never overrides** the deterministic decisions.

---

## 7. Offline verified behavior

The whole 5-point agent, the metric, and the scorer run with **zero torch and no network**, deterministic and reproducible from a seed — mirroring P15's OCR SeedEngine.

- **Synthetic self-describing scenes.** `data/synth_scene.py` draws non-overlapping colored shapes on a PIL canvas, embeds the scene spec as JSON in the PNG metadata (`PngInfo().add_text('scene_spec', …)`), and templates `(question, gold, type)` triples per question type, including a 10-answer list so soft VQA accuracy is computable offline.
- **`SceneStubVQA`** reads the embedded scene spec back out of the PNG (`json.loads(Image.open(path).text['scene_spec'])`), computes the true answer via the same template logic, and returns a **realistic top-k distribution (not one-hot)** — ~0.7–0.9 mass on the correct answer, the remainder spread over plausible type-consistent distractors — with a difficulty/noise knob that deliberately lowers `p_max` or injects a type-wrong top-1 on a fraction of items.
- **Verified behavior.** The stub answers seed scenes correctly, the distributions make `p_max` / entropy / margin meaningful, and **all 5 decision points fire** on the synthetic eval set: D1 input gate (rejects blank images and non-questions), D2 question-type router, D3 model run (top-k + entropy + margin), D4 calibrated abstention (the noise knob triggers `abstained:low_confidence`), and D5 type-consistency re-rank (the type-wrong-top1 injection triggers `ok:reranked` and `abstained:type_mismatch`).
- **Drop-in swap to production.** `SceneStubVQA` and the real `ViltForQuestionAnswering` wrapper share the same `predict(image, question) -> {topk, p_max, entropy, margin, regime}` signature, so swapping in the real model flips the system to production with **no agent changes**. Real-model paths are gated behind an availability check so CI stays torch-free.

---

## 8. Ethical considerations

- **Sensitive imagery.** VQA on user photos is inherently sensitive: images can contain faces, homes, documents, medical content, and are used assistively by blind / low-vision users (the VizWiz use case). Treat every input image as potentially personal data.
- **Privacy by default.** Process locally where possible; **do not retain raw images** by default; obtain consent for any image upload. The FastAPI / Gradio surfaces are built to answer-and-discard, not to log raw images.
- **Assist, never assert certainty.** The system is designed to ASSIST and to ABSTAIN or flag low-confidence answers — it must never assert certainty, especially for accessibility or medical-adjacent questions where a confident-wrong answer can mislead a user who cannot verify it against the image. The abstention gate exists precisely for this reason.
- **Bias.** VQA models carry strong **language priors** (answering "yes" / "2" / "white" while largely ignoring the image) and inherit **demographic and scene biases from COCO**. The required mitigations are reporting the blind question-only baseline and the per-answer-type / per-question-type breakdowns, so bias is measured rather than hidden behind a single accuracy number.
- **Robustness obligations.** Overconfidence, the language-prior shortcut, out-of-vocabulary answers, genuinely unanswerable questions, image quality/blur, and adversarial questions are all known failure modes; the calibrated abstention and type-consistency gates are the first-line defenses, and low-confidence answers are surfaced to the user as `unsure` rather than emitted as fact.
- **License hygiene.** Ship on the declared-clean base (`dandelin/vilt-b32-finetuned-vqa`, Apache-2.0) and declared-clean eval data (`lmms-lab/VQAv2`, CC-BY-4.0). Any FLAGGED dataset mirror (undeclared license) must be verified before commercial use; non-commercial sources (e.g. `thangduong0509/daquar_vqa`, CC-BY-NC-SA-4.0) and non-permissive models (LLaVA-1.5 `llama2`, Qwen2.5-VL-3B Qwen Research) are excluded from the default path.

---

## 9. Reproducibility summary

| Item | Value |
|---|---|
| Trainable core | `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0, ~113M) |
| Model / processor classes | `ViltForQuestionAnswering` / `ViltProcessor` (`ViltImageProcessor` + `BertTokenizerFast`) |
| Custom-vocab base | `dandelin/vilt-b32-mlm` (Apache-2.0) |
| Train data | `HuggingFaceM4/VQAv2` (FLAG license; `trust_remote_code=True`) |
| Eval data | `lmms-lab/VQAv2` validation (CC-BY-4.0) |
| Answer vocab | 3129 labels from the base config `id2label` |
| Metric | Re-implemented official VQA soft accuracy (10× leave-one-out) + per-answer-type + per-question-type + abstention/coverage |
| Offline path | Synthetic scene generator + `SceneStubVQA` (no torch, no network) |
| Hardware tiers | T4 fine-tunes ViLT (batch 16–32 @384px); A100/L4 full speed; H100 for BLIP-2 / Qwen via LoRA + bf16 |

This card covers the classification VQA core only. The deterministic agent (input gate, type router, model run, calibrated abstention, type-consistency re-rank), the metric implementation, and the deployment surfaces are documented separately in the P17 design brief and accompanying docs.
