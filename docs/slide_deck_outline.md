# P17 — Multimodal Question Answering (VQA): Slide Deck Outline

> Presentation outline for the P17 "Multimodal Question Answering (VQA)" system.
> Package `mmqa` · folder `17_Multimodal_QA` · Author: Le Dinh Minh Quan (23127460).
> ~12 slides, one slide per section, 3–5 sub-bullets each. Speaker notes call out the
> verified Hugging Face ids, licenses (with FLAGs), and the two value-add agent behaviors.

This is the FIRST multimodal-vision project in the series (P02–P15 were text / document /
OCR; P16 done). The deck is built to be delivered in 12–15 minutes: roughly one minute per
slide, with the agent slide (Slide 7) and the metrics slide (Slide 8) as the two centerpieces.

---

## Slide 1 — Title

- **Multimodal Question Answering (VQA)** — given an image + a natural-language question, produce a short answer. Subtitle: "A trainable VQA core wrapped in a deterministic, abstaining, type-aware agent."
- One-line hook example on the title image: a scene of colored shapes + *"how many red squares?"* → **"2"**; a kitchen photo + *"what color is the table?"* → **"white"**.
- Author / id: Le Dinh Minh Quan, student 23127460. Package `mmqa`, folder `17_Multimodal_QA`. Project P17 in the 19-system NLP-industry series.
- Positioning badge: **first multimodal-vision project** in the series (image patches + text tokens fused in one transformer), building on the text/OCR foundations of P02–P16.
- Footer: default core `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0) · runs fully offline via a synthetic-scene stub · deployable as FastAPI + Gradio + Docker + HF Space.

*Speaker note:* set expectations — the trainable artifact is the VQA model; the headline contribution is the agent that makes an overconfident model honest (abstains) and on-format (type-constrained).

---

## Slide 2 — Problem and use cases

- **Task definition:** input is an `(image, question)` pair; output is a short answer string (a word or a small number), not a caption and not a bounding box. Two regimes: classification VQA (fixed 3129-answer vocab, DEFAULT) and generative VQA (open-vocab free text, alternative).
- **Why it matters / use cases:** visual assistance for blind and low-vision users (the VizWiz scenario), image-based search and triage, content moderation/inspection, retail and inventory ("how many … are on the shelf?"), and educational/scene-understanding demos.
- **What "good" looks like:** a correct *short* answer of the *right type* — "yes"/"no" for a yes/no question, a number for "how many", a color word for "what color" — with a confidence signal and the ability to say **"unsure"** instead of guessing.
- **Scope (in):** natural-image VQA (COCO-style scenes), VQAv2 yes/no + number + other answer types, an agentic wrapper, and an offline synthetic harness.
- **Scope (out):** document/infographic/OCR VQA as the primary target (TextVQA is only a side benchmark), multi-turn dialog, and region grounding / boxes as outputs.

*Speaker note:* anchor the assistive-tech use case early — it justifies the abstention design on Slide 7 and the ethics emphasis on Slide 10.

---

## Slide 3 — Why VQA is hard

- **Language priors (the shortcut).** Models learn to answer from the *question text alone* — "what color is the banana?" → "yellow" regardless of the image, "how many?" → "2", "is there…?" → "yes". The image is often ignored. We measure this directly with a **blind, question-only baseline**.
- **Overconfidence.** VQA softmax probabilities are poorly calibrated and skew high; a wrong answer can come back with `p_max ≈ 0.9`. Raw argmax gives no trustworthy "I don't know" — dangerous for assistive use.
- **Type-wrong fluency.** The argmax can be fluent but the wrong *kind* of answer — replying "cat" to "how many?", or a number to a color question — because the head ranks over one flat vocabulary.
- **Hard ceilings and edge cases:** the classification head is fixed at **3129 answers** — anything outside is unreachable (out-of-vocab gold = a hard miss); plus unanswerable questions, blurry/low-quality images, and adversarial or presuppositional questions.
- **Takeaway:** a raw model is accurate-on-average but *unreliable per-item*. P17's contribution is the wrapper that converts an overconfident, type-agnostic model into one that abstains and stays on-format.

*Speaker note:* this slide sets up the two value-add gates (D4 abstention, D5 type constraint) — call them out as the direct answers to "overconfidence" and "type-wrong fluency."

---

## Slide 4 — Data: VQAv2 + synthetic scenes

- **Eval (clean):** `lmms-lab/VQAv2` validation — **cc-by-4.0**, clean parquet, 8 columns including the full **10-annotator** `answers` list and `answer_type ∈ {yes/no, number, other}`. This is the canonical scoring set.
- **Train:** `HuggingFaceM4/VQAv2` — the only common mirror *with* a train split + 10-annotator schema. **FLAG: license undeclared on the repo** (COCO/VQA upstream is CC-BY-4.0); a loading-script dataset needing `trust_remote_code=True` and fragile on newer `datasets` — reserve it strictly for the train split.
- **Demo / tiny:** `merve/vqav2-small` (quick smoke loop, only 3 cols → not valid for official scoring) and `Multimodal-Fatima/VQAv2_sample_train` (1K rows, FULL 10-annotator schema, offline-friendly). Both **FLAG** (undeclared license).
- **Answer vocabulary (3129):** pulled from the `dandelin/vilt-b32-finetuned-vqa` config `id2label` — the canonical ViLT VQA label space, also used for the type-constraint sets and report buckets.
- **Synthetic scenes (offline, required):** `data/synth_scene.py` draws colored shapes (square/circle/triangle × 6 colors) on a PIL canvas, **embeds the scene spec in PNG metadata**, and templates `(question, gold, type)` triples — so the agent, metric, and tests run with **no torch, no model, no network**.

*Speaker note:* emphasize the licensing discipline — every undeclared-license mirror is FLAGGED; the commercially-defensible path is `lmms-lab/VQAv2` (cc-by-4.0) + ViLT (apache-2.0). Side benchmarks (GQA-mit, TextVQA-cc-by-4.0, VizWiz for abstention) are mentioned only if asked.

---

## Slide 5 — The ViLT trainable core + training

- **Default core:** `dandelin/vilt-b32-finetuned-vqa` — **Apache-2.0**, ~113M params, a single-stream ViLT vision-language transformer with a **classification head over ~3129 answers**. Chosen for the cleanest metric, the cleanest confidence signal, and the easiest fine-tune/eval.
- **Why classification over generative here:** the per-answer softmax distribution is the cleanest possible input for the agent's calibrated abstention and top-k re-ranking; generative models would force a noisier logprob proxy.
- **Pipeline:** `ViltProcessor` (`ViltImageProcessor` for RGB-resize-normalize at 384px + `BertTokenizerFast` for `[CLS] question [SEP]`) → `ViltForQuestionAnswering` → logits → softmax → top-k. Only the multimodal encoder + answer head are trained; everything else is pretrained/deterministic.
- **Training tiers:** fine-tune comfortably on a **T4** (batch 16–32 @ 384px), from `dandelin/vilt-b32-finetuned-vqa` or base `dandelin/vilt-b32-mlm` for a custom vocab; A100/L4 for full speed.
- **Alternatives (named, not default):** generative `Salesforce/blip-vqa-base` (BSD-3, ~385M) and `microsoft/git-base-vqav2` (MIT, ~177M); H100 upgrades `Salesforce/blip2-flan-t5-xl` (MIT, ~3.9B, clean constrained decoding) and `Qwen/Qwen2-VL-2B-Instruct` (Apache). **Avoid/FLAG:** `google/pix2struct-vqav2-base` (does not exist, 404), `llava-hf/llava-1.5-7b-hf` (llama2, non-commercial-ish), `Qwen2.5-VL-3B` (Qwen research license).

*Speaker note:* the "swap the model, keep the agent" point — the wrapper exposes one `predict(image, question) → topk+probs` signature, so ViLT ↔ BLIP ↔ stub are interchangeable with no agent changes.

---

## Slide 6 — The question-type classifier

- **What it is:** a deterministic, lowercase-leading-token keyword/regex classifier — **no training, no model** — that tags each question with a type and thereby defines the *allowed answer set*. It is the anchor for the agent's value-add.
- **Coarse buckets (agent router):** `yes_no` (aux/copula/"is there") → `{yes, no}`; `count` ("how many" / "number of") → digits/number-words; `color` ("what color/colour") → a closed color lexicon; `object/other` → open vocab (constraint becomes a no-op).
- **Full 10-rule reference:** yes_no, number, color, where, who, why, when, which, what/object, default — evaluated **top-down, first match wins** (order is critical: "how many" must beat "how"; "what color" must beat generic "what").
- **Maps onto reporting:** the four coarse buckets fold into VQAv2's 3 answer-types (`yes_no→yes/no`, `count→number`, else `→other`) and the 65 fine question-types for per-type tables.
- **Why deterministic:** auditable, instant, offline-testable, and side-effect-free — exactly the kind of decision logic that belongs outside the trained model.

*Speaker note:* this slide is short — it exists to set up D2 on the next slide. Stress "first match wins" with the "how many" vs "how" example.

---

## Slide 7 — The 5-decision agent (the value-add)

- **Shape:** a deterministic finite-state machine wrapping the frozen VQA model. Only **D3** touches the model; the other four gates are pure logic over the model's top-k + softmax. Every decision point is auditable and offline-testable.
- **D1 Input gate** → valid nonzero RGB image (variance > eps, not blank) + non-empty interrogative question (1–128 tokens). Bad image → `error:bad_image`; bad question → `error:bad_question`. Never reaches the model. **D2 Question-type router** → the keyword classifier picks the type and the allowed answer set.
- **D3 Run VQA** → real `ViltForQuestionAnswering` logits *or* the offline `SceneStubVQA`; emits top-k (k=5), `p_max`, top1–top2 `margin`, and Shannon `entropy`. Exception → `error:model_failure`.
- **D4 Calibrated abstention gate (value-add #1):** because VQA models are **overconfident**, thresholds are temperature-calibrated on a held-out split — abstain unless `p_max ≥ τ_conf` **and** `entropy ≤ τ_ent` **and** `margin ≥ τ_margin`. Else answer = **"unsure"**, `status='abstained:low_confidence'`, flag `needs_review`.
- **D5 Type-consistency gate + re-rank (value-add #2):** force the answer to match the D2 type by re-ranking *within* top-k (e.g. count question, top1="cat" 0.4, top2="3" 0.3 → return "3", `status='ok:reranked'`). No type-consistent candidate → `status='abstained:type_mismatch'`. Optional LLM "brain" is **off by default, advisory only, never overrides**.

*Speaker note:* this is the centerpiece. State the thesis plainly — a stock model only does raw argmax; the agent adds **calibrated abstention** + **type-aware constraint**, beating raw argmax with *no extra training*, all from the model's own distribution.

---

## Slide 8 — Metrics and results

- **Primary metric:** official **soft VQA accuracy** — leave-one-out average over the 10 annotators of `min(1, matches/3)`, after the canonical answer normalization (lowercase, strip punctuation, drop articles a/an/the, number-word + contraction mapping). Re-implemented from `VQAEval` (there is **no** `evaluate-metric/vqa` on HF Hub).
- **Headline breakdown:** report overall plus **per-answer-type** accuracy (`yes/no`, `number`, `other`) — expect yes/no highest, number lowest — and a per-question-type table (65 buckets) in the appendix.
- **Agent metrics (the differentiators):** **answer-when-confident accuracy** and **abstention rate / coverage** vs a raw-argmax baseline, plus the **re-rank rate** from D5. The story: trade a little coverage for a large precision/reliability gain.
- **Baselines (report all):** prior "yes"; most-common-answer-per-type prior (yes / "2" / "white"); the **blind question-only** baseline (the gap above it = how much the image actually helps / measures language-prior bias); and zero-shot pretrained ViLT.
- **Offline reproducibility:** the full agent + scorer produce a real accuracy / abstention-rate / re-rank-rate report from the synthetic stub — seed-deterministic, torch-free, runnable in CI and Colab-free mode.

*Speaker note:* lead with the abstention table — "answer-when-confident accuracy vs raw argmax" is the single most defensible headline number. Show the blind-baseline gap to prove the model uses the image.

---

## Slide 9 — Deployment

- **FastAPI service:** `POST /ask` (multipart image upload + question → `{answer, confidence, abstain_flag, question_type, topk}`, route gated on `python-multipart`) and `POST /ask-scene` (JSON path for the synthetic/embedded-scene flow).
- **Gradio UI:** upload an image, type a question, see the answer with its confidence and an explicit "unsure / needs review" indicator when the agent abstains.
- **Docker + HF Space:** containerized image (needs **libGL** for Pillow), publishable as a Hugging Face Space for a live demo.
- **Reused infrastructure:** config / logging / registry / autoreport / monitoring / automation / grading / CLI / API templates carried over from sibling projects (P15 imgtrans, P14 doctrans) — no re-invention.
- **Offline-first operation:** every surface (API, CLI, tests) runs against `SceneStubVQA` with no torch/model/network; swapping in the real `ViltForQuestionAnswering` wrapper flips to production with **no agent changes**.

*Speaker note:* the demo-on-a-Space + offline-stub combo is the "it actually runs anywhere" pitch; mention the libGL gotcha as a real deployment lesson.

---

## Slide 10 — Ethics, privacy, and bias

- **Sensitivity of VQA on user photos:** images can contain faces, homes, documents, and medical content; a major use case is **assistive QA for blind users** (VizWiz). This demands extra care, not generic ML caution.
- **Privacy by default:** require consent, prefer local processing, and **retain no raw images by default** — answer, then discard.
- **Honesty over confidence:** the tool **assists and abstains** — it flags low-confidence answers and says "unsure" rather than asserting certainty, which is critical for accessibility and any medical-adjacent image. This is the D4 gate doing ethical work, not just a metric.
- **Bias — language priors + dataset bias:** VQA models answer "yes"/"2"/"white" from the question alone, and COCO carries demographic skew. We surface this by **reporting the blind-prior baseline and per-type accuracy** rather than hiding behind a single overall number.
- **Robustness threats to name:** overconfidence, the language-prior shortcut, out-of-vocab answers, unanswerable questions, image quality/blur, and adversarial questions — each mapped to a mitigation (calibrated abstention, blind baseline, type constraint, input gate).

*Speaker note:* tie each ethics point back to a concrete mechanism already shown — abstention (D4), input gate (D1), blind baseline (Slide 8). The system's ethics are implemented, not aspirational.

---

## Slide 11 — Conclusion

- **What was built:** a trainable ViLT classification VQA core (`dandelin/vilt-b32-finetuned-vqa`, Apache-2.0) wrapped in a deterministic 5-decision agent that classifies the question type, runs the model, **abstains when uncertain**, and **constrains the answer to the question type**.
- **The thesis, proven:** type-aware constraint + calibrated abstention beat raw argmax with **no extra training** — measured by answer-when-confident accuracy, abstention/coverage, and the blind-baseline gap.
- **Engineering wins:** a re-implemented official VQA-accuracy metric (no HF metric exists), a self-describing synthetic-scene generator + `SceneStubVQA` enabling a **fully offline, torch-free, seed-deterministic** test path, and a one-signature model wrapper that makes ViLT/BLIP/stub interchangeable.
- **Discipline:** every dataset/model id is HF-verified; every undeclared or non-commercial license is **FLAGGED**; the default stack is permissive-clean (apache-2.0 model + cc-by-4.0 eval set).
- **First multimodal milestone:** extends the P02–P16 text/doc/OCR line into vision-language, reusing the shared config/logging/registry/CLI/API templates.

*Speaker note:* end with the one-sentence takeaway — "the same frozen model, made honest and on-format by deterministic logic."

---

## Slide 12 — Future work

- **Stronger / generative cores:** swap in `Salesforce/blip-vqa-base` (open-vocab, escapes the 3129 ceiling) or H100-tier `Salesforce/blip2-flan-t5-xl` / `Qwen/Qwen2-VL-2B-Instruct` (Apache) via LoRA + bf16 — using the encoder-decoder's clean constrained decoding to deepen the D5 type constraint.
- **Better calibration & abstention:** per-question-type thresholds, temperature/Platt scaling on a larger held-out split, and an explicit **unanswerable** detector validated on `lmms-lab/VizWiz-VQA` (real blind-user photos with an unanswerable label).
- **Wider evaluation:** add side benchmarks — `lmms-lab/GQA` (compositional reasoning, MIT), `facebook/textvqa` (text-in-image, cc-by-4.0) — and an image-only ablation to complete the bias-ablation set.
- **Reduce the language prior:** image-only / answer-prior ablations and prior-debiasing so the full-model gap over the blind baseline grows.
- **Productionization:** activate the optional advisory LLM brain for hard "why/where" reasoning (still never overriding the deterministic gates), and harden the FastAPI/Docker/Space deployment for real assistive-tech traffic.

*Speaker note:* frame future work as turning the existing knobs (model tier, thresholds, benchmarks) — the architecture already supports each upgrade without redesign.
