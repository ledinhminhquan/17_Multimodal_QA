# P17 Multimodal Question Answering (VQA) — Project Plan

> Package: `mmqa` · Folder: `17_Multimodal_QA`
> Author: Le Dinh Minh Quan (student 23127460)
> Task: given an `(image, question)` pair → a short answer ("how many red squares?" → "2")
> Default trainable core: `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0, classification VQA over the canonical 3129-answer vocab)
> Companion documents: `docs/DESIGN_BRIEF.md` (authoritative spec), `docs/project_plan.md` (this file).

This plan turns the design brief into an executable schedule: what gets built, in what order, on what hardware, with which risks tracked and who owns what. It is deliberately concrete to **P17 — Visual Question Answering**, in which the **VQA model (`dandelin/vilt-b32-finetuned-vqa`, Apache-2.0) is the only trained stage**, wrapped by a deterministic five-decision agent that classifies the question type, runs the model, **abstains when uncertain**, and **constrains the answer to the question type**.

P17 is the **first multimodal-vision project** in the series (P02–P15 were text / document / OCR; P16 is done). The reuse strategy therefore leans on the sibling text projects for everything non-visual (config, logging, registry, autoreport, monitoring, automation, grading, CLI, API) and builds a focused set of **new vision components** (the VQA model wrapper, image handling, the synthetic-scene generator + stub, the official VQA-accuracy metric, the question-type classifier, and the type-aware abstaining agent).

---

## 1. Goal and definition of done

**Goal.** Answer a natural-language question *about an image* with a short, correct, **type-consistent** answer — and, just as importantly, **say "unsure" instead of hallucinating** when the model is not confident. The headline value-add is not raw accuracy but the deterministic agent that wraps a frozen model with **calibrated abstention** (D4) and a **type-aware answer constraint** (D2→D5), both of which a stock argmax model lacks.

**Definition of done (ship criteria).**

1. One fine-tuned (or zero-shot-validated) `vilt-b32-finetuned-vqa` checkpoint is evaluated with the **official soft VQA accuracy** on `lmms-lab/VQAv2` validation, beating the four bias baselines (prior-"yes", per-type most-common prior, blind question-only, zero-shot ViLT before any target-split tuning).
2. The **synthetic-scene generator** (`data/synth_scene.py`) emits reproducible, self-describing PNGs (scene spec embedded in PNG metadata) plus templated `(image, question, gold_answer, question_type)` items with a full 10-answer list, matching the `lmms-lab/VQAv2` 8-column schema for proper soft scoring.
3. The **`SceneStubVQA`** stub reads the embedded scene spec and returns a realistic top-k distribution (not one-hot), so the **entire 5-decision agent + scorer run with no torch, no model download, no network** — and **all five decisions (D1–D5) fire** on the synthetic eval set (input rejects, type routing, abstention, type-mismatch re-rank).
4. The **VQA-accuracy metric** is re-implemented from the canonical `VQAEval` reference (normalization + 10× leave-one-out soft accuracy + per-answer-type and per-question-type aggregation) — there is **no `evaluate-metric/vqa` on HF Hub**, so it must be hand-written.
5. The agent reports **answer-when-confident accuracy, abstention rate, coverage, and re-rank rate** against a raw-argmax baseline — the single most defensible headline result.
6. FastAPI (`POST /ask`: image upload + question → answer + confidence + abstain flag, gated on `python-multipart`; `POST /ask-scene` JSON for the synthetic path) + Gradio UI (upload image, type a question) + Docker (needs `libGL` for Pillow) + a published HF Space.
7. Autoreport, monitoring, grading, and automation templates wired (ported from P15 `imgtrans` / P14 `doctrans`); ethics/privacy doc complete.

**Verified offline-seed sanity floor (not the headline result).** The synthetic harness is already verified offline: the stub answers seeded scenes correctly, all five decisions fire, and the soft-VQA scorer produces an accuracy / abstention-rate / re-rank-rate report **with no torch and no network**. The seed accuracy saturates because the stub reads the gold scene spec — it is a *plumbing* check, not a model result. The honest headline numbers come from the real ViLT on `lmms-lab/VQAv2` validation (typical ordering: yes/no highest ~80–90%, "other" mid, number lowest), reported per answer-type ×100.

---

## 2. Pipeline diagram

```
                                  ┌──────────────────────────────────────────────┐
                                  │              AGENT (deterministic FSM)         │
                                  │   5 decision points, runs fully offline        │
                                  │   (SceneStubVQA + keyword classifier, no torch)│
                                  └──────────────────────────────────────────────┘

   input (image + question  |  synthetic scene spec)
         │
         ▼
   ┌───────────┐   D1 INPUT GATE (PIL + regex, no torch)
   │  INGEST   │── valid RGB image (nonzero, variance > eps) AND non-empty
   │           │     interrogative question (≥1 alpha token, len 1..128) → D2
   └───────────┘── corrupt/blank image → halt status='error:bad_image'
         │       ── empty / non-question → halt status='error:bad_question'
         ▼
   ┌───────────┐   D2 QUESTION-TYPE ROUTER (keyword/regex, first-match-wins)
   │ CLASSIFY  │── 'is/are/does..' → yes_no {yes,no}
   │           │── 'how many'      → count  {0..20, none}
   │           │── 'what color'    → color  {red,blue,green,yellow,...}
   │           │── else            → object/other (open vocab; D5 = pass-through)
   └───────────┘     defines the ALLOWED ANSWER SET used at D5
         │   question_type, allowed_set
         ▼
   ┌───────────┐   D3 RUN VQA (the ONLY step that touches the model)
   │  ANSWER   │── real: ViltForQuestionAnswering → logits → softmax → top-k
   │ (TRAINED) │── offline: SceneStubVQA reads embedded scene_spec → realistic dist
   └───────────┘── exception → halt status='error:model_failure'
         │   top-k (k=5), p_max=p1, entropy H, margin (p1-p2), regime
         ▼
   ┌───────────┐   D4 CALIBRATED ABSTENTION GATE (VQA models are OVERCONFIDENT)
   │ CALIBRATE │── p_max ≥ τ_conf AND H ≤ τ_ent AND margin ≥ τ_margin → confident → D5
   │           │── fails any (temperature-scaled, per-type thresholds) →
   └───────────┘     answer='unsure' status='abstained:low_confidence' (+needs_review)
         │
         ▼
   ┌───────────┐   D5 TYPE-CONSISTENCY GATE + RE-RANK (within top-k)
   │ CONSTRAIN │── top1 in allowed set → return {answer, status='ok'}
   │           │── lower top-k entry consistent → re-rank → status='ok:reranked'
   │           │── NO top-k entry matches type → abstain status='abstained:type_mismatch'
   └───────────┘     object/other: open set → pass-through
         │
         ▼
   output {answer, status, question_type, confidence, topk}
          → canonical VQA answer-normalization (§5) → soft VQA-accuracy scoring
```

Only the **ANSWER** stage (D3) is trained. The image processor (RGB convert / resize 384px / normalize) and tokenizer (`[CLS] question [SEP]`, WordPiece) are pretrained; the input gate, type router, abstention gate, type re-rank, answer normalization, and scorer are **deterministic algorithm — no training**. Swapping `SceneStubVQA` for the real `ViltForQuestionAnswering` wrapper (same `predict(image, question) → {topk, p_max, entropy, margin, regime}` signature) flips the system to production with **no agent changes**.

---

## 3. Milestones and timeline

Plan spans **8 weeks** of part-time effort (one student author) organised into nine milestones. Weeks overlap where a milestone is unblocked by an earlier deliverable. Each milestone lists its **exit criterion** — the concrete, checkable thing that marks it done.

### Milestone table

| # | Milestone | Week(s) | Key deliverables | Exit criterion | Hardware |
|---|-----------|---------|------------------|----------------|----------|
| **M0** | Research & scaffold | 1 | Re-verify every HF id (`hub_repo_details`) + license; confirm `google/pix2struct-vqav2-base` is still 404 and stays out; port P02 repo/notebook/docs template; port config/logging/registry from P15/P14; pull the 3129-answer vocab from `dandelin/vilt-b32-finetuned-vqa` config `id2label` | All ids resolve; non-commercial/empty-license ids flagged (LLaVA llama2, Qwen2.5-VL, undeclared mirrors); 3129-vocab loaded (`len==3129`); offline smoke cell runs on stdlib+Pillow | CPU / free T4 |
| **M1** | Synthetic scenes + stub | 1–2 | `data/synth_scene.py` `make_scene(seed)` → self-describing PNG (scene_spec in PNG metadata) + templated `(image, question, gold, type)` over the 4 types; 10-answer list per item (VQAv2 8-col schema); `SceneStubVQA` returning realistic top-k (+ noise/difficulty knob); committed tiny fixtures | Same seed → same image (hash-stable); stub answers seed scenes correctly; **all 5 decisions fire** on the synthetic set; soft-VQA scorer runs torch-free | CPU |
| **M2** | VQA model wrapper + fine-tune | 2–3 | Uniform `predict()` wrapper over `ViltForQuestionAnswering` (classification, default) + `BlipForQuestionAnswering` (generative alt); the 3 notebook entry points (pipeline / explicit ViltProcessor+logits / Blip `.generate()`); fine-tune on `HuggingFaceM4/VQAv2` train (or zero-shot validate ViLT) | Wrapper returns `{topk, p_max, entropy, margin, regime}`; ViLT zero-shot beats the bias baselines on a held-out slice; if fine-tuned, beats zero-shot | **T4 (batch 16–32 @384px) → A100/L4 default; H100 for BLIP-2/Qwen via LoRA+bf16** |
| **M3** | VQA-accuracy metric | 3–4 | Re-implemented `VQAEval`: answer normalization (§5) + 10× leave-one-out soft accuracy + per-answer-type (yes/no, number, other) + per-question-type (65-bucket) aggregation; `exact_match`/`accuracy` only as auxiliary sanity checks | Metric reproduces the canonical formula on hand-checked cases; one normalization function used for BOTH prediction and all 10 GT answers; per-type tables emitted | CPU |
| **M4** | Question-type classifier | 4 | `vqa/qtype.py` 10-rule keyword scheme (§8, first-match-wins), mapping to the 4 coarse agent buckets {yes_no, count, color, object/other} and the 3 VQAv2 answer-types; closed lexicons (color words, number words {0..20,none}) | 'how many' beats 'how'; 'what color' beats 'what'; classifier matches gold `question_type` on the synthetic + a `lmms-lab/VQAv2` slice | CPU |
| **M5** | Agentic FSM | 5–6 | `src/mmqa/agent/` 5 decision points (D1 input gate, D2 router, D3 model, D4 calibrated abstention, D5 type re-rank); temperature scaling on a held-out split; per-decision trace logging; optional advisory LLM "brain" (anthropic) OFF by default, never overrides | Agent routes ok / ok:reranked / abstained:low_confidence / abstained:type_mismatch / error:* on crafted inputs; runs fully offline (Stub + keyword classifier) | CPU (+ T4 for τ-calibration) |
| **M6** | Evaluation & baselines | 6 | Soft VQA accuracy (overall + per-answer-type + per-question-type ×100); the 4 baselines (prior-'yes', most-common prior, **blind question-only**, zero-shot ViLT); **answer-when-confident accuracy + abstention rate + coverage + re-rank rate** vs raw argmax; autoreport tables/plots; optional VizWiz unanswerable abstention test | All baseline numbers reported; blind-prior gap quantified ("how much the image helps"); abstention precision/coverage trade plotted | T4/A100 |
| **M7** | Deploy | 7 | FastAPI (`POST /ask` image upload + question, gated on python-multipart; `POST /ask-scene` JSON), Gradio UI (upload + question box, shows answer + confidence + abstain flag), Dockerfile (libGL for Pillow), HF Space | Image upload → answer + confidence + abstain flag via API; Space live; Docker builds | CPU host + HF Space |
| **M8** | Hardening & docs | 8 | Robustness pass (overconfidence, language-prior shortcut, out-of-vocab answers, unanswerable questions, blur/quality, adversarial questions); ethics/privacy doc (faces/homes/medical/assistive); monitoring/grading/automation; final README + report | Grading harness green; ethics section complete; report reproduces headline accuracy + abstention numbers | CPU |

### Narrative timeline

- **Week 1 — Research & data start (M0, M1).** Re-verify all model/dataset ids against the brief's VERIFIED STACK; do **not** invent ids (especially: `google/pix2struct-vqav2-base` does not exist — keep it out). Pull the canonical **3129-answer vocab** from the `dandelin/vilt-b32-finetuned-vqa` config — it is the source of truth for the metric, the answer-constraint sets, and the report buckets. Stand up the P02 repo skeleton and the P15/P14 config/logging/registry/autoreport templates. Begin the **synthetic-scene generator** — it is on the critical path because the offline agent, the scorer, and CI all depend on its self-describing PNGs.
- **Week 2 — Scenes + stub done, model wrapper starts (M1, M2).** Lock determinism (per-seed PNG, embedded `scene_spec`, hash-stable). Verify the stub returns a *realistic* distribution (≈0.7–0.9 on the truth, mass spread over type-consistent distractors) so D4/D5 are actually exercised. Then build the uniform `predict()` wrapper and validate zero-shot ViLT.
- **Week 3 — Metric lands (M2, M3).** Re-implement `VQAEval` from the canonical reference (no HF metric exists). This is delicate: punctuation handling must not split decimals (`2.5` stays) and must strip digit-group commas (`100,000`→`100000`); the same normalization runs on prediction *and* all 10 GT answers; implement the **10× leave-one-out** average, not just `min(1, matches/3)`.
- **Week 4 — Question-type classifier (M4).** The 10-rule keyword table (§8), order-critical ('how many' before 'how', 'what color' before 'what'). This is the **value-add anchor** — the chosen type defines the allowed answer set at D5.
- **Weeks 5–6 — Agent + eval (M5, M6).** Build the five-decision FSM; calibrate D4 thresholds with **temperature scaling on a held-out split** (raw `p_max` alone is not enough — VQA models are overconfident). Then run the full evaluation: the four bias baselines and the headline **answer-when-confident accuracy / abstention rate / coverage** vs raw argmax.
- **Week 7 — Deploy (M7).** FastAPI + Gradio + Docker (libGL) + HF Space.
- **Week 8 — Hardening & docs (M8).** Robustness suite (overconfidence, language priors, OOV, unanswerable, blur, adversarial), ethics/privacy, monitoring/grading/automation, final report.

### Critical path

`M1 (scenes + stub) → M3 (metric) / M4 (qtype) → M5 (agent) → M6 (eval & baselines)`.
The synthetic generator (M1) is the single most upstream dependency: it produces the gold + self-describing images that drive the offline agent, the soft-VQA scorer, and CI. M5→M6 is the headline deliverable (calibrated abstention + type constraint, then the eval that proves it beats raw argmax); slipping it is more costly than slipping deploy (M7), which can be reduced to API-only if time runs short. **M2 (the real fine-tune) is parallel and not on the offline critical path** — the entire agent and scorer are validated against `SceneStubVQA` first, and the real model is dropped in behind the same `predict()` signature.

---

## 4. Risk register

Severity = Low / Med / High. Likelihood = Low / Med / High. Ordered by combined exposure.

| ID | Risk | Likelihood | Severity | Mitigation | Owner |
|----|------|------------|----------|------------|-------|
| R1 | **Overconfident VQA models.** Classification softmax is poorly calibrated and skews high — thresholding on raw `p_max` alone would either never abstain or abstain randomly | High | High | D4 is a **tuned** gate, not a constant: temperature-scale the softmax on a held-out split and combine `p_max` + entropy + top1–top2 margin; optionally per-type thresholds (count stricter, yes/no higher). Headline metric = answer-when-confident accuracy + abstention rate vs raw argmax. | Author |
| R2 | **Language-prior bias.** Model answers from the question text alone ('what color is the banana?'→'yellow' regardless of image); also COCO demographic bias | High | High | Always run the **blind question-only baseline**; the full-model gap over it is the real "image helps" signal. Report per-answer-type accuracy. VQAv2 is built to depress this, but it is not zero — say so honestly in the report. | Author |
| R3 | **Dataset fragility / `trust_remote_code`.** `HuggingFaceM4/VQAv2` (the only train mirror with 10 annotators) is a loading-script dataset: Viewer returns 501, needs `trust_remote_code=True`, may fetch COCO images by URL, fragile on newer `datasets` | High | Med | Reserve it strictly for the **train split** and pin `datasets`/`transformers` versions. Use `lmms-lab/VQAv2` clean parquet for **eval**. For offline/CI, never touch it — use the synthetic generator. `Multimodal-Fatima/VQAv2_sample_train` (1K, full 10-annotator schema) for offline real scoring. | Author |
| R4 | **Answer-vocab mismatch (hard ceiling).** Classification head is fixed at the canonical 3129 answers; any gold answer outside is unreachable — an accuracy ceiling | Med | High | Pull the vocab from `dandelin/vilt-b32-finetuned-vqa` config `id2label` and use the **same** vocab for the metric, the D5 answer-constraint sets, and the report buckets. Report the OOV rate (gold answers not in vocab) as a known ceiling. Generative BLIP avoids the ceiling but trades away the clean confidence signal — note the trade. | Author |
| R5 | **Non-commercial / undeclared licenses.** LLaVA-1.5-7B (llama2), Qwen2.5-VL-3B (Qwen Research, empty HF field), and several VQAv2 mirrors (`HuggingFaceM4/VQAv2`, `merve/vqav2-small`, `Multimodal-Fatima/*`, VizWiz, OK-VQA) declare no license; `thangduong0509/daquar_vqa` is CC-BY-NC | Med | High | Hard rule: ship only declared-permissive (ViLT Apache, BLIP BSD-3, GIT MIT; `lmms-lab/VQAv2` CC-BY-4.0, `facebook/textvqa` CC-BY-4.0, `lmms-lab/GQA` MIT). Flag every undeclared mirror as "license unconfirmed — verify before commercial use"; LLaVA/Qwen2.5-VL only as flagged upper-bound comparisons, never default. **Never** use the CC-BY-NC DAQUAR mirror. | Author |
| R6 | **`SceneStubVQA` too easy (one-hot) → abstention never triggers.** If the stub returns a one-hot truth, D4/D5 are never exercised and the offline harness proves nothing | Med | Med | Stub returns a **realistic** distribution (~0.7–0.9 on truth, remainder over plausible type-consistent distractors); a `noise`/`difficulty` knob deliberately lowers `p_max` or injects a type-wrong top1 on a fraction of items so D4 (abstain) and D5 (re-rank) fire and can be unit-tested. CI asserts every status appears. | Author |
| R7 | **Metric re-implementation bugs.** No `evaluate-metric/vqa` exists; a hand-written normalizer easily mishandles decimals/commas/articles/contractions → correct answers score as misses (esp. for generative free-text) | Med | High | Re-implement from the canonical `VQAEval` reference; unit-test the tricky cases (`2.5` keeps its period; `100,000`→`100000`; number-word→digit; drop a/an/the; ~80 contractions). One normalization function = single source of truth for predictions AND references. Cross-check overall against `exact_match`/`accuracy` as loose sanity bounds only. | Author |
| R8 | **T4 OOM / slow fine-tune** of ViLT on free Colab | Med | Med | ViLT ~113M trains comfortably on T4 (batch 16–32 @384px) — the lightest viable core. If tight: reduce batch + grad-accum, or fall back to **zero-shot** ViLT as the validated baseline (the agent value-add is independent of fine-tuning). A100/L4 is the comfortable default; H100 only for BLIP-2/Qwen LoRA experiments. | Author |
| R9 | **Generative-path eval is harder.** BLIP free-text answers must pass the exact canonical normalization before exact-match, or correct answers miss; confidence is only a logprob proxy | Med | Med | Route the generative output through the **same** §5 normalization as classification; use sequence/token logprob as the abstention proxy and document it as weaker than the classification softmax. Keep classification (ViLT) as the default the agent is designed against. | Author |
| R10 | **No clean train split with annotators.** The only train mirror with the 10-annotator schema (`HuggingFaceM4/VQAv2`) is the fragile/undeclared one; `lmms-lab/VQAv2` has **no train split** | Med | Med | Train on `HuggingFaceM4/VQAv2` (flagged, pinned) **or** validate zero-shot ViLT and skip fine-tune; evaluate on the clean `lmms-lab/VQAv2` validation. Offline real scoring uses `Multimodal-Fatima/VQAv2_sample_train` (1K, full schema). Be explicit in the report about which split produced which number. | Author |
| R11 | **Heavy / flaky image deps in CI/Colab-free.** torch + transformers + vision stack + multi-GB parquet/COCO are heavy and break headless CI | Med | Low | The synthetic-scene + `SceneStubVQA` path keeps **tests and the full agent torch-free and network-free**; gate all real-model paths behind an availability check (mirror of P15's SeedEngine). CI runs on stdlib + Pillow only. | Author |
| R12 | **Unanswerable / out-of-distribution questions.** Real user photos produce questions the closed vocab cannot answer; a forced argmax is a confident wrong answer | Med | Med | D4 abstention + D5 type-mismatch handle most; validate against `lmms-lab/VizWiz-VQA` (explicit **unanswerable** label — ideal abstention test; FLAG license, eval-only). Report abstention precision on the unanswerable slice. | Author |
| R13 | **Schedule slip on M5/M6** (the headline agent + abstention eval) | Med | Med | These are the highest-value deliverables — protect their weeks. Reduce M7 deploy to API-only (drop Docker/Space polish) and skip the optional BLIP-2/Qwen H100 experiments before cutting the agent or its evaluation. | Author |
| R14 | **Pillow / metadata round-trip.** The self-describing PNG depends on `PngInfo` text chunks surviving save/reopen; a re-encode could strip `scene_spec` | Low | Med | Write spec via `PngImagePlugin.PngInfo().add_text('scene_spec', ...)`, read via `Image.open(path).text['scene_spec']`; CI asserts round-trip survives and the stub recovers the exact spec. Keep the spec also returned in-memory so the stub never depends solely on disk. | Author |

---

## 5. Resource needs

### Compute (Colab tiers, per the brief's GPU defaults)

| Tier | Hardware | Used for | When |
|------|----------|----------|------|
| **Free** | Colab **T4** (~16 GB) | ViLT ~113M fine-tune (batch 16–32 @384px) or zero-shot validation; BLIP-base inference / LoRA; D4 temperature calibration; all offline tests | M2 first pass, M5 calibration, CI |
| **Default** | **A100 / L4** (mid–large GPU) | Comfortable ViLT full-speed fine-tune + full `lmms-lab/VQAv2` validation eval; `blip-vqa-base` / `blip-vqa-capfilt-large` full fine-tune; `git-base-vqav2` generative contrast | M2 main runs, M6 eval |
| **Upgrade** | **H100 / A100 80 GB** | `blip2-flan-t5-xl` (clean constrained/instruction decoding, fits the agent's type constraints) / `blip2-opt-2.7b` / `Qwen2-VL-2B` via **LoRA + bf16** — upper-bound generative comparisons only | Optional quality experiments only |

The synthetic generator (Pillow), `SceneStubVQA`, the question-type classifier, the agent FSM, the VQA-accuracy metric, and all scoring run **CPU-only**. GPU is needed exclusively for the real VQA model (fine-tune + zero-shot inference) and the optional BLIP-2 / Qwen upgrade experiments.

### Software / accounts

- **HF Pro** (recommended): faster Hub downloads of the 769K-row `lmms-lab/VQAv2` parquet, Space hosting for the demo, and hosting for the fine-tuned ViLT checkpoint.
- **Hugging Face account** (authenticated as `ledinhminhquan`) for `hub_repo_details` id verification, dataset access, and the published Space.
- **Anthropic API key** (optional): the advisory LLM "brain" in the agent is **OFF by default**, advisory-only, and **never overrides** the deterministic decision — only needed if the optional path is demoed.
- System packages for Docker/deploy: **`libGL`** (Pillow/OpenCV runtime for image handling in the container).
- Python deps: `transformers`, `torch`, `Pillow`, `datasets` (pinned — `HuggingFaceM4/VQAv2` is fragile on newer versions and needs `trust_remote_code=True`), `fastapi`, `gradio`, `python-multipart` (the `POST /ask` image-upload route is gated on it); optional `peft`/`bitsandbytes` for BLIP-2/Qwen LoRA, optional `qwen-vl-utils` for the Qwen path.

### Data

- **Train:** `HuggingFaceM4/VQAv2` train (~443K, only mirror with the 10-annotator schema; **FLAG** license, needs `trust_remote_code=True`, pin versions) — used only when fine-tuning.
- **Primary eval:** `lmms-lab/VQAv2` validation (214.4K rows, CC-BY-4.0 clean parquet, 8 cols with full 10-annotator `answers`, `question_type`, `answer_type`).
- **Demo / fast smoke:** `merve/vqav2-small` (quick, 3 cols, demo only — not valid for soft scoring) or `Multimodal-Fatima/VQAv2_sample_train` (1K, full 10-annotator schema, offline-friendly when real scoring is needed).
- **Answer vocab:** `dandelin/vilt-b32-finetuned-vqa` config `id2label` — the canonical **3129** answers (load once; `len == 3129`).
- **Side benchmarks (eval-only):** `lmms-lab/GQA` (MIT, compositional), `facebook/textvqa` (CC-BY-4.0, text-in-image), `lmms-lab/VizWiz-VQA` (explicit **unanswerable** label → abstention test; FLAG license).
- **Primary offline data:** synthetic, generated by `data/synth_scene.py` (gold known by construction, scene spec embedded in the PNG) — drives CI, the torch-free agent, and the soft-VQA scorer with no network.

---

## 6. Division of work

Single author (Le Dinh Minh Quan, 23127460). "Division of work" is organised by **workstream** so dependencies and reuse are explicit, not by team. The agentic tools/automation templates assist with scaffolding and reporting.

| Workstream | Scope | Reuse source | New code |
|------------|-------|--------------|----------|
| **Data / vision** | Synthetic-scene generator, self-describing PNG (embedded `scene_spec`), templated QA over 4 types, 10-answer lists, fixtures; image handling (RGB decode/resize/normalize, degenerate-blank detection, no-PIL `SceneImage` carrier) | corpus/loading pattern from P15 | `data/synth_scene.py` (NEW), `vision/image_utils.py` + `SceneImage` (NEW) |
| **VQA core** | Uniform `predict()` wrapper over `ViltForQuestionAnswering` (classification, default) + `BlipForQuestionAnswering`/generative; the 3 notebook entry points; `SceneStubVQA` torch-free stub; fine-tune / zero-shot validation; baselines | HF Trainer + baseline plumbing pattern from P14/P15 | VQA model wrapper (NEW), `SceneStubVQA` (NEW), config/training glue |
| **Metric** | `VQAEval` re-implementation: answer normalization + 10× leave-one-out soft accuracy + per-answer-type + per-question-type aggregation | grading harness skeleton from P14/P15 | `vqa/metric.py` + `vqa/normalize.py` (NEW) — no HF metric exists |
| **Question-type classifier** | 10-rule keyword scheme (§8), coarse→4-bucket mapping, closed color/number lexicons | — | `vqa/qtype.py` (NEW) |
| **Agent** | 5-decision FSM (input gate, type router, model, calibrated abstention, type re-rank), temperature scaling, per-decision trace, optional advisory brain OFF by default | automation/monitoring/grading templates from P14/P15 | `src/mmqa/agent/` 5 decision points (NEW), abstention + re-rank wiring (NEW) |
| **Deploy** | FastAPI (`/ask`, `/ask-scene`), Gradio UI, Docker, HF Space | deploy template from P14/P15 | route handlers (image upload + confidence/abstain flag) + Dockerfile (libGL) |
| **Docs / QA** | Design brief, this plan, README, ethics/privacy, report | P02 docs template | ethics/privacy section (faces/homes/medical/assistive), final report |

**Reuse summary.** Config / logging / registry / autoreport / monitoring / automation / grading / CLI / API templates come from **P15 imgtrans / P14 doctrans**; the HF-Trainer + baseline plumbing pattern is reused for the VQA fine-tune. **NEW for P17** (the first multimodal-vision project): the multimodal VQA model wrapper (ViLT classification + BLIP generative + `SceneStubVQA`), image handling (`vision/image_utils` + the no-PIL `SceneImage` carrier), the synthetic-scene generator, the official VQA-accuracy metric + answer normalization, the question-type classifier, and the type-aware + abstaining five-decision agent.

---

## 7. Ethics, privacy, and robustness (carried into the plan)

These are not an afterthought — they are scheduled work in **M8** and a constraint on **M7**.

- **Privacy.** VQA runs on **user photos**, which routinely contain faces, homes, documents, medical images, and the photos of blind/low-vision users (the VizWiz assistive use case). Defaults: **local processing, no raw-image retention by default**, explicit consent language in the UI. The API returns a confidence and an abstain flag, not just an answer.
- **The tool assists and abstains — it never asserts certainty.** This matters most for **accessibility and medical** contexts, where a confident wrong answer is worse than "unsure". D4 (calibrated abstention) and D5 (type-mismatch) route uncertain cases to `unsure` / `needs_review` rather than emitting a hallucinated answer.
- **Bias — language priors + demographic bias.** VQA models have strong **language priors** (answering 'yes' / '2' / 'white' while ignoring the image) and inherit COCO demographic bias. Mitigation is measurement: always report the **blind question-only baseline** (the gap to the full model = how much the image actually helps) and **per-answer-type accuracy**, so the report is honest about where the model is really just guessing from the question.
- **Robustness work (M8).** Overconfidence, the language-prior shortcut, out-of-vocab answers (the 3129-vocab ceiling), unanswerable questions (VizWiz), image quality / blur, and adversarial questions — mitigated by the calibrated abstention gate (D4), the type-consistency re-rank (D5), the OOV-rate report (R4), and the unanswerable abstention test (R12).

---

## 8. Open items to verify before/at kickoff

1. Re-run `hub_repo_details` on every shipped id (`dandelin/vilt-b32-finetuned-vqa`, `dandelin/vilt-b32-mlm`, `Salesforce/blip-vqa-base`, `microsoft/git-base-vqav2`) to confirm license tags still resolve; re-confirm `google/pix2struct-vqav2-base` is still **404** and keep it out of the registry (R5).
2. Load `dandelin/vilt-b32-finetuned-vqa` config and assert `len(id2label) == 3129`; freeze this as the single answer vocab for the metric, the D5 constraint sets, and the report buckets (R4).
3. Confirm `HuggingFaceM4/VQAv2` still loads with `trust_remote_code=True` on the pinned `datasets` version; if it breaks, fall back to zero-shot ViLT + `Multimodal-Fatima/VQAv2_sample_train` for offline real scoring (R3, R10).
4. Unit-test the VQA-accuracy normalizer on the tricky cases (`2.5` keeps period; `100,000`→`100000`; number-word→digit; drop a/an/the; ~80 contractions) before trusting any reported number (R7).
5. Confirm the `SceneStubVQA` distribution actually triggers D4 (abstain) and D5 (re-rank) on the synthetic eval set — CI must assert every status (`ok`, `ok:reranked`, `abstained:low_confidence`, `abstained:type_mismatch`, `error:*`) appears (R6).
6. Verify the PNG `scene_spec` metadata round-trips through save/reopen and the stub recovers the exact spec (R14).
7. Confirm T4 fits the ViLT fine-tune at batch 16–32 @384px; if not, reduce batch + grad-accum or fall back to zero-shot (R8).
