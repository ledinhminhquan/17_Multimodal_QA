# P17 Multimodal Question Answering (VQA) — Privacy & Robustness

> Scope: this document covers the privacy posture and the robustness/failure-mode engineering for **P17 — Multimodal Question Answering** (package `mmqa`). The trainable core is a Visual Question Answering model (default `dandelin/vilt-b32-finetuned-vqa`, Apache-2.0, classification over the canonical 3129-answer vocab); the production surface is a deterministic 5-decision agent that classifies the question type, runs the model, **abstains when uncertain**, and **constrains** the answer to the question type. P17 is the first multimodal-vision project in this series (P02–P16 were text / document / OCR), which is exactly why privacy and robustness need their own treatment: the input is now a **user photograph**, not a text string.

---

## 1. Why this matters for VQA specifically

A VQA system ingests two things: a **natural-language question** and an **arbitrary image the user supplies**. The image is the sensitive surface. Unlike the earlier text/document projects, a VQA input can contain:

- **Faces and people** — friends, family, children, bystanders, the user themselves.
- **Homes and private spaces** — interiors, addresses, mail, screens, whiteboards.
- **Documents** — IDs, bank statements, prescriptions, letters (even though document-VQA is out of scope as a *task*, nothing stops a user uploading a document *photo*).
- **Medical images** — skin, wounds, scans, medication packaging.
- **Assistive-use photos** — the VizWiz scenario: blind and low-vision users photographing their surroundings to ask "what is this?" / "is the stove on?" / "what does this label say?". These photos are taken in the user's most private contexts (kitchen, bathroom, bedroom, medicine cabinet) and the user often **cannot independently verify** what is in frame.

Because the model is also **overconfident by construction** (softmax VQA classifiers skew high) and exhibits strong **language priors** (answering from the question text while ignoring the image), an un-guarded VQA system is both a privacy risk *and* a reliability risk. This document specifies how P17 mitigates both. The two halves are linked: the same agent gates that protect reliability (D4 abstention, D5 type-consistency) are also what keep the system from asserting confident, wrong, and potentially harmful answers about someone's medicine or surroundings.

---

## 2. Privacy

### 2.1 Data-handling principles

P17 adopts a **minimise, localise, do-not-retain** posture for images.

| Principle | What P17 does | Where enforced |
|---|---|---|
| **Local / on-device processing by default** | The default trainable core is ViLT (~113M, Apache-2.0) and the alternative BLIP-base (~385M). Both are small enough to run **on the user's own machine / Space / container** without sending the image to a third-party API. No image ever has to leave the deployment boundary for the core task. | Model choice (§3 of the design brief); FastAPI/Gradio/Docker self-host. |
| **No raw-image retention by default** | The `POST /ask` endpoint processes the uploaded image in memory and returns `{answer, confidence, abstain}`. The raw image is **not written to disk, not logged, and not cached** unless retention is explicitly turned on. Only the model's image processor sees the pixels (RGB-convert, resize to 384px, normalize), and the resized tensor is discarded after inference. | API layer; config flag `retain_images=False` (default). |
| **The LLM "brain" is OFF by default** | The optional advisory LLM (`anthropic`) in the agent is **disabled by default** and is **advisory only — it never overrides** a decision. With it off, **no image and no question is transmitted to any external LLM provider.** Turning it on is an explicit, documented opt-in that changes the data-flow boundary, and even then it is fed text signals (question, question-type, top-k labels), not the raw photo, unless a vision-LLM path is deliberately enabled. | Agent config (`llm_brain.enabled=False`); §6 of the design brief. |
| **Consent and purpose limitation** | The UI and API docs state plainly what the image is used for (answering the one question) and that it is not stored, sold, or used for training. For any deployment that *does* retain images (e.g. to build a feedback set), consent must be explicit and the retention surfaced in the UI. | UI copy; ethics note; deployment checklist. |
| **Logs carry derived signals, not pixels** | The structured per-decision trace (D1..D5 status, question type, `p_max`, entropy, margin, final status) is safe to log and is what monitoring/autoreport consume. The **image bytes and any embedded EXIF are never logged.** Question text logging is opt-in and should be reviewed (a question can itself be sensitive, e.g. "is this mole cancerous?"). | Logging template; monitoring. |

### 2.2 Metadata and EXIF

Photographs carry metadata the user may not realise is there: **GPS coordinates, device id, timestamp, orientation**. P17:

- Uses only the decoded RGB pixel array for inference; the image processor does not consume EXIF.
- **Strips/ignores EXIF** on ingest — nothing downstream reads GPS or device fields, and nothing persists them.
- Note the asymmetry with the offline path: the **synthetic generator deliberately embeds** a `scene_spec` JSON in PNG `tEXt` metadata so the `SceneStubVQA` is self-describing for tests. That is a *test-fixture* mechanism on synthetic images only and must **never** be confused with reading metadata from real user uploads — real uploads are treated as opaque pixels and their metadata is discarded.

### 2.3 Accessibility / assistive use is a privacy hot-spot

The VizWiz-style use case (blind users) deserves a dedicated callout because it concentrates several risks:

- The user **cannot see** the photo, so they cannot self-censor what is in frame (a face, an address, a document may be captured unintentionally).
- The photos are taken in intimate settings and the user is **dependent** on the answer.
- The user **cannot verify** a wrong answer, so the cost of a confident error is higher than for a sighted user.

P17's response: keep processing local, retain nothing, and — critically — **abstain rather than guess** (see §3.1). For accessibility use the tool must say "I'm not sure" instead of asserting a confident wrong answer, because the user has no fallback. This is a design constraint, not a nicety.

### 2.4 What P17 deliberately does **not** do

- Does not perform face recognition, identity inference, or any demographic classification of people in the image. VQA answers the user's question; it does not profile the subject.
- Does not phone home, send telemetry containing image content, or upload images for "improvement" without explicit opt-in.
- Does not assert medical, legal, or safety-critical conclusions as fact (see robustness §3.8).

---

## 3. Robustness

VQA has a well-characterised set of failure modes. P17 maps each to a concrete mitigation, most of which live in the deterministic agent (D1–D5) and the evaluation harness, and are therefore **testable offline** with `SceneStubVQA` (no torch, no network).

### 3.1 Overconfidence → calibrated abstention (D4)

**Failure.** VQA classifiers are notoriously **overconfident**: the softmax over the 3129-answer head skews high, so raw `argmax` confidence is not a trustworthy "should I answer?" signal. Thresholding on raw `p_max` alone is unsafe.

**Mitigation — D4 calibrated abstention gate.** The agent does not trust a single number. It requires **all** of:

- `p_max >= tau_conf` (e.g. 0.30) — enough mass on the top answer,
- Shannon entropy `H <= tau_ent` (e.g. 1.5 nats) over the truncated/renormalized top-k — the distribution is peaked, not flat,
- top1–top2 **margin** `(p1 - p2) >= tau_margin` — a clear winner, not a coin-flip between two answers.

Thresholds are **calibrated** (temperature-scaled softmax) on a held-out split, optionally **per question type** (count is harder → stricter; yes/no can tolerate a higher tau), not hard-coded constants. Failing any condition yields `answer="unsure"`, `status="abstained:low_confidence"`, and a `needs_review` flag. This trades a little coverage for a large precision/reliability gain and is the system's **single most defensible headline metric**: *answer-when-confident accuracy and abstention rate vs. a raw-argmax baseline.*

**Verification.** The `SceneStubVQA` returns realistic (not one-hot) distributions and has a `noise`/`difficulty` knob that deliberately lowers `p_max` on a fraction of items so D4 actually fires; unit tests assert the abstention path triggers and that abstained items are excluded from coverage.

### 3.2 Language-prior shortcut → blind baseline + image-dependent gates

**Failure.** Models answer from the **question text alone**, ignoring the image: "what color is the banana?" → "yellow" regardless of what is shown; "how many…?" → "2"; "is there…?" → "yes". This is the classic VQA shortcut, and VQAv2 was specifically constructed (balanced complementary pairs) to depress it — but it is **not zero**.

**Mitigation.**

1. **Report the blind / question-only baseline** (a classifier on the question text with **no image**). The **gap between the full model and the blind baseline is the real signal that the image is being used.** Also report the "most-common-answer per-type prior" (`yes` / `2` / `white`) and the "always-yes" prior. If the full model barely beats the blind baseline on a question type, that type is being answered by language prior, not vision, and the report makes that visible.
2. **Per-answer-type and per-question-type accuracy** (yes/no, number, other; plus the 65 VQAv2 question-type buckets) so prior-driven inflation cannot hide inside an aggregate number.
3. The agent's gates operate on the **model's own distribution**, so a confidently-wrong prior answer is still subject to D4 (is it actually peaked?) and D5 (is it type-consistent?).

### 3.3 Out-of-vocabulary answers → known ceiling + generative escape hatch

**Failure.** The classification head is **fixed at 3129 answers**. Any gold answer outside that vocab is **unreachable** by the classification model — a hard accuracy ceiling, independent of how well the model "understands" the image.

**Mitigation.**

- This ceiling is **documented and reported**, not hidden: the answer-constraint sets (D5), the metric buckets, and the report all use the **same** 3129 vocab pulled from `dandelin/vilt-b32-finetuned-vqa` config `id2label`, so the OOV ceiling is a stated property of the classification regime.
- If a custom vocab is trained (fine-tune from `dandelin/vilt-b32-mlm`), the **metric, the D5 constraint sets, and the report buckets must use that same vocab** — no silent mismatch.
- The **generative regime is the escape hatch**: `Salesforce/blip-vqa-base` (BSD-3) / `blip-vqa-capfilt-large`, or `microsoft/git-base-vqav2` (MIT), produce open-vocabulary free-text answers and avoid the ceiling, at the cost of a cleaner confidence signal (logprob/beam-score instead of a softmax). Generative outputs go through the **same canonical VQA normalization** before exact-match scoring, so the comparison is apples-to-apples.

### 3.4 Unanswerable questions → abstention as the correct behaviour

**Failure.** Some `(image, question)` pairs have **no answer**: the object asked about is not in the image, the photo is too blurry to tell, or the question presupposes something false ("what color is the dog?" when there is no dog). A naïve VQA model still emits a confident answer.

**Mitigation.**

- **D4 abstention is the designed response**: when the true answer is absent or ambiguous, a well-behaved model's distribution is flatter / less peaked, and the entropy + margin conditions push the agent to `"unsure"` rather than a fabricated answer.
- **`lmms-lab/VizWiz-VQA`** carries an explicit **`unanswerable`** label and is the **ideal abstention test set** for P17 — measure how often the agent correctly abstains on items humans marked unanswerable. (License is undeclared on the repo → **FLAG**; CC-BY-4.0 upstream; verify before commercial use.)
- The synthetic harness exercises this too: `color` questions are only emitted when the target shape is **unique** (so gold is unambiguous), and the difficulty knob injects flat/ambiguous distributions so the unanswerable→abstain path is unit-tested offline.

### 3.5 Image blur / low quality → input gate (D1)

**Failure.** Corrupt, blank, all-one-color, truncated, or severely degraded images produce garbage answers if fed to the model.

**Mitigation — D1 input gate, before any model call.** The image must:

- decode to a valid, non-zero **RGB** array,
- not be **degenerate-blank** (pixel-variance above an epsilon),

and pass with PIL + a variance check — **no torch**. A corrupt/blank image **halts with `status="error:bad_image"` and never reaches the model.** D1 is the cheapest possible defence: it rejects pathological inputs deterministically and offline. (Genuinely-in-focus-but-hard images that pass D1 but the model can't read are caught downstream by D4 abstention.)

### 3.6 Malformed / non-question text → input gate (D1)

**Failure.** Empty strings, non-interrogative text, code, or absurdly long inputs.

**Mitigation — the question half of D1.** The question must be non-empty after strip, contain ≥1 alphabetic token, be within `[1, 128]` tokens, and be **interrogative** (ends in `?` OR starts with a wh-/aux- word: `what / which / how / where / is / are / does / do / can`). Failures halt with `status="error:bad_question"`. Pure regex, no torch.

### 3.7 Adversarial / leading / presuppositional questions → type-consistency (D5) + abstention (D4)

**Failure.** Leading questions ("how many **red** squares?" when there are none red; "what color is the **car**?" when there is no car) try to coax the model into agreeing with a false premise. Type-mismatched fluent answers are another mode: the argmax to "how many?" comes back `"cat"`.

**Mitigation — D5 type-consistency gate + re-rank.** After D2 classifies the question type (`yes_no` / `count` / `color` / `object/other`), D5 forces the answer to be **type-consistent**:

- `yes_no` → answer must be in `{yes, no}`,
- `count` → a number (digit / number-word, normalized),
- `color` → a color from the closed lexicon,
- `object/other` → open vocab (D5 is a pass-through no-op).

D5 **re-ranks within the top-k**: if `top1` is type-wrong, it scans for the highest-prob entry that **is** consistent (e.g. count question, `top1="cat"(0.4)`, `top2="3"(0.3)` → return `"3"`, `status="ok:reranked"`). If **no** top-k entry matches the type, the agent **abstains** with `status="abstained:type_mismatch"` rather than emit a fluent-but-wrong answer. Combined with D4, a leading question whose true answer is "none"/"0"/"no" tends to either surface the low-confidence abstention or get re-ranked to the type-correct candidate, instead of confidently affirming the false premise.

### 3.8 Demographic and dataset bias → measure, report, and never assert certainty

**Failure.** VQA models inherit bias from their training data. COCO/VQAv2 scenes over-represent certain contexts, objects, and people, and the model can carry **demographic bias** plus the **language priors** of §3.2 (e.g. defaulting answers about people based on dataset frequency rather than the image).

**Mitigation.**

- **Quantify, don't assume.** Report the **blind question-only baseline** (language-prior bias), the **per-answer-type** and **per-question-type** accuracy breakdowns, and the abstention/coverage rates. Bias that lives in a prior shows up as a small full-model-minus-blind gap on the affected types.
- **The tool assists and abstains; it never asserts certainty.** Especially for accessibility and any health-adjacent question, the system surfaces confidence and the `abstain`/`needs_review` flag and says "unsure" rather than fabricating. P17 does **not** make medical, legal, or identity claims as fact — those are out of scope and the UI copy says so.
- **Permissive, declared-license data preferred** for anything that must be defensible: `lmms-lab/VQAv2` (CC-BY-4.0), `facebook/textvqa` (CC-BY-4.0), `lmms-lab/GQA` (MIT). License-unconfirmed mirrors (`HuggingFaceM4/VQAv2`, `merve/vqav2-small`, `Multimodal-Fatima/*`, `lmms-lab/OK-VQA`, `lmms-lab/VizWiz-VQA`) are **FLAGGED**, and the non-commercial `thangduong0509/daquar_vqa` (CC-BY-NC-SA-4.0) is **avoided outright**. A biased or mislicensed source is a robustness liability, not just a legal one.

### 3.9 Heavy / flaky dependencies → torch-free offline path

**Failure.** `torch` + `transformers` + a vision stack + multi-GB parquet/COCO images are heavy and flaky in CI and Colab-free environments — a robustness problem for the *system around* the model.

**Mitigation.** The **synthetic-scene generator + `SceneStubVQA`** (mirroring P15's OCR `SeedEngine`) run the **entire 5-point agent and the scorer with zero torch and no network**, deterministically from a seed. Real-model paths are gated behind an availability check. Swapping `SceneStubVQA` for the real `ViltForQuestionAnswering` wrapper (same `predict(image, question) -> {topk, p_max, entropy, margin}` signature) flips to production **with no agent changes**. This means privacy and robustness behaviour is verified in CI on every run, independent of the heavy stack.

---

## 4. Failure-mode → mitigation summary

| # | Failure mode | Primary mitigation | Where | Offline-testable |
|---|---|---|---|---|
| 1 | Model overconfidence | Calibrated abstention (`p_max` + entropy + margin, temperature-scaled, per-type) | **D4** | Yes (stub difficulty knob) |
| 2 | Language-prior shortcut | Blind question-only baseline + per-type accuracy + image-dependent gates | Eval + D4/D5 | Yes |
| 3 | Out-of-vocab answers | 3129-vocab ceiling documented/reported; generative escape hatch (BLIP/GIT) | Metric + model | Partial |
| 4 | Unanswerable questions | Abstain (flat/low-margin → "unsure"); VizWiz `unanswerable` eval | **D4** | Yes |
| 5 | Image blur / corrupt / blank | Input gate: RGB-decode + pixel-variance check | **D1** | Yes |
| 6 | Empty / non-question text | Input gate: interrogative + length + alpha-token checks | **D1** | Yes |
| 7 | Adversarial / leading / type-wrong | Type-consistency gate + top-k re-rank, else abstain | **D5** (+D4) | Yes |
| 8 | Demographic / dataset bias | Measure (blind baseline, per-type), assist-not-assert, prefer clean-license data | Eval + ethics | Partial |
| 9 | Heavy / flaky deps | Torch-free `SceneStubVQA` + synthetic scenes; availability-gated real path | §7 harness | Yes |

| # | Privacy risk | Primary mitigation | Default state |
|---|---|---|---|
| P1 | Image leaves the device | Local on-device core (ViLT/BLIP), self-host | Local |
| P2 | Raw image retained / logged | In-memory only, `retain_images=False`, derived-signal logs | No retention |
| P3 | Image/question sent to external LLM | LLM brain **OFF**, advisory-only, never overrides | Off |
| P4 | EXIF / GPS leakage | EXIF ignored & discarded; only pixels used | Stripped |
| P5 | Assistive user can't verify a wrong answer | Abstain over guess; surface confidence + `needs_review` | Abstain-first |
| P6 | Sensitive question content in logs | Question logging opt-in & reviewed | Off / reviewed |

---

## 5. Defaults, config flags, and the deployment checklist

**Safe defaults (shipped):**

- `llm_brain.enabled = False` (no external LLM; advisory-only when on)
- `retain_images = False` (no raw-image persistence)
- `log_questions = False` (questions can be sensitive)
- abstention gate **on**, thresholds calibrated per split/type
- D1 input gate **on** (image-quality + question-validity)
- offline `SceneStubVQA` path used for CI / torch-free runs

**Deployment checklist (before exposing `POST /ask` publicly):**

1. Confirm the image is processed in-memory and not written to disk or logs.
2. Confirm EXIF/GPS is discarded.
3. Confirm the LLM brain is off (or that opt-in consent + data-flow disclosure exist if on).
4. Confirm consent / privacy copy is shown in the Gradio UI and API docs.
5. Confirm abstention and type-consistency gates are active and thresholds were calibrated on a held-out split — not left at placeholder constants.
6. For accessibility/medical-adjacent deployments, confirm the UI surfaces the `abstain`/`needs_review` flag and never presents a low-confidence answer as certain.
7. Confirm the dataset(s) in use are license-clean for the deployment context (prefer CC-BY-4.0 / MIT / Apache; FLAG or avoid undeclared/non-commercial mirrors).
8. Confirm Docker image ships `libGL` (Pillow dependency) so image decode does not silently fail at the boundary.

---

## 6. One-line summary

P17 keeps user images **local, unretained, and unsent** (LLM brain off by default), and makes the VQA model **honest about uncertainty** — the deterministic agent (D1 input gate, D4 calibrated abstention, D5 type-consistency re-rank) turns an overconfident, language-prior-prone, fixed-vocab classifier into a system that **answers when confident, abstains when not, and never asserts certainty** — all verifiable offline with the torch-free `SceneStubVQA` harness.
