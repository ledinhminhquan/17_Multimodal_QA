# P17 Multimodal QA (VQA) — Continual Learning & Monitoring

> How the deployed Visual Question Answering system is watched in production, how
> human feedback is captured and turned into training signal, and when and how the
> trainable core (the VQA model) is re-fine-tuned. Specific to P17: a ViLT
> classification core (`dandelin/vilt-b32-finetuned-vqa`, 3129-answer head) wrapped
> by a deterministic 5-decision agent (`src/mmqa/agent/`) that classifies the
> question type, runs the model, **abstains when uncertain**, and **constrains the
> answer to the question type**. The agent and scorer run fully offline against the
> synthetic-scene generator + `SceneStubVQA`, so every monitoring and continual-learning
> mechanism here is exercisable in CI with **no torch, no model download, no network**.

---

## 1. Why VQA needs continual monitoring (and why it is different)

A deployed VQA system degrades in ways a single accuracy number at ship time cannot
catch. P17's risk profile makes monitoring non-optional:

- **Overconfident models.** ViLT/BLIP softmax probabilities skew high and are poorly
  calibrated. The abstention gate (D4) is a *tuned* gate (temperature scaling +
  `tau_conf` + `tau_ent` + `tau_margin`), tuned on a held-out split. Calibration drifts
  as the input distribution shifts, so the thresholds that gave a good
  precision/coverage trade-off at ship time silently stop doing so.
- **Language-prior shortcut.** The model can answer from the question text alone
  ("what color is the banana?" -> "yellow") ignoring the image. The size of this
  shortcut is data-dependent; on a new image domain the blind-prior gap can collapse,
  meaning the model has stopped using the image and we would not know without
  re-running the blind baseline.
- **Fixed 3129-answer ceiling.** The classification head cannot emit any answer outside
  the canonical VQAv2 vocab. As real user questions drift toward new objects/answers,
  the *unreachable-gold* rate climbs — a ceiling no amount of confidence tuning fixes.
- **Sensitive, assistive use.** VQA on user photos touches faces, homes, documents,
  medical images, and blind-user assistance (the VizWiz use case). A confidently wrong
  answer here is a real harm, which is exactly why the headline production metric is
  **answer-when-confident accuracy at a given coverage**, not raw accuracy.

Monitoring therefore tracks four families of signal — **abstention/coverage**,
**accuracy on a labeled slice**, **calibration drift**, and **distribution drift
(answers, questions, image domain)** — and feeds them into explicit **retraining
triggers**.

---

## 2. What the deployed system emits (the raw material)

Every request through `agent.predict(image, question)` already produces a structured,
auditable record. The agent's value-add *is* its instrumentation: each of the five
decision points (D1..D5) emits a status, and D3/D4 emit the full distribution signals.
Monitoring is built by aggregating these records over time — no new model
instrumentation is required.

### 2.1 Per-request log record

Logged via the reused structured logger (`config/logging` from P15/P14), one JSON line
per request:

| Field | Source | Used by |
|---|---|---|
| `request_id`, `ts` | API/agent | join, ordering |
| `question`, `question_len_tokens` | input | question-distribution drift |
| `image_meta` (`w`, `h`, `mode`, `pixel_var`, `bytes`, `blurriness`) | D1 image-quality | image-domain drift; **no raw image retained** |
| `question_type` (`yes_no`/`number`/`color`/`object`/`location`/`person`/`reason`/`other`) | D2 router | per-type slicing, question drift |
| `answer_type` (`yes/no`/`number`/`other`) | D2 -> VQA bucket map | per-answer-type accuracy |
| `topk` `[(answer, prob) x5]` | D3 | answer-distribution drift, vocab growth |
| `p_max`, `margin` (`p1-p2`), `entropy` | D3 | calibration drift, abstention analysis |
| `regime` (`classification`/`generative`) | D3 | which confidence proxy applies |
| `status` (`ok` / `ok:reranked` / `abstained:low_confidence` / `abstained:type_mismatch` / `error:bad_image` / `error:bad_question` / `error:model_failure`) | D1..D5 | abstention rate, coverage, error rate |
| `answer` (or `"unsure"`) | final | feedback join |
| `reranked` (bool), `rerank_from` -> `rerank_to` | D5 | re-rank rate |
| `latency_ms` (total + per-stage `decode`/`tokenize`/`model`/`agent`) | timing | latency monitoring |
| `model_id`, `model_sha`, `vocab_sha`, `agent_cfg_sha` | registry/config | attributing drift to a version |

Privacy: by default **no raw image bytes and no raw user question are persisted**
beyond the in-flight request (see Ethics in §11). The `image_meta` block is a small set
of derived scalars (dimensions, pixel variance, blur estimate) — enough for image-domain
drift, nothing reconstructable. Question text retention is opt-in/consented; drift on
questions can otherwise be tracked over *question_type* + length + leading-token
histograms without storing the raw string.

### 2.2 The monitoring entry point

`monitoring/drift_report.py` (reused skeleton from P15 imgtrans / P14 doctrans, extended
for VQA) consumes a window of these job logs plus an optional labeled slice and emits a
single report (JSON + the autoreport markdown table). It is the one command the
scheduled monitor and the CI smoke test both call:

```
python -m mmqa.monitoring.drift_report \
    --logs runs/prod/2026-06-*/agent.jsonl \
    --baseline artifacts/reference_window.json \
    --labeled-slice data/labeled_slice.parquet \   # optional, see §4
    --out reports/drift_2026-06-27.json
```

Run against the synthetic generator's logs it is fully offline; in CI we feed it a
`SceneStubVQA` run so every metric below has a deterministic test.

---

## 3. Operational monitoring — abstention, coverage, latency

These are computable from job logs alone (no labels), so they run on **100% of
production traffic** every window.

### 3.1 Abstention rate and coverage

The most important operational gauges for an abstaining system:

- **Coverage** = fraction of valid requests answered (`status in {ok, ok:reranked}`).
- **Abstention rate** = fraction abstained, split by cause:
  - `abstained:low_confidence` (D4 — the calibrated gate fired),
  - `abstained:type_mismatch` (D5 — no top-k candidate matched the question type).
- **Error rate** = `error:bad_image` / `error:bad_question` / `error:model_failure`
  (D1/D3 hard fails — these are input/infra problems, not model uncertainty, and are
  tracked separately).
- **Re-rank rate** = fraction of answered requests where D5 changed the answer
  (`ok:reranked`).

Why these matter: a rising **low-confidence abstention rate** is the earliest, label-free
signal of input drift or calibration drift — the model is seeing inputs it is unsure
about. A rising **type-mismatch abstention rate** says the model's top-k no longer
contains a type-consistent answer (new question phrasings the router maps to a type the
model rarely produces, or genuine out-of-domain images). Both are tracked **per
question_type**, because count questions are intrinsically harder (stricter D4 thresholds)
and a uniform abstention number hides which slice is breaking.

Alert bands (configured, not hard-coded — set from the reference window in §6):

- Coverage drops > X points below the reference-window coverage -> warn.
- Either abstention cause rises > Y points -> warn.
- `error:model_failure` rate > Z -> page (infra, not drift).

### 3.2 Confidence-signal monitoring (label-free)

Even without labels, the *distributions* of `p_max`, `margin`, and `entropy` are
monitored window-over-window. These are the inputs to the D4 gate, so a shift here is a
direct precursor to a calibration problem. Tracked as histograms + summary stats
(mean, p10/p50/p90) per question_type, with a population-stability index (PSI) or a
1-D Wasserstein distance against the reference window. A drop in mean `margin` or a rise
in mean `entropy` with **stable abstention thresholds** means more borderline inputs —
expect coverage to fall and, if the threshold is now mis-set, accuracy on answered items
to fall too.

### 3.3 Latency

Per-stage latency from the log (`decode` / `tokenize` / `model` / `agent`). The agent
stages (D1, D2, D4, D5) are deterministic and cheap; the dominant term is D3 (the model
forward pass). Monitored as p50/p95/p99 per route (`/ask` image-upload vs `/ask-scene`
JSON) and per model_id. A p95 regression usually means a hardware/batching change or a
larger model swap (e.g. ViLT -> BLIP-2 on the H100 tier) and is correlated against
`model_sha` so a regression is attributable to a deploy. The synthetic `/ask-scene` path
gives a torch-free latency floor for the agent logic itself.

---

## 4. Accuracy monitoring on a labeled slice

Operational metrics catch *that* something changed; only labels tell us whether
**accuracy** moved. We cannot label all traffic, so we maintain a small, refreshed
**labeled production slice** and score it with the same official VQA-accuracy metric used
at training time (`src/mmqa/metric/`, the re-implemented `VQAEval` — there is no HF
`evaluate-metric/vqa`).

### 4.1 Building the slice

- **Sampling:** stratify by `question_type` and `answer_type` so the three headline
  buckets (yes/no, number, other) and the hard slices (count, color) are each
  represented. Oversample abstained and `ok:reranked` items — those are exactly the
  cases the agent's value-add targets and where labels are most informative.
- **Labeling:** human annotators provide the answer; to compute the *soft* VQA accuracy
  honestly we collect **multiple annotations** (the metric is a 10-annotator
  leave-one-out average of `min(1, matches/3)`). Where 10 annotators per item is too
  costly for a monitoring slice, we collect 3 and report it explicitly as a
  reduced-annotator soft accuracy (a known, documented approximation), never silently
  mixing it with full 10-annotator numbers.
- **Consent/privacy:** only consented images enter the labeled slice; otherwise the slice
  is drawn from the synthetic generator + held-back public VQAv2 validation
  (`lmms-lab/VQAv2`, cc-by-4.0) as a stable control.

### 4.2 What is reported (every window)

Run through `drift_report.py --labeled-slice`, exactly mirroring the training-time
autoreport so numbers are comparable across time:

- **Overall soft VQA accuracy** (x100).
- **Per-answer-type:** `yes/no`, `number`, `other` (the headline 3-bucket breakdown;
  expect yes/no highest, number lowest — a *change in this ordering* is itself a signal).
- **Per-question-type:** the finer buckets (`how many`, `what color is the`, `is there`,
  ...) as an appendix table.
- **Answer-when-confident accuracy @ coverage** — the single most defensible production
  metric for this agent: accuracy computed **only over answered items**, reported
  alongside the coverage it was achieved at. This is what the calibrated-abstention
  value-add is supposed to protect; if coverage holds but answered-accuracy falls, the
  D4 gate has gone stale.
- **Baselines, re-scored on the same slice** so the system's lead over them is tracked,
  not just its absolute score:
  - **Most-common-answer prior** (`yes` overall; `2` for count; `white` for color),
  - **Blind / question-only prior** (answer from `question_type` alone) — the
    language-prior bias gauge; **a shrinking full-model-minus-blind gap is a flashing red
    light** that the model has reverted to the language shortcut on the current traffic,
  - **Zero-shot pretrained ViLT** (`dandelin/vilt-b32-finetuned-vqa`, no fine-tuning) as
    the "did fine-tuning still help?" reference.

If answered-accuracy on the slice drops more than a configured margin, or the blind-prior
gap closes, that is a **retraining trigger** (§6).

---

## 5. Drift detection in detail

`drift_report.py` computes four drift families against a frozen **reference window**
(the distribution at the last good deploy, §6.1). All are label-free except §5.4.

### 5.1 Confidence / calibration drift

The agent assumes a particular relationship between `p_max`/`entropy`/`margin` and
correctness — that is what D4 was calibrated on. Drift here breaks the abstention gate.

- **Distribution drift:** PSI / Wasserstein on `p_max`, `margin`, `entropy` vs reference
  (label-free, §3.2).
- **Calibration drift (needs the labeled slice):** reliability — bin answered items by
  `p_max`, compare empirical accuracy per bin to the bin's mean confidence; track
  **Expected Calibration Error (ECE)** and the fitted **temperature** `T`. If the optimal
  `T` on the fresh slice differs materially from the deployed `T`, the temperature is
  re-fit and the D4 thresholds re-tuned — often a **config change, not a retrain**
  (cheap, ship-fast). Only if recalibration cannot restore the accuracy/coverage
  trade-off do we escalate to re-fine-tuning.

### 5.2 Answer-distribution drift

Histogram of predicted `answer` (and of `topk[0]`) vs the reference. Watch for:

- **Mode collapse onto priors** — a rising share of `yes`/`2`/`white` is the model
  leaning on the language prior; cross-check against the blind-prior gap (§4.2).
- **Vocab-edge pressure** — see §7 (the answer-vocabulary growth problem): rising mass on
  catch-all/`other` answers, or rising `type_mismatch` abstentions on color/count, hints
  that the true answers are drifting outside the 3129-answer head.

### 5.3 Question-distribution drift

The router (D2) and the model were tuned on a particular mix of question types. Track:

- **`question_type` mix** vs reference (e.g. a surge in `reason`/`why` or `count`
  questions — both hard, low-accuracy slices).
- **Leading-token / template histogram** and **question length** distribution.
- **Router fall-through rate** — fraction landing in `object/other` (the open, unconstrained
  bucket). A rising fall-through rate means new phrasings the keyword classifier (§8 of
  the brief) does not recognize; that is a **router-rule update** (deterministic, no
  retrain) flagged as its own action.

### 5.4 Image-domain drift

VQA accuracy is sensitive to image domain (indoor vs outdoor, photo vs synthetic, blurry
assistive-user photos à la VizWiz). Without retaining raw images we track drift on the
derived `image_meta` scalars: resolution distribution, aspect-ratio, `pixel_var` (blank
detection), and the blur/quality estimate. A spike in low-quality / high-blur images
(the VizWiz regime) predicts both lower accuracy and higher abstention and is reported so
the cause of an abstention spike is attributable to *input quality* rather than model
decay. Optionally, a lightweight embedding-drift check (CLIP image-embedding centroid
distance) can run on consented images only.

---

## 6. Retraining triggers and the reference window

### 6.1 The reference window

At each accepted deploy we freeze a **reference window**: the production
metric/distribution snapshot (abstention rate, coverage, `p_max`/`margin`/`entropy`
histograms, answer/question/image-meta histograms, and the labeled-slice accuracy
breakdown) plus the `model_sha`/`vocab_sha`/`agent_cfg_sha`/`T` that produced it. All
drift in §5 is measured against this snapshot, and a new snapshot is taken only when a
re-fine-tune (or a config change) is accepted.

### 6.2 Triggers (ordered cheapest-fix-first)

A trigger does **not** mean "retrain immediately" — it means "run the response ladder".
We escalate only as far as needed:

1. **Recalibrate (config only).** Trigger: calibration drift (ECE up, `T` shifted) with
   answered-accuracy still recoverable by re-tuning thresholds. Action: re-fit `T`,
   re-tune `tau_conf`/`tau_ent`/`tau_margin` per question_type on the fresh slice. No
   model training. Fastest path; most coverage/accuracy regressions are fixed here.
2. **Update deterministic logic (config/code only).** Trigger: router fall-through rate
   up (§5.3) or a new question template; or a new closed-set answer (a color/number form)
   the constraint sets miss. Action: extend the §8 keyword classifier / the D5
   color-lexicon / number map. No model training.
3. **Vocabulary refresh + re-fine-tune.** Trigger: the answer-vocabulary growth problem
   (§7) — unreachable-gold rate or `other`/type-mismatch pressure crosses threshold.
   Action: see §7.2 (re-fine-tune from `dandelin/vilt-b32-mlm` with an expanded head).
4. **Full re-fine-tune of the core.** Trigger: labeled-slice answered-accuracy drop
   beyond margin **that recalibration did not fix**, OR a closing blind-prior gap (model
   reverted to the language shortcut), OR sustained answer/question/image-domain drift.
   Action: §8.

Triggers are **AND-gated against volume** (a drop must persist over N windows / M
requests, not a single noisy window) and **per-slice** (a number-bucket regression can
trigger a targeted retrain even while yes/no holds). Every trigger firing is logged with
the drift evidence that fired it, so a retrain is always justifiable.

### 6.3 Scheduled vs event-driven

- **Scheduled:** `drift_report.py` runs every window (e.g. daily on a rolling 7-day log
  window) via the reused automation harness; a weekly digest goes to the autoreport.
- **Event-driven:** a hard alert (coverage cliff, `error:model_failure` spike, ECE past a
  ceiling) fires immediately. A surge of human corrections on flagged answers (§ feedback)
  is itself an event trigger.

---

## 7. The answer-vocabulary growth problem (P17-specific)

The default core is a **classification** model: a fixed 3129-answer head pulled from
`dandelin/vilt-b32-finetuned-vqa`'s `id2label`. This is a hard ceiling — **any gold
answer outside those 3129 strings is unreachable**, no matter how good the image
features. Over time, real questions drift toward answers the head cannot emit (new
product names, finer colors, larger counts, long-tail objects). This is the most
VQA-specific continual-learning problem in the project.

### 7.1 Detecting vocab pressure

Tracked every window from labeled slice + logs:

- **Unreachable-gold rate** (labeled slice): fraction of items whose normalized gold
  answer is not in the 3129 vocab. This is an accuracy ceiling that retraining the same
  head cannot lift — it can only be lifted by changing the vocab or the regime.
- **`other`-bucket mass and `type_mismatch` abstentions** on closed-type questions
  (color/count): when the right color/number is outside the constrained set, D5 abstains;
  a rising rate on a *closed* type is a strong vocab-pressure signal.
- **Out-of-vocab top-k saturation:** how often the correct answer would have been in a
  larger vocab — estimated on the labeled slice.

All vocab math uses the **same normalization** (§5 of the brief: lowercase, strip
punctuation, drop articles, number-word + contraction maps) as the metric and the
constraint sets — one normalization function, single source of truth, so a "new" answer
is genuinely new and not just a spelling/`white`-vs-`White` artifact.

### 7.2 Responses to vocab growth

1. **Curate the new answers.** From the labeled slice, collect gold answers above a
   frequency floor that are not in the head; review for quality (avoid absorbing noise/
   typos). The frequency floor matters — VQAv2's own 3129 vocab is itself a
   min-frequency cut of the training answers.
2. **Expand the head and re-fine-tune from the MLM base.** Add the curated answers to the
   label space and fine-tune from `dandelin/vilt-b32-mlm` with the **expanded** vocab.
   **Critical invariant:** the metric, the D5 answer-constraint sets (color lexicon,
   number map), and the report buckets must **all** use the new vocab and the same
   `vocab_sha`. A mismatch silently mis-scores. `vocab_sha` is logged per request so we
   can tell which vocab answered a given question.
3. **Generative escape hatch.** If unreachable-gold is structurally high (open-ended
   "other"/`what is`-style traffic), switch the core to the generative tier
   (`Salesforce/blip-vqa-base`, bsd-3, or `microsoft/git-base-vqav2`, mit) which has no
   fixed-vocab ceiling — at the cost of a noisier confidence signal (the D4 gate then
   keys on sequence/token logprob instead of a clean softmax). This is an *architecture*
   trigger, decided when vocab growth stops being a tail problem and becomes the
   distribution.

The agent itself needs **no changes** for any of these — the wrapper's
`predict() -> {topk, p_max, entropy, margin, regime}` contract is unchanged, exactly as
swapping `SceneStubVQA` for `ViltForQuestionAnswering` requires no agent changes.

---

## 8. Periodic re-fine-tuning of the VQA core

The trainable artifact is the VQA model only; everything else (image preprocess,
tokenization, normalization, the 5-decision agent, scoring) is pretrained or
deterministic and is **never** trained. So "retraining" in P17 means re-fine-tuning the
VQA core (and, downstream, re-tuning the deterministic gates against the new model).

### 8.1 Cadence

- **Default:** review on every monitoring window; **re-fine-tune on trigger** (§6), not on
  a blind calendar. A pure time-based cadence (e.g. quarterly) is a fallback floor to
  absorb slow drift even absent a sharp trigger.
- **Recalibration (§6.2 step 1)** runs far more often than full retraining — it is cheap
  and the first line of defense.

### 8.2 Training data assembly

- **Base corpus:** VQAv2 train. The only common mirror with a train split *and* the
  10-annotator schema is `HuggingFaceM4/VQAv2` (license **FLAG**: undeclared on the repo,
  CC-BY-4.0 upstream; a loading-script dataset that needs `trust_remote_code=True`,
  returns a 501 in the Dataset Viewer, may pull COCO images by URL, and is fragile on
  newer `datasets` — pin versions and reserve it strictly for the train split).
- **Production-derived data:** the labeled production slice (§4) and **human corrections
  from feedback** (§9), normalized into the VQAv2 10-annotator schema (synthesize the
  annotator list from the curated correction when only one gold is available, and mark
  these items so their weight is auditable). This is what makes retraining adapt to the
  *deployed* distribution rather than re-learning the static benchmark.
- **Hard-negative emphasis:** oversample the slices monitoring flagged — the drifting
  question_type, the low-accuracy `number`/`reason` buckets, the abstained-and-corrected
  items.
- **Offline path:** the synthetic-scene generator provides a deterministic, network-free
  training/regression set so the *training pipeline itself* is testable in CI with
  `SceneStubVQA` standing in for the model.

### 8.3 Tuning recipe (by GPU tier, from the brief)

- **T4 (free Colab):** fine-tune ViLT (~113M) from `dandelin/vilt-b32-mlm`, batch 16-32 @
  384px — the default path.
- **A100 / L4:** full-speed ViLT fine-tune; or the generative tier
  (`Salesforce/blip-vqa-base` / `blip-vqa-capfilt-large`) if §7.3 pushed us off
  classification.
- **H100:** `Salesforce/blip2-flan-t5-xl` (mit) or `blip2-opt-2.7b` (mit) via bf16 +
  LoRA / Q-Former-only — reserved for an upper-bound refresh, not the routine core.
- License hygiene at every retrain: stay on permissive ids (ViLT apache-2.0, BLIP bsd-3,
  GIT/BLIP-2 mit). **Never** retrain the production core on flagged/non-commercial ids
  (`llava-hf/llava-1.5-7b-hf` llama2, `Qwen/Qwen2.5-VL-3B-Instruct` Qwen-research) — those
  remain flagged upper-bound comparisons only.

### 8.4 Re-tuning the deterministic gates after a retrain

A new model has a new confidence landscape, so a retrain is **not done** until the
agent's gates are re-fit against it:

1. Re-fit temperature `T` and re-tune `tau_conf`/`tau_ent`/`tau_margin` (per
   question_type) on a held-out split — the D4 gate is meaningless on stale thresholds.
2. Refresh D5 constraint sets if the vocab changed (§7.2).
3. Re-confirm the §8.5 acceptance gates.

### 8.5 Validation gate before promotion (shadow -> canary -> promote)

A retrained core ships only if, on a frozen eval split (`lmms-lab/VQAv2` validation,
cc-by-4.0) **and** the fresh labeled slice, it:

- improves (or holds within tolerance) **overall** and **per-answer-type** soft VQA
  accuracy, with **no regression on yes/no** (the easy, high-traffic bucket) untraded for
  a `number` gain;
- improves or holds **answer-when-confident accuracy @ matched coverage** — the headline
  agent metric;
- maintains or widens the **blind-prior gap** (it must still use the image);
- maintains or improves **ECE** after recalibration;
- passes the full offline `SceneStubVQA` regression (all 5 decisions fire, abstention and
  re-rank rates within expected bands).

Promotion is staged: **shadow** (score in parallel, serve old), then **canary** (small
traffic %, watch the §3 operational gauges live), then **full**. On promotion a new
reference window (§6.1) is frozen. Rollback is a `model_sha` switch since the agent
contract is version-independent.

---

## 9. Feedback capture

Human feedback is the bridge from monitoring to retraining. P17's agent is designed to
make feedback cheap and *targeted*, because it already flags exactly which answers are
uncertain.

### 9.1 What we capture

- **Thumbs up / down** on every answered response (API `/ask` response and Gradio UI),
  recorded against the full log record (so a down-vote carries its `p_max`, `topk`,
  `question_type`, `status` — we learn *which kind* of answer fails).
- **Human corrections** of `abstained:*` and `needs_review` items: when the agent said
  `unsure`, an operator (or the user, in assistive settings) can supply the right answer.
  These are gold by construction and the **highest-value training signal** — they directly
  target the model's blind spots.
- **Corrections of `ok:reranked` answers:** confirms or refutes the D5 type-constraint
  value-add — if re-ranked answers get more down-votes than direct answers, the constraint
  logic or the constraint set is wrong.
- **Implicit signals (low weight):** user rephrasing the same question, or abandoning, as
  weak negative signal.

### 9.2 Turning feedback into training data

- A correction is normalized (§5 normalization, single source of truth) and written to a
  **feedback store** in the VQAv2 10-annotator schema; a single correction synthesizes a
  consensus answer, multiple corrections aggregate into a real annotator list.
- Feedback items are **deduped** and **quality-reviewed** before entering a training set
  (guard against adversarial or mistaken corrections — a single user's down-vote is not
  ground truth).
- The feedback store feeds both the **labeled production slice** (§4, for monitoring) and
  the **production-derived training data** (§8.2, for retraining). A high inflow of
  corrections on a particular slice is itself an event-driven retraining trigger (§6.3).
- **Calibration use:** down-voted high-confidence answers and up-voted abstentions are
  exactly the items that reveal mis-calibration; they feed the ECE/temperature re-fit
  (§5.1) even before any model retrain.

### 9.3 Privacy of feedback

Corrections on user images inherit the §11 constraints: only consented images/questions
are retained; otherwise the feedback record keeps the structured signals (the
correction, the type, the confidence) without the raw image. Assistive-use feedback
(blind users) is handled with extra care — the system **assists and flags**, and a human
correction must never be presented back as a certainty.

---

## 10. Putting it together — the monitoring loop

```
   production traffic
         │  agent.predict() -> structured log record (D1..D5 status, topk, p_max/margin/entropy, latency)
         ▼
   ┌─────────────────────────────┐
   │ monitoring/drift_report.py  │  (every window; offline-testable with SceneStubVQA)
   │  • abstention/coverage/err  │
   │  • latency p50/p95/p99      │
   │  • confidence-signal drift  │
   │  • answer / question /      │
   │    image-domain drift       │
   │  • labeled-slice VQA acc +   │
   │    baselines (blind-prior!) │
   └───────────────┬─────────────┘
                   ▼
        triggers (§6, cheapest-fix-first)
           1 recalibrate (config)  ──────────────► re-fit T, re-tune tau_* (no training)
           2 update router / constraint sets ────► deterministic edit (no training)
           3 vocab refresh ──────────────────────► expand head, re-fine-tune from vilt-b32-mlm
           4 full re-fine-tune ──────────────────► §8 recipe
                   │
                   ▼
        feedback store (thumbs, corrections of abstained/flagged)
           → labeled slice  +  production-derived training data
                   │
                   ▼
        retrain core → re-tune gates → shadow/canary/promote (§8.5)
                   │
                   ▼
        freeze new reference window (§6.1) ; loop
```

Every box is exercisable with **no torch and no network** via the synthetic generator +
`SceneStubVQA`, so the whole continual-learning loop has deterministic CI coverage, and
swapping in the real `ViltForQuestionAnswering` wrapper flips it to production with no
changes to the agent or the monitoring code.

---

## 11. Ethics, privacy, and safety constraints on the loop

The continual-learning loop operates under P17's privacy and safety posture; these are
constraints on monitoring/feedback/retraining, not afterthoughts:

- **No raw-image retention by default.** Monitoring runs on derived scalars
  (`image_meta`) and structured signals; raw images and raw questions are retained only
  with explicit consent (e.g. opted-in feedback). This is why image-domain drift uses
  resolution/blur/`pixel_var` and optional consented-only embedding drift, not a stored
  image lake.
- **Local processing where possible**; minimize what leaves the device/session.
- **Assist and abstain, never assert certainty** — especially for accessibility
  (blind-user / VizWiz-style) and any medical/document imagery. The abstention gate is a
  safety feature, and monitoring's job is to keep it well-calibrated. A retrain that
  raises raw accuracy but *worsens* answer-when-confident accuracy or ECE is **rejected**
  by the §8.5 gate.
- **Bias is monitored continuously, not measured once.** The blind / question-only prior
  is re-run on every labeled window: a closing full-model-minus-blind gap means the system
  is leaning on language priors (and COCO demographic bias), and is treated as a
  retraining trigger. Per-answer-type and per-question-type accuracy are always reported
  so a regression hidden inside an aggregate (e.g. `number` collapsing while overall
  holds) surfaces.
- **Feedback is not blindly trusted.** Corrections are deduped and reviewed before
  entering training, so the loop cannot be steered by adversarial or mistaken input.
- **The optional LLM brain (anthropic) stays advisory and OFF by default** — it never
  overrides the deterministic gates and is out of the trained core, so it is not part of
  any retraining or vocab-growth path.
