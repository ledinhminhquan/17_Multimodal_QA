# P17 Multimodal QA (VQA) — Agent Architecture

> **Package:** `mmqa` · **Module:** `src/mmqa/agent/` · **Author:** Le Dinh Minh Quan (23127460)
> **Scope:** the deterministic agent that wraps the frozen VQA model. This is the mandatory *agentic* component of P17.

---

## 1. Why an agent at all

A stock VQA model — `dandelin/vilt-b32-finetuned-vqa` (ViLT, classification head over the canonical 3129-answer VQAv2 vocabulary) or a generative `Salesforce/blip-vqa-base` — does exactly one thing: it takes an `(image, question)` pair and returns the **raw argmax** answer. That is not enough for production:

- VQA classifiers are **notoriously overconfident**. The softmax over 3129 answers skews high even when the model is wrong, so a naive "trust the top label" system emits confident hallucinations.
- The argmax is frequently **fluent but type-wrong**: a model can answer `"cat"` to *"how many?"* or `"yes"` to *"what color is the car?"*. The string is a valid English answer, just not a valid answer *of the right kind*.
- A bare model **cannot say "I don't know."** For accessibility and assistive use (VizWiz-style blind-user photos, medical images), silently asserting a wrong answer is worse than abstaining.

The P17 agent is a **deterministic finite-state machine (FSM)** with **five decision points**. Only one of them (D3) touches the model; the other four are pure deterministic logic over the model's own top-k distribution and the question text. Net result, with **zero extra training**:

1. **Calibrated abstention** — the system says `"unsure"` + `needs_review` instead of guessing when the distribution is flat/uncertain.
2. **Type-aware answer constraint + re-rank** — the answer is forced to be consistent with the question type (`yes/no` → `{yes,no}`, count → a number, color → a color word), re-ranking *within* the top-k to the best type-consistent candidate.

Both behaviours are **auditable** (every decision emits a `ToolTrace` line) and **offline-testable** (the whole FSM runs with no torch and no network against `SceneStubVQA`).

---

## 2. The finite-state machine

The agent is a strictly forward FSM: `ingest → classify → answer → calibrate → constrain`. There are no loops and no backward edges — every path either reaches a terminal `ok*` / `abstained:*` state or short-circuits to an `error:*` halt. This is what makes it auditable and deterministic.

```
                 (image, question)
                        │
                ┌───────▼────────┐
                │  D1  INGEST     │  input gate (PIL + regex, no torch)
                │  valid image?   │──bad image──────► HALT  status = error:bad_image
                │  valid question?│──bad question───► HALT  status = error:bad_question
                └───────┬────────┘
                        │ ok
                ┌───────▼────────┐
                │  D2  CLASSIFY   │  keyword question-type router
                │  yes_no/count/  │  (defines the ALLOWED answer set for D5)
                │  color/other    │
                └───────┬────────┘
                        │ question_type
                ┌───────▼────────┐
                │  D3  ANSWER     │  run VQA — the ONLY model call
                │  ViLT logits |  │──exception──────► HALT  status = error:model_failure
                │  SceneStubVQA   │  → topk(k=5), p_max, margin, entropy, regime
                └───────┬────────┘
                        │ distribution attached
                ┌───────▼────────┐
                │  D4  CALIBRATE  │  calibrated abstention gate
                │  confident?     │──no─────────────► answer = "unsure"
                └───────┬────────┘                     status = abstained:low_confidence
                        │ yes                          needs_review = true
                ┌───────▼────────┐
                │  D5  CONSTRAIN  │  type-consistency gate + re-rank within top-k
                │  type-match in  │──no top-k match──► answer = "unsure"
                │  top-k?         │                     status = abstained:type_mismatch
                └───────┬────────┘
                        │
          return { answer, status ∈ {ok, ok:reranked}, question_type,
                   answer_type, confidence, topk, needs_review, trace }
```

### State / status vocabulary

| Status | Stage that emits it | Meaning |
|---|---|---|
| `error:bad_image` | D1 | Image missing, undecodable, or degenerate-blank. Hard halt before the model. |
| `error:bad_question` | D1 | Question empty / non-interrogative / out of length bounds. Hard halt. |
| `error:model_failure` | D3 | The model wrapper raised at inference time. Fail-soft halt. |
| `abstained:low_confidence` | D4 | Distribution too flat/uncertain; answer suppressed to `"unsure"`. |
| `abstained:type_mismatch` | D5 | No top-k candidate is consistent with the question type. |
| `ok` | D5 | `topk[0]` was already type-consistent and confident. |
| `ok:reranked` | D5 | A lower-ranked top-k candidate was promoted to satisfy the type constraint. |

`needs_review = true` is attached to every `abstained:*` outcome so a downstream queue / human-in-the-loop can pick them up. `error:*` halts are operational failures, not abstentions.

---

## 3. AgentConfig — the tunable surface

All thresholds live in a single dataclass (reused config pattern from P02/P15), so the agent has **no magic numbers** inline. Defaults below; per-question-type overrides are supported because **count is harder than yes/no** and deserves stricter gating.

```python
@dataclass
class AgentConfig:
    # D1 input gate
    min_question_tokens: int = 1
    max_question_tokens: int = 128
    blank_pixel_var_eps: float = 1e-4     # below this variance -> degenerate-blank image

    # D3 distribution
    top_k: int = 5
    temperature: float = 1.0              # softmax temperature (calibration, >1 softens)

    # D4 calibrated abstention (global defaults)
    tau_conf: float = 0.30                # min p_max
    tau_margin: float = 0.10              # min (p1 - p2)
    tau_entropy: float = 1.50             # max Shannon entropy H (nats)

    # D4 per-question-type overrides (count is hardest -> strictest)
    tau_conf_by_type: dict = field(default_factory=lambda: {
        "yes_no": 0.40,   # binary -> a confident model should be well over 0.5
        "count":  0.45,   # counting is brittle -> demand more
        "color":  0.30,
        "other":  0.25,
    })

    # D5 type lexicons (closed sets the answer must fall into)
    color_lexicon: frozenset = field(default_factory=lambda: frozenset({
        "white","black","red","blue","green","yellow","brown",
        "orange","pink","purple","gray","grey","silver","gold","tan",
    }))
    count_lexicon: frozenset = field(default_factory=lambda: frozenset(
        {str(i) for i in range(0, 21)} | {"none"}
    ))

    # Optional advisory LLM brain (OFF by default)
    use_llm_brain: bool = False
    llm_provider: str = "anthropic"
```

> **Calibration, not constants.** The thresholds are **tuned on a held-out split with temperature scaling** of the softmax — *not* hand-picked once. ViLT/BLIP softmaxes are miscalibrated, so `temperature` is fit so that `p_max` means something before `tau_conf`/`tau_entropy`/`tau_margin` are swept for the best coverage/precision trade. The values above are sane starting points, not gospel.

---

## 4. The five decision points

### D1 — Ingest (input gate)

**Module:** `agent/ingest.py` · **Touches the model?** No (PIL + regex only).

**Gates on:** validity of `(image, question)` *before any model work is paid for*.

- **Image** must decode to a valid, nonzero **RGB** array (RGBA/L/P are converted) and must not be **degenerate-blank** — pixel variance must exceed `blank_pixel_var_eps`. A solid-white or all-zero buffer is treated as "no usable image". (For the synthetic path, the `SceneImage` no-PIL carrier exposes the same variance signal so D1 fires identically offline.)
- **Question** must be non-empty after `strip()`, contain ≥1 alphabetic token, have a token length in `[min_question_tokens, max_question_tokens]` = `[1, 128]`, and be **interrogative**: it ends in `?` **OR** starts with a wh-/aux- word (`what/which/how/where/when/who/why/is/are/does/do/can/has/was/were`).

**Branches:**

| Condition | Next | Status |
|---|---|---|
| valid image **AND** valid question | → D2 | (continue) |
| image corrupt / blank / undecodable | **HALT** | `error:bad_image` |
| question empty / too long / not interrogative | **HALT** | `error:bad_question` |

This is a **hard fail** — a D1 reject never reaches the model. Catching bad input here is also a privacy win: a blank or garbage upload never gets embedded/processed.

---

### D2 — Classify (question-type router)

**Module:** `agent/classify.py` · **Touches the model?** No.

**Gates on:** a deterministic lexical mapping of the question to a coarse type. This is the **value-add anchor**: the type chosen here defines the **allowed answer set** that D5 will enforce.

The coarse router collapses the full 10-rule keyword table (see §8 of the design brief) into four buckets. Rules are evaluated **top-down, first match wins** — ordering is critical (`how many` must beat `how`; `what color` must beat generic `what`).

| Bucket | Trigger (lowercased, leading tokens) | Allowed answer set at D5 | VQA `answer_type` |
|---|---|---|---|
| `yes_no` | starts `is/are/was/were/do/does/did/has/have/had/can/could/will/would/should`, or `is there`/`are there`, or contains `or not` | `{yes, no}` | `yes/no` |
| `count` | `how many`, `how much`, `number of`, `count`, `what number` | `count_lexicon` = digits `0..20` + `none` (number-words mapped to digits) | `number` |
| `color` | `what color`, `which color`, `what colour`, `color of` | `color_lexicon` (closed color set) | `other` |
| `object`/`other` | anything else (catch-all `what`, `where`, `who`, `why`, `when`, `which`) | **open vocab** → D5 is a pass-through no-op | `other` |

**Branches:** exactly one bucket is always selected (`object/other` is the default). The bucket plus its allowed set are attached to the trace and carried to D5. No halt is possible here.

> **Reporting mapping:** `yes_no → yes/no`, `count → number`, everything else `→ other` — exactly the three VQA answer-type buckets the metric reports.

---

### D3 — Answer (run VQA, capture the full distribution)

**Module:** `agent/answer.py` → calls `mmqa.vision`/model wrapper · **Touches the model?** **Yes — this is the only model call.**

**Gates on:** running the VQA core and capturing the **full distribution, not just argmax**. The wrapper exposes a uniform signature regardless of regime:

```python
predict(image, question) -> {
    "topk":    [(answer, prob), ...],   # k = top_k = 5
    "p_max":   p1,                      # prob of rank-1 candidate
    "margin":  p1 - p2,                 # top1-top2 gap
    "entropy": H,                       # Shannon entropy over renormalized top-k
    "regime":  "classification" | "generative",
}
```

- **Classification (default):** `ViltForQuestionAnswering → outputs.logits`, temperature-scaled softmax, `topk(5)`. This is the regime the agent is designed against — a clean per-answer probability for D4.
- **Generative (alternative):** `BlipForQuestionAnswering.generate()`; confidence proxy is the sequence/token logprob, mapped into the same `topk`/`p_max` fields.
- **Offline (CI / torch-free):** `SceneStubVQA` reads the `scene_spec` embedded in the PNG metadata, computes the true answer, and returns a **realistic (non one-hot) distribution** — ~0.7–0.9 mass on the correct answer, the remainder over plausible type-consistent distractors (other colors, adjacent counts). A `difficulty`/`noise` knob deliberately lowers `p_max` or injects a type-wrong top1 on a fraction of items so **D4 and D5 actually fire** in tests.

`entropy` is computed over the renormalized top-k:  `H = -Σ pᵢ · ln pᵢ`.

**Branches:**

| Condition | Next | Status |
|---|---|---|
| inference succeeds | → D4 (distribution attached) | (continue) |
| wrapper raises at inference | **HALT** | `error:model_failure` |

Because the stub and the real wrapper share the exact `predict()` signature, **swapping `SceneStubVQA` for `ViltForQuestionAnswering` flips the system to production with no agent changes**.

---

### D4 — Calibrate (calibrated abstention gate)

**Module:** `agent/calibrate.py` · **Touches the model?** No (operates on D3's distribution).

**Gates on:** whether the model is confident enough to answer **at all**. Because VQA classifiers are overconfident, a *single* signal is unreliable — D4 requires **three** signals to agree, all measured on the temperature-scaled distribution:

| Signal | Test | Default threshold | Intuition |
|---|---|---|---|
| max probability | `p_max ≥ tau_conf` | `0.30` (per-type: yes_no `0.40`, count `0.45`) | the top answer must carry real mass |
| top1–top2 margin | `margin ≥ tau_margin` | `0.10` | top-1 must clearly beat its nearest rival |
| entropy | `entropy ≤ tau_entropy` | `1.50` nats | the distribution must not be flat/diffuse |

The applicable `tau_conf` is the **per-question-type override** from D2's bucket when present (count is hardest → `0.45`), else the global `tau_conf`.

**Branches:**

| Condition | Next | Status |
|---|---|---|
| `p_max ≥ tau_conf` **AND** `margin ≥ tau_margin` **AND** `entropy ≤ tau_entropy` | → D5 (CONFIDENT) | (continue) |
| **any** test fails | abstain: `answer = "unsure"`, `needs_review = true` | `abstained:low_confidence` |

This gate is the single most defensible headline metric for P17: **answer-when-confident accuracy** and **abstention rate** versus a raw-argmax baseline. Trading a little coverage for a large precision/reliability gain is the whole point — the system declines instead of hallucinating.

---

### D5 — Constrain (type-consistency gate + re-rank)

**Module:** `agent/constrain.py` · **Touches the model?** No (re-ranks D3's top-k).

**Gates on:** forcing the emitted answer to be consistent with D2's question type, by re-ranking **within** the top-k — never inventing an answer outside the model's own candidates.

Logic:

1. Look up the **allowed set** for the D2 bucket (`{yes,no}`, `count_lexicon`, `color_lexicon`, or open).
2. If the bucket is `object/other` → **pass-through**, return `topk[0]` (no constraint to apply).
3. Else normalize candidate form first (number-word ↔ digit for counts; canonical color spelling) and check membership:
   - if `topk[0]` is already in the allowed set → return it.
   - else scan `topk[1..k]` for the **highest-probability** candidate that *is* in the allowed set and promote it.
   - if **no** top-k entry is in the allowed set → abstain.

**Branches:**

| Condition | Status |
|---|---|
| `topk[0]` already type-consistent | `ok` |
| a lower-ranked top-k candidate is type-consistent (promoted) | `ok:reranked` |
| no top-k candidate matches the type | `abstained:type_mismatch` (+ `needs_review`) |

**Worked re-rank:** count question *"how many squares?"*, model returns `topk = [("cat", 0.40), ("3", 0.30), ("dog", 0.15), ...]`. `"cat"` ∉ `count_lexicon`; scan finds `"3"` (highest-prob count) → promote. Output `answer="3"`, `status="ok:reranked"`. The same frozen model now scores correct where raw argmax scored zero.

The final emitted answer is run through the **canonical VQA answer-normalization** (lowercase, strip punctuation, drop articles, number-word + contraction mapping) before scoring — the *same* normalization applied to all 10 gold annotators.

---

## 5. Decision table (consolidated)

| D | Stage | Signal it gates on | Pass → | Fail / branch → |
|---|---|---|---|---|
| D1 | Ingest | valid RGB image (var > eps) **and** non-empty interrogative question (≤128 tok) | D2 | `error:bad_image` / `error:bad_question` (HALT) |
| D2 | Classify | leading-keyword question type (first match wins) | D3 with bucket + allowed set | n/a (always routes; default `object/other`) |
| D3 | Answer | model `predict()` → `topk, p_max, margin, entropy` | D4 with distribution | `error:model_failure` (HALT) |
| D4 | Calibrate | `p_max ≥ τ_conf` ∧ `margin ≥ τ_margin` ∧ `entropy ≤ τ_ent` | D5 | `abstained:low_confidence` → `"unsure"` |
| D5 | Constrain | `topk[i]` ∈ allowed set for the D2 type | `ok` / `ok:reranked` | `abstained:type_mismatch` → `"unsure"` |

---

## 6. ToolTrace — the audit log

Every decision point appends one structured record to a `ToolTrace` (reuses the P02/P15 structured-logging pattern). The trace is **the audit trail** and is returned in the response alongside the answer, so any outcome can be replayed and explained offline.

Each entry:

```json
{
  "stage": "D4",
  "name": "calibrate",
  "inputs":  {"p_max": 0.31, "margin": 0.06, "entropy": 1.42,
              "tau_conf": 0.45, "tau_margin": 0.10, "tau_entropy": 1.50,
              "question_type": "count"},
  "decision": "abstain",
  "reason":  "margin 0.06 < tau_margin 0.10",
  "status":  "abstained:low_confidence",
  "latency_ms": 0.3
}
```

Trace properties:

- **Complete** — D1…D5 each emit exactly one entry (halts emit their entry then stop). The sequence of `stage`/`decision` fields is the full execution path.
- **Reason-bearing** — every gate records *which* threshold/test produced the branch (e.g. `"margin 0.06 < tau_margin 0.10"`), so an abstention is never a black box.
- **Deterministic & replayable** — given the same `(image, question)`, seed, and `AgentConfig`, the trace is byte-identical. CI asserts on the trace, not just the final answer (e.g. "this difficulty=0.6 scene must trigger D5 re-rank").
- **Monitorable** — the autoreport/monitoring templates aggregate traces into abstention rate, re-rank rate, per-type coverage, and latency.

---

## 7. Optional LLM brain (advisory, OFF by default)

The agent supports an **optional** LLM "brain" (`use_llm_brain=False` by default; provider `anthropic`). It is strictly **advisory** and is wired so it **can never override the deterministic gates**:

- It is only consulted in *ambiguous* cases — typically a borderline D4 (near-threshold confidence) or a D5 with no clean type match — to suggest a tie-break among the model's existing top-k candidates or to rephrase an ambiguous question for re-classification.
- It **cannot introduce an answer outside the model's top-k**, cannot relax a threshold, and cannot turn an abstention into an assertion. If the LLM is unavailable, errors, or disagrees, the deterministic FSM outcome stands unchanged.
- With it disabled (the default), the agent is **fully deterministic, offline, and reproducible** — and that is the configuration used for all CI, grading, and the headline metrics. The brain is a quality knob, never a correctness dependency.

This keeps the system's correctness story honest: the defensible numbers come from deterministic logic over the model's own distribution, not from a second opaque model.

---

## 8. Fail-soft behaviour

The agent never throws an unhandled error at its caller and never emits a confident answer it cannot justify:

- **Bad input** (D1) → halt with `error:bad_image` / `error:bad_question`, no model call, no cost, no image processed.
- **Model failure** (D3) → caught and reported as `error:model_failure`; the API/CLI returns a clean error envelope rather than a stack trace.
- **Uncertain model** (D4) → answer suppressed to `"unsure"`, `status="abstained:low_confidence"`, `needs_review=true`.
- **Type-wrong model** (D5) → if even re-ranking can't find a type-consistent candidate, answer suppressed to `"unsure"`, `status="abstained:type_mismatch"`, `needs_review=true`.

Every non-`ok*` outcome carries a machine-readable `status` and (for abstentions) a `needs_review` flag, so the deploy surface (FastAPI `/ask`, Gradio UI) renders an honest "I'm not sure — flagged for review" rather than a fabricated answer. This is especially important for the assistive/accessibility use cases (VizWiz-style blind-user photos, medical images) where a confident wrong answer is the worst outcome.

---

## 9. Value-add: why this beats raw argmax

A stock ViLT/BLIP model does raw argmax. The agent adds two production-grade behaviours **with no extra training**, both purely from deterministic logic over the model's own top-k + softmax:

1. **Calibrated abstention (D4).** Temperature-scaled `p_max` + top1–top2 margin + entropy gate ⇒ the system refuses to answer when uncertain instead of emitting a confident wrong answer. Headline metric: *answer-when-confident accuracy* and *abstention rate* vs raw argmax.
2. **Type-aware constraint + re-rank (D2→D5).** Classify the question type, then restrict/re-rank to a type-consistent top-k candidate (`yes/no → {yes,no}`, `how many → a number`, `what color → a color word`). Fixes the fluent-but-type-wrong failure (answering `"cat"` to *"how many?"*). The same frozen model yields higher *constrained* accuracy.

Both are measured directly: the autoreport contrasts **raw-argmax accuracy** against **constrained + abstaining accuracy / coverage**, broken down per answer-type (`yes/no`, `number`, `other`). Because the five decision points are explicit and traced, the orchestration is **auditable** and **offline-testable** (the whole FSM runs against `SceneStubVQA` with no torch and no network).

---

## 10. End-to-end example traces

### Example A — clean count, re-ranked to type

**Input:** synthetic scene (3 red squares, 1 blue circle) + *"how many red squares?"*

| D | Decision | Key signals | Status |
|---|---|---|---|
| D1 | accept | image var > eps; question interrogative, 4 tokens | (continue) |
| D2 | route → `count` | leading `how many` → bucket `count`, allowed = digits 0..20 + none | (continue) |
| D3 | predict | `topk=[("squares",0.41),("3",0.33),("2",0.14),("4",0.08),("red",0.04)]`, `p_max=0.41`, `margin=0.08`, `entropy=1.39` | (continue) |
| D4 | **borderline** | `tau_conf(count)=0.45`; `p_max 0.41 < 0.45`? → with temperature calibration `p_max` lifts to 0.47, `margin` 0.12 ≥ 0.10, `entropy` 1.39 ≤ 1.50 → CONFIDENT | (continue) |
| D5 | **re-rank** | `topk[0]="squares"` ∉ count_lexicon; scan finds `"3"` (highest-prob count) → promote | `ok:reranked` |

**Output:** `{ answer: "3", status: "ok:reranked", question_type: "count", answer_type: "number", confidence: 0.47, needs_review: false }` — correct, where raw argmax (`"squares"`) would have scored 0.

### Example B — low confidence, abstain

**Input:** blurry scene + *"what color is the triangle?"* (two triangles, ambiguous)

| D | Decision | Key signals | Status |
|---|---|---|---|
| D1 | accept | valid image + interrogative question | (continue) |
| D2 | route → `color` | `what color` → bucket `color`, allowed = color_lexicon | (continue) |
| D3 | predict | `topk=[("red",0.28),("orange",0.24),("blue",0.20),("green",0.16),("purple",0.12)]`, `p_max=0.28`, `margin=0.04`, `entropy=1.59` | (continue) |
| D4 | **abstain** | `p_max 0.28 < tau_conf 0.30`; `margin 0.04 < 0.10`; `entropy 1.59 > 1.50` — fails all three | `abstained:low_confidence` |

**Output:** `{ answer: "unsure", status: "abstained:low_confidence", question_type: "color", confidence: 0.28, needs_review: true }` — the flat distribution is correctly refused rather than guessing `"red"`.

### Example C — bad input, hard halt

**Input:** all-white image + *"how many cats?"*

| D | Decision | Key signals | Status |
|---|---|---|---|
| D1 | **reject** | pixel variance `3e-6 < blank_pixel_var_eps 1e-4` → degenerate-blank | `error:bad_image` (HALT) |

**Output:** `{ status: "error:bad_image", trace: [D1] }` — the model is never called; no image is processed or retained.

---

## 11. File map

| Path | Responsibility |
|---|---|
| `src/mmqa/agent/__init__.py` | `Agent.predict(image, question) -> result` — drives D1…D5 |
| `src/mmqa/agent/ingest.py` | D1 input gate (image validity, question validity) |
| `src/mmqa/agent/classify.py` | D2 keyword question-type router (+ full 10-rule table) |
| `src/mmqa/agent/answer.py` | D3 model call via the VQA wrapper (`predict()` contract) |
| `src/mmqa/agent/calibrate.py` | D4 calibrated abstention gate |
| `src/mmqa/agent/constrain.py` | D5 type-consistency gate + re-rank |
| `src/mmqa/agent/trace.py` | `ToolTrace` structured audit log |
| `src/mmqa/agent/config.py` | `AgentConfig` thresholds/lexicons |
| `src/mmqa/vision/` | model wrappers (`ViltForQuestionAnswering`, BLIP) + `SceneStubVQA` + `SceneImage` |

The agent depends on the model **only** through the `predict()` contract at D3, so the production model and the offline `SceneStubVQA` are fully interchangeable — the FSM, the gates, and every test stay identical.
