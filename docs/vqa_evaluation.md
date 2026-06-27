# P17 Multimodal QA — VQA Evaluation Methodology

> Author: Le Dinh Minh Quan (student 23127460) · Package `mmqa` · Folder `17_Multimodal_QA`
>
> This is the **quality document** for P17. It defines exactly how a Visual Question Answering (VQA) system is scored: the official soft VQA-accuracy formula, the exact answer-normalization steps, the per-answer-type / per-question-type breakdowns, the abstention and risk–coverage view introduced by the agent, the baseline ladder that exposes language-prior bias, and how the offline synthetic harness relates to real VQAv2. Read this before trusting any number the autoreport emits.

VQA is the first multimodal-vision project in this series (P02–P15 were text / document / OCR; P16 done). The task: given an **image** and a natural-language **question** about it, produce a short **answer** — e.g. a scene of colored shapes plus *"how many red squares?"* → `"2"`, or *"what color is the circle?"* → `"blue"`, or *"is there a triangle?"* → `"yes"`. Because the gold answer is a short string and humans disagree, evaluation is not plain accuracy: it is a **disagreement-robust soft accuracy averaged over 10 human annotators**, plus the constraint/abstention metrics that the agent adds on top.

---

## 1. Why VQA needs its own metric

A VQA answer is an open-ended short phrase, and ten humans shown the same image+question routinely give **different but all-correct** answers (`"couch"` vs `"sofa"`, `"2"` vs `"two"`, `"grey"` vs `"gray"`). Plain exact-match against a single gold string would punish correct paraphrases and reward nothing for partial human agreement. The community standard (Antol et al. 2015, *VQA*; Goyal et al. 2017, *VQAv2*) instead:

1. collects **10 independent human answers** per question,
2. **normalizes** prediction and all references through one canonical function, then
3. scores a prediction by **how many of the 10 humans it matches**, saturating at 3.

This makes the metric robust to inter-human disagreement: matching the majority view is full credit; matching a minority opinion earns partial credit; matching nobody earns zero.

> **There is NO `evaluate-metric/vqa` space on the Hugging Face Hub** (verified — returns *Not found*). The metric is re-implemented in P17 from the canonical `VQAEval` reference (~40 lines: `processPunctuation` + `processDigitArticle` + the accuracy loop). `evaluate-metric/exact_match` and `evaluate-metric/accuracy` resolve and are wired in only as **auxiliary** strict/loose sanity checks — they are **not** the headline soft VQA accuracy.

---

## 2. The official VQA accuracy formula

### 2.1 Per-answer credit

For a single candidate answer `a` (already normalized — see §3), credit is the fraction of the 10 annotators that gave `a`, capped at 1 when 3 or more agree:

```
acc(a) = min( 1, (# of the 10 human annotators whose normalized answer == a) / 3 )
```

| # of 10 annotators matching | credit |
|---|---|
| ≥ 3 | 1.000 |
| 2 | 0.667 |
| 1 | 0.333 |
| 0 | 0.000 |

The divisor `3` is deliberate: in VQAv2 a prediction that **three independent humans** also produced is treated as fully correct.

### 2.2 Canonical 10× leave-one-out averaging (the reference `VQAEval` loop)

The official implementation does **not** simply count matches against all 10 — it averages over the 10 *leave-one-out* subsets, comparing the prediction against 9 annotators at a time:

```
acc = (1/10) * Σ over the 10 leave-one-out subsets S  of   min( 1, (# of the 9 answers in S equal to the prediction) / 3 )
```

The leave-one-out step matters only when **the prediction itself is one of the 10 gold answers**: holding out one matching annotator at a time prevents a prediction from getting credit for "agreeing with itself." When the prediction is *not* among the gold answers, every leave-one-out subset gives the same count and the whole thing collapses to the simplified `per-question acc = min(1, matches/3)`.

> **P17 implements the full 10× leave-one-out average** to match the reference exactly, not the simplification — the two differ on questions where the model's answer coincides with a gold answer, which is the common case.

Both the prediction **and** all 10 gold answers pass through the **same** normalization (§3) **before** any comparison.

### 2.3 Dataset score

```
VQA accuracy (dataset) = mean of per-question acc over all questions  × 100   (a percentage)
```

All headline numbers in P17 are reported `×100`.

---

## 3. Answer normalization (single source of truth)

Normalization is applied **identically** to the prediction and to all 10 gold answers, and **order matters**. This is one function, reused by the metric, the agent's type-constraint sets (§6), and the report buckets — never duplicated, because a mismatch silently turns correct answers into misses.

1. **Whitespace:** replace `\n` and `\t` with a single space; strip leading/trailing whitespace.
2. **Lowercase** the whole string.
3. **`processPunctuation`** — remove punctuation in the set `[ ; / [ ] " { } ( ) = + \ _ - > < @ \` , ? ! ]`, with two numeric exceptions:
   - a **period** is removed only if it is **not** between two digits → `2.5` stays `2.5`;
   - a **comma between two digits** is removed **without** inserting a space → `100,000` → `100000`.
   Then collapse the result.
4. **`processDigitArticle`** — token by token:
   - **a.** number-word → digit via `manualMap` `{none:0, zero:0, one:1, two:2, three:3, four:4, five:5, six:6, seven:7, eight:8, nine:9, ten:10}`;
   - **b.** drop articles `{a, an, the}`;
   - **c.** canonicalize contractions via the official `contractions` dict (~80 entries): `dont→don't`, `isnt→isn't`, `arent→aren't`, `wasnt→wasn't`, `wont→won't`, `cant→can't`, `didnt→didn't`, `doesnt→doesn't`, `havent→haven't`, `im→i'm`, `ive→i've`, `youre→you're`, `theyre→they're`, `thats→that's`, `whats→what's`, `wheres→where's`, `hes→he's`, `shes→she's`, …
5. **Collapse** multiple spaces to one; final strip.

After normalization a prediction **matches** a gold answer **iff the strings are exactly equal**. Two failure modes this exact ordering guards against, both directly relevant to P17's shape-scene questions:

- **Counts:** the model emitting `"two"` while gold is `"2"` — step 4a maps both to `2`, so the count question *"how many red squares?"* scores correctly regardless of word/digit form.
- **Colors / decimals:** `"gray"`/`"grey"` are distinct strings and remain distinct (the metric does not synonym-merge colors — synonym handling is a *modeling* problem, not a metric one); but `2.5` is never mangled into `25`.

---

## 4. Per-type reporting

A single overall number hides where a VQA model actually fails. P17 reports two orthogonal breakdowns, both grouped over the *same* per-question acc values.

### 4.1 Answer-type — the headline 3-bucket breakdown

Every question carries `answer_type ∈ {yes/no, number, other}` (present in `lmms-lab/VQAv2` and `HuggingFaceM4/VQAv2`). Report accuracy within each bucket **plus** overall — always all four numbers, `×100`:

- `accuracy['overall']`
- `accuracy['perAnswerType']['yes/no']`
- `accuracy['perAnswerType']['number']`
- `accuracy['perAnswerType']['other']`

The typical ordering is **yes/no highest** (a 2-way space, ~80–90% for a decent model), **other** in the middle, **number lowest** (counting is the hardest VQA skill). The agent's question-type router maps onto these three: `yes_no → yes/no`, `count → number`, everything else → `other` (§6).

### 4.2 Question-type — the fine 65-bucket breakdown

Every question also carries a `question_type` = canonical leading phrase (`"how many"`, `"what color is the"`, `"is there a"`, `"is the"`, `"are there"`, `"what is the"`, …). Report `accuracy['perQuestionType'][qt]` as a sorted appendix table. This is where you see, e.g., `"how many"` dragging the `number` bucket down or `"why"` questions near the floor.

### 4.3 Implementation note

Re-implement the three aggregations (`overall`, `perAnswerType`, `perQuestionType`) over the per-question acc values, grouping by the annotation's `answer_type` / `question_type` fields. For **train/val** splits the gold is local → compute locally. For **test-dev / test-std** the gold is held out on the EvalAI server → only `overall` + the three answer-types come back, never the per-question gold.

---

## 5. Abstention, coverage, and risk–coverage

The agentic wrapper (a deterministic 5-decision FSM) is what distinguishes P17 from a raw model, and it **changes what "accuracy" means**: the agent is allowed to **abstain** (`answer = "unsure"`, `status = abstained:low_confidence` or `abstained:type_mismatch`) instead of guessing. Evaluation must therefore separate *did it answer* from *was the answer right*. The agent never silently guesses — it either answers, abstains on low confidence (D4), or abstains on type mismatch (D5).

### 5.1 Core abstention metrics

Let `N` = total questions, `A` = questions the agent actually answered (did not abstain).

| Metric | Definition | Reads as |
|---|---|---|
| **Coverage** | `A / N` | fraction of questions answered |
| **Abstention rate** | `1 − A/N` | fraction sent to `"unsure"` / needs_review |
| **Answer-when-confident accuracy** (selective accuracy) | mean VQA-acc over the `A` answered questions only | how good answers are *when it commits* |
| **Full-coverage accuracy** | mean VQA-acc over all `N`, abstentions scored 0 | apples-to-apples vs a model that always answers |
| **Re-rank rate** | fraction with `status = ok:reranked` | how often D5's type-constraint changed the argmax |

The headline claim of the agent is: **selective accuracy on the answered slice should beat the raw-argmax model's accuracy**, because the agent declines exactly the cases where the model is overconfident and wrong. The cost is coverage < 1.

### 5.2 Risk–coverage curve

Because D4's abstention is a **calibrated threshold** (temperature-scaled `p_max`, entropy `H`, top1–top2 margin), sweeping the threshold traces a **risk–coverage curve**: as you demand more confidence, coverage drops and selective accuracy rises. Report:

- the **curve** (selective accuracy vs coverage), and
- a single summary, the **area under the risk–coverage curve** (or, equivalently, selective accuracy at a few fixed coverage points, e.g. 100% / 90% / 75% / 50%).

A good abstention gate makes risk fall fast as coverage drops — i.e. the questions it abstains on really are the ones it would have gotten wrong. This curve is the **single most defensible artifact** in the P17 evaluation: it directly visualizes the precision/coverage trade the agent buys with no extra training.

### 5.3 VizWiz-style unanswerable check

For the abstention gate's *external* validity, evaluate on a set with genuinely unanswerable questions — `lmms-lab/VizWiz-VQA` ships an explicit **unanswerable** label (license undeclared on repo → **FLAG** before commercial use). There, the right behavior is to abstain, so **abstention-vs-unanswerable agreement** (does the agent abstain on the questions humans marked unanswerable?) becomes a real accuracy number rather than a pure coverage trade.

---

## 6. The agent's effect on scoring

Two agent stages move the score, and both are evaluated explicitly so the value-add is auditable, not asserted:

- **D5 type-consistency + re-rank.** A count question whose argmax is `"cat"` (fluent but type-wrong) gets re-ranked to the highest-prob *number* in the top-k. Measure: VQA accuracy **with** vs **without** the re-rank, plus the **re-rank rate**. The same frozen model should score higher constrained.
- **D4 calibrated abstention.** As in §5 — measure selective accuracy and coverage, and the risk–coverage curve.

Every emitted answer (whether from argmax, re-rank, or a baseline) is run through the canonical normalization of §3 **before** scoring, so all rows of the comparison table are measured on identical terms.

---

## 7. The baseline ladder (and the language-prior story)

VQA's central failure is the **language prior**: a model that answers from the *question text alone* — *"what color is the banana?"* → `"yellow"` regardless of the image. VQAv2 was specifically constructed (balanced complementary image pairs) to depress this shortcut, but it is never zero. The baseline ladder exists to **quantify how much the image actually helps**; every P17 eval table reports the full model against all of these.

| # | Baseline | What it does | Reads as | Expected |
|---|---|---|---|---|
| 1 | **Prior `"yes"`** | always answer `"yes"` | lower bound for yes/no; near-0 on number/other | ~chance on yes/no (VQAv2 is balanced) |
| 2 | **Most-common-answer (per-type) prior** | emit the single globally most-frequent answer for the type (overall `"yes"`; count → `"2"`; color → `"white"`) | the classic "prior" table from the VQA papers | floor that any real model must clear |
| 3 | **BLIND / question-only prior** | classifier on the **question text only**, **no image** (BoW/LSTM over the question → answer) | **measures dataset language bias** | full-model gap over this = *the image's contribution* |
| 4 | **Zero-shot pretrained ViLT** | `dandelin/vilt-b32-finetuned-vqa`, no fine-tuning on the target split | calibration point for the classification core | strong; sets the bar fine-tuning must beat |
| 5 | *(optional)* image-only / answer-prior ablation | blind to the **question**, predict from image features only | completes the bias-ablation set | low; symmetric to #3 |

Baseline #3 is the load-bearing one. **`full-model accuracy − blind-question-only accuracy`** is the headline "does the model use the image" number; a small gap is a red flag that the model is riding the language prior. For zero-shot references prefer the permissive options — `dandelin/vilt-b32-finetuned-vqa` (Apache-2.0), `Salesforce/blip-vqa-base` (BSD-3, generative); use heavier references like `Salesforce/blip2-flan-t5-xl` (MIT) or `llava-hf/llava-1.5-7b-hf` (**FLAG: llama2 license, non-commercial-ish**) only as flagged upper-bound comparisons, never as the default.

---

## 8. Reading overconfidence and calibration

VQA softmax probabilities are **poorly calibrated and skew high** — the model says `0.95` and is wrong far more than 5% of the time. This is *why* D4's abstention gate is tuned, not a fixed constant. Diagnose it with:

- **Reliability diagram** — bin predictions by `p_max`, plot empirical accuracy per bin against the diagonal. A curve sagging **below** the diagonal = overconfident (the default for raw VQA classifiers).
- **Expected Calibration Error (ECE)** — weighted average gap between confidence and accuracy across bins; a single overconfidence number to track before/after temperature scaling.
- **Temperature scaling** — fit one scalar `T > 1` on a held-out split to soften the softmax; report ECE before/after. The agent's `tau_conf` / `tau_ent` / `tau_margin` thresholds (D4) are set against the **temperature-scaled** distribution, optionally per-question-type (count is harder → stricter; yes/no can tolerate a higher `tau`).
- **Confidence-vs-correctness separation** — the practical question for abstention: is `p_max` (and entropy, and top1–top2 margin) systematically lower on the wrong answers than the right ones? If yes, the risk–coverage curve (§5.2) will be steep and the gate is worth its coverage cost; if `p_max` is high on wrong answers too (the overconfidence trap), `p_max` alone is useless and you lean on **entropy + margin** instead. This is exactly why D4 combines all three signals rather than thresholding raw `p_max`.

---

## 9. Offline synthetic vs real VQAv2

P17 evaluates on two surfaces, by design — mirroring P15's torch-free OCR `SeedEngine`.

### 9.1 Offline synthetic (CI / Colab-free / torch-free)

The synthetic-scene generator (`data/synth_scene.py`) draws colored shapes (`square / circle / triangle` × `red / blue / green / yellow / purple / orange`) on a PIL canvas, **embeds the exact scene spec in the PNG metadata**, and templates `(question, gold answer, type)` triples across all four routed types (yes_no / count / color / object). The `SceneStubVQA` model reads the embedded spec back out and returns a **realistic distribution** (≈0.7–0.9 mass on the truth, the rest spread over type-consistent distractors — adjacent counts, other colors) **with a difficulty knob** that deliberately lowers `p_max` or injects a type-wrong top1 on a fraction of items.

This surface exists so the metric and the **full 5-decision agent run with no torch, no model, no network** — deterministic and reproducible from a seed. Crucially the difficulty knob means **D4 (abstain) and D5 (re-rank) actually fire** on the synthetic eval set, so coverage, re-rank rate, and the risk–coverage curve are real, non-trivial numbers in CI — not a degenerate all-correct run. The synthetic gold is built as a full 10-answer list from the unambiguous template gold, so the **identical** soft-VQA-accuracy code path (§2) runs offline.

**What the synthetic surface proves:** that the scorer, normalizer, type router, abstention gate, and re-rank logic are *correct and wired together* — a software-correctness guarantee. **What it does NOT prove:** real-world accuracy. The stub answers from ground truth, so its "accuracy" only reflects the injected difficulty, not a model's true visual reasoning. Never quote a synthetic accuracy as a model result.

### 9.2 Real VQAv2

| Role | Dataset | License | Note |
|---|---|---|---|
| **Primary eval** | `lmms-lab/VQAv2` (validation) | cc-by-4.0 (clean) | clean parquet, 8 cols incl. the 10-annotator `answers`, `question_type`, `answer_type` → soft-VQA-accuracy ready |
| **Primary train** | `HuggingFaceM4/VQAv2` | undeclared on hub → **FLAG** (COCO/VQA upstream cc-by-4.0) | only common mirror with a train split + 10-annotator schema; loading-script, needs `trust_remote_code=True`, fragile |
| Offline-friendly real scoring | `Multimodal-Fatima/VQAv2_sample_train` (1K) | undeclared → **FLAG** | keeps the full 10-annotator schema → proper soft scoring offline once cached |
| Demo / smoke only | `merve/vqav2-small` | undeclared → **FLAG** | **3 cols only, NO 10 annotators** → **not valid for official soft accuracy** or type routing; demo loop only |
| Answer vocab (3129) | `dandelin/vilt-b32-finetuned-vqa` config `id2label` | apache-2.0 | the classification label space; any gold outside it is unreachable (a hard ceiling) |

Side benchmarks for targeted probes: `lmms-lab/GQA` (mit, compositional), `facebook/textvqa` (cc-by-4.0, text-in-image), `lmms-lab/VizWiz-VQA` (undeclared → **FLAG**; the unanswerable/abstention probe of §5.3).

**The two-surface contract:** the synthetic harness guarantees the evaluation *code* is correct and the agent's decisions are exercised; real VQAv2 (validation, soft accuracy) produces the *publishable* numbers. Swapping `SceneStubVQA` for the real `ViltForQuestionAnswering` wrapper (same `predict()` returning top-k + probs) flips the system from CI to production **with no change to the metric, the agent, or this methodology**.

---

## 10. The headline metrics table

Every P17 evaluation run emits this table (the autoreport template). Each system is scored on the **same** normalized references; abstentions are scored 0 under *full-coverage accuracy* and excluded from *selective accuracy*.

| System | Overall | yes/no | number | other | Coverage | Selective acc | Re-rank rate | ECE |
|---|---|---|---|---|---|---|---|---|
| Prior `"yes"` | — | — | — | — | 1.00 | — | — | — |
| Most-common-answer (per-type) prior | — | — | — | — | 1.00 | — | — | — |
| **BLIND** question-only prior | — | — | — | — | 1.00 | — | — | — |
| Zero-shot ViLT (`dandelin/vilt-b32-finetuned-vqa`) | — | — | — | — | 1.00 | — | — | — |
| Fine-tuned ViLT (raw argmax) | — | — | — | — | 1.00 | — | n/a | — |
| **Fine-tuned ViLT + agent (D4+D5)** | — | — | — | — | <1.00 | — | — | — |

Notes on reading the table:
- **Overall / yes/no / number / other** are soft VQA accuracy ×100 (§2–4), computed at **full coverage** (abstentions = 0) so every row is comparable.
- **Selective acc** is the §5.1 answer-when-confident accuracy — only the agent row has Coverage < 1; it should **exceed** the raw-argmax row's overall.
- **BLIND − full-model gap** (§7) is the language-prior diagnostic; small = warning.
- **Re-rank rate** quantifies D5's type-constraint activity; **ECE** quantifies overconfidence (§8) — track it before vs after temperature scaling.
- Pair the table with the **risk–coverage curve** (§5.2) and the **per-question-type appendix** (§4.2).

---

## 11. Pitfalls checklist

- **One normalizer, everywhere.** The §3 function is the single source of truth for predictions *and* references — for classification *and* generative outputs. A second copy silently scores correct answers as misses (the #1 generative-VQA eval bug).
- **Implement the 10× leave-one-out average**, not the `min(1, matches/3)` simplification — they differ exactly when the prediction is one of the 10 gold answers.
- **`merve/vqav2-small` is demo-only** — 3 columns, no annotators → cannot produce official soft accuracy. Don't report a number from it.
- **Answer-vocab ceiling.** The classification head is fixed at 3129 answers (from `dandelin/vilt-b32-finetuned-vqa` config). Any gold outside it is unreachable — a hard accuracy ceiling. If fine-tuning from `dandelin/vilt-b32-mlm` with a custom vocab, the metric, the constraint sets (§6), and the report buckets must all use the **same** vocab.
- **Never threshold on raw `p_max` alone** — calibrate first (temperature scaling) and combine `p_max` + entropy + margin (§8).
- **Always run the BLIND baseline** — without it you cannot tell skill from language prior (§7).
- **Synthetic accuracy is not a model result** — it validates the *code path* (§9.1); quote real-VQAv2 numbers as results.
- **Held-out splits give partial reports** — test-dev/test-std return overall + 3 answer-types only (EvalAI), never per-question gold.

---

*Sources: Antol et al. 2015 (VQA), Goyal et al. 2017 (VQAv2 balanced), the canonical `VQAEval` reference implementation. Dataset/model ids and licenses verified on the Hugging Face Hub during P17 research; undeclared-license mirrors are FLAGGED inline. See `docs/DESIGN_BRIEF.md` §5 and §7 for the upstream specification.*
