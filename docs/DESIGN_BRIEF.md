# P17 Multimodal Question Answering (VQA) — Design Brief

> Status: spec for the engineer. Decisive and prescriptive. Where a choice exists, the DEFAULT is named and the alternatives are tiered. Verified Hugging Face ids, licenses, and dataset schemas are baked in. Anything non-commercial is FLAGGED in-line.

---

## 1. Problem and scope

**Task.** Visual Question Answering: given an `(image, question)` pair, produce a short natural-language `answer`. Example: image of a kitchen + "how many chairs are there?" -> "2"; "what color is the table?" -> "white"; "is there a stove?" -> "yes".

**The trainable core.** The single trainable artifact in this project is the **VQA model** — the multimodal transformer that consumes image patches + question tokens and emits an answer. Everything else (image decode/resize, question tokenization, answer normalization, the agent's decision logic, scoring) is pretrained or deterministic and is NOT trained.

**Two answer regimes** (the project supports both, defaults to the first):

- **Classification VQA** (DEFAULT). A fixed answer vocabulary (the canonical 3129-answer VQAv2 label space) with a linear head -> logits -> softmax over the vocab -> top-k answers with calibrated-ish probabilities. Clean metric, clean confidence signal, easy to train and evaluate. This is what the agent is designed against.
- **Generative VQA** (alternative). Autoregressive decoding of free-form answer text (open vocabulary). Handles answers outside the fixed vocab; confidence is the sequence/token logprob. Harder to evaluate (must normalize and exact-match), used as a complement and as an upper-bound comparison.

**In scope:** natural-image VQA (COCO-style scenes), VQAv2-style yes/no + number + other answer types, the agentic wrapper, an offline synthetic test harness. **Out of scope:** document/infographic/OCR VQA as the primary target (TextVQA is only a side benchmark), multi-turn dialog, region grounding/boxes as outputs.

---

## 2. Pipeline architecture

```
            image ─────────────► [image preprocess]  ─┐
                                  RGB convert,         │
                                  resize/normalize     │
                                  (model image proc)   ├──► [VQA MODEL] ──► [answer post-process] ──► answer
            question ──────────► [question tokenize] ──┘     (TRAINABLE       classification:           (+ status,
                                  WordPiece/BPE,             core)            argmax/top-k label         confidence,
                                  [CLS] q [SEP]                               lookup over vocab          question_type)
                                                                             generation:
                                                                             decode + VQAv2
                                                                             answer-normalization
```

| Stage | What it is | Trained? |
|---|---|---|
| Image preprocess | RGB-convert, resize to model resolution (ViLT/BLIP 384px), normalize. Done by the model's own image processor. | Pretrained / algorithmic |
| Question tokenize | WordPiece (ViLT/BLIP use BertTokenizerFast) or BPE/T5 tokenizer; fused as `[CLS] question [SEP]` with image patches. | Pretrained |
| Multimodal encode | Single-stream transformer over concatenated image-patch + text tokens (ViLT), or image encoder + Q-Former/cross-attn text decoder (BLIP / BLIP-2 / GIT). | **TRAINED (the core)** |
| Answer head | Classification: 3129-way linear head -> softmax. Generation: autoregressive decoder. | **TRAINED (the core)** |
| Answer post-process | Classification: argmax/top-k label lookup. Generation: decode string + apply canonical VQAv2 answer-normalization. | Algorithmic |
| Agent wrapper | Deterministic 5-point state machine over the model's top-k + softmax (see §6). | Algorithmic (no training) |
| Scoring | Soft VQA accuracy over 10 annotators (see §5). | Algorithmic |

**Three production entry points the notebook must demonstrate:**

1. `transformers.pipeline('visual-question-answering', model='dandelin/vilt-b32-finetuned-vqa')` — returns `[{answer, score}, ...]` top-k. The quick classification path.
2. Explicit `ViltProcessor` + `ViltForQuestionAnswering` — exposes raw `outputs.logits` -> softmax -> top-k + confidence. **This is the path the agent uses** (it needs the full distribution, not just argmax).
3. `BlipProcessor` + `BlipForQuestionAnswering` with `.generate()` — the open-vocab generative path.

LAVIS (Salesforce) `load_model_and_preprocess('blip_vqa')` is the upstream reference for the HF BLIP port; cite it, do not depend on it.

**Architectural key point:** classification VQA gives a per-answer softmax distribution that is the cleanest input for a calibrated abstention gate; generative VQA must use logprob/beam-score as the confidence proxy. The agent targets regimes (1)/(2).

---

## 3. Models

All ids below are **verified on the HF Hub**. Processor/tokenizer and model class are named for each. Licenses are stated; non-commercial / unclean licenses are **FLAGGED**.

### Tier table

| Role | id | License | Params | Class + Processor |
|---|---|---|---|---|
| **DEFAULT (trainable core, classification)** | `dandelin/vilt-b32-finetuned-vqa` | apache-2.0 | ~113M | `ViltForQuestionAnswering` + `ViltProcessor` (ViltImageProcessor + BertTokenizerFast) |
| Pretrain base for a custom answer vocab | `dandelin/vilt-b32-mlm` | apache-2.0 | ~113M | `ViltForQuestionAnswering(num_labels=your_vocab)` + `ViltProcessor` |
| **GENERATIVE ALTERNATIVE (default tier)** | `Salesforce/blip-vqa-base` | bsd-3-clause | 384.7M | `BlipForQuestionAnswering` + `BlipProcessor` (BlipImageProcessor + BertTokenizerFast) |
| Stronger generative (default/A100) | `Salesforce/blip-vqa-capfilt-large` | bsd-3-clause | ~470M | `BlipForQuestionAnswering` + `BlipProcessor` |
| Lightweight generative (T4/default) | `microsoft/git-base-vqav2` | mit | 177.2M | `AutoModelForCausalLM` + `AutoProcessor` (GitProcessor: CLIP image proc + BertTokenizer) |
| **H100 UPGRADE (generative, encoder-decoder)** | `Salesforce/blip2-flan-t5-xl` | mit | ~3.94B | `Blip2ForConditionalGeneration` + `Blip2Processor` (image proc + T5TokenizerFast); bf16 + LoRA / Q-Former-only |
| H100 alternative (decoder-only) | `Salesforce/blip2-opt-2.7b` | mit | ~3.74B | `Blip2ForConditionalGeneration` + `Blip2Processor` (image proc + GPT2TokenizerFast). Note: OPT weights carry an upstream Meta research-use caveat; HF metadata says mit |
| Large permissive instruction-VLM | `Qwen/Qwen2-VL-2B-Instruct` | apache-2.0 | ~2.21B | `Qwen2VLForConditionalGeneration` + `AutoProcessor` (+ qwen-vl-utils) |
| Top H100 instruction-VLM | `Qwen/Qwen2-VL-7B-Instruct` | apache-2.0 | ~8.29B | `Qwen2VLForConditionalGeneration` + `AutoProcessor`; LoRA only |
| Modern small VLM — **FLAG LICENSE** | `Qwen/Qwen2.5-VL-3B-Instruct` | **Qwen Research (non-commercial); HF metadata empty — FLAG** | ~3.75B | `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` (+ qwen-vl-utils) |

### T4 fallback (16GB, free Colab)
- DEFAULT `dandelin/vilt-b32-finetuned-vqa` (or fine-tune from `dandelin/vilt-b32-mlm`): ViLT ~113M, classification head over ~3129 answers, trains comfortably on T4, batch 16-32 at 384px.
- `Salesforce/blip-vqa-base` (~385M, generative) fits T4 for inference and LoRA/small-batch fine-tune.
- `microsoft/git-base-vqav2` (177M) small generative contrast.

### Default tier (A100 / L4)
ViLT full-speed fine-tune; `Salesforce/blip-vqa-base` / `blip-vqa-capfilt-large` full fine-tune; `microsoft/git-base-vqav2` generative.

### H100 upgrade
`Salesforce/blip2-flan-t5-xl` (RECOMMENDED — encoder-decoder T5 gives cleaner constrained / instruction-style decoding, which fits the agent's type constraints, plus strong zero-shot) or `Salesforce/blip2-opt-2.7b`; both need 8-bit/bf16 + LoRA / Q-Former-only tuning. Largest tier: Qwen2-VL / Qwen2.5-VL or LLaVA-1.5-7B (LoRA only).

### AVOID / FLAG
- **`google/pix2struct-vqav2-base` — DOES NOT EXIST (404 on HF Hub). Do NOT use this id.** Pix2Struct only ships document/infographic variants (`pix2struct-infographics-vqa-*`, `pix2struct-ai2d-base`) that target rendered text/diagrams/screenshots, NOT natural-image VQAv2 — wrong fit for scene questions.
- **`llava-hf/llava-1.5-7b-hf` — license `llama2` (custom Llama 2 Community License, restricted-use / non-commercial-ish). FLAG; not clean permissive. Also 7B is too heavy for the trainable core.** Use only as a top-tier generative upper-bound comparison, never as default.
- **`Qwen/Qwen2.5-VL-3B-Instruct` — Qwen Research License (non-commercial); HF license field is EMPTY. FLAG before any commercial use.** (Note the 7B Qwen2-VL is Apache-2.0 and clean; the 3B is not.)
- `Qwen/Qwen2-VL-7B-Instruct` is Apache-2.0 (license-clean) but ~8B params is overkill for a fine-tunable classification core; reserve for the top tier only.

**Default decision:** ship `dandelin/vilt-b32-finetuned-vqa` (apache-2.0) as the trainable core. It is the cleanest permissive + HF-Trainer-friendly option, is already a classification head over the canonical 3129 answers (clearest metric and confidence), and maps directly onto the agent's type-constraint + abstention gates.

---

## 4. Datasets

All ids verified on HF Hub. **Any repo with no declared license is treated as "license unconfirmed" and FLAGGED**, even though VQAv2's upstream source (COCO 2014/2015 images CC-BY-4.0 + VQA annotations CC-BY-4.0) is commercially usable.

| id | Role | License | Schema / notes |
|---|---|---|---|
| `lmms-lab/VQAv2` | **PRIMARY EVAL** (canonical benchmark) | cc-by-4.0 (clean) | 8 cols: `question_type`, `multiple_choice_answer` (consensus), `answers` (List of 10 `{answer, answer_confidence∈{yes,maybe,no}, answer_id∈1..10}`), `image_id`, `answer_type∈{yes/no,number,other}`, `question_id`, `question`, `image` (PIL). 769.5K rows: validation 214.4K, testdev 107.4K, test 447.8K. **NO train split.** 10-annotator structure verified via preview. Soft-VQA-accuracy ready. |
| `HuggingFaceM4/VQAv2` | **PRIMARY TRAIN** (only common mirror with a train split + full annotator schema) | unspecified-on-hub — **FLAG** (annotations + COCO images CC-BY-4.0 upstream) | Full VQAv2 fields incl. 10-annotator `answers`. train ~443K + val ~214K + test. **CAVEAT: loading-script dataset; Dataset Viewer returns 501; requires `load_dataset(..., trust_remote_code=True)`; fragile on newer `datasets`; may pull COCO images by URL.** Use ONLY when you must fine-tune on VQAv2 train. |
| `merve/vqav2-small` | DEMO / fast Colab smoke loop | unspecified — **FLAG** | ONLY 3 cols: `image`, `question`, `multiple_choice_answer`. validation 21.4K rows, parquet, embedded images. **NOT valid for official soft VQA-accuracy (no 10 annotators) or type routing (no question_type/answer_type).** Demo only. |
| `Multimodal-Fatima/VQAv2_sample_train` | **TINY TRAIN with FULL schema** (offline-friendly) | unspecified — **FLAG** | 13 cols incl. `answers_original` (the 10 annotator dicts), `question_type`, `answer_type`, `question_id`, `image` (embedded), plus extras (blip_caption, clip_tags, DETA_detections). train 1.0K rows (155MB). Best small set that keeps the 10-annotator structure for proper scoring; works offline once cached. |
| `dandelin/vilt-b32-finetuned-vqa` | **ANSWER-VOCAB SOURCE** (3129 labels) | apache-2.0 (clean) | MODEL, not a dataset. `config.json` `id2label`/`label2id` is the canonical 3129-answer ViLT VQA vocabulary. Load: `AutoConfig.from_pretrained('dandelin/vilt-b32-finetuned-vqa').id2label` (len 3129); answer list = `list(config.id2label.values())`. |

**Alternative benchmarks** (side evaluations, not the main loop):
- `lmms-lab/GQA` — mit (clean), compositional / structured-scene reasoning.
- `facebook/textvqa` — cc-by-4.0 (clean), text-in-image / OCR-reasoning ("read the text" type).
- `lmms-lab/OK-VQA` — unspecified (FLAG), knowledge-based VQA; good abstention test.
- `lmms-lab/VizWiz-VQA` — unspecified (FLAG), real blind-user photos with an explicit **unanswerable** label; ideal for testing calibrated abstention.
- `Luo-wj/DAQUAR` — unspecified (FLAG), small classic indoor-scene VQA; only viable DAQUAR mirror with real data.

### SYNTHETIC scene generator (offline) — REQUIRED
For offline/CI and torch-free runs, build a tiny in-memory dataset of items
`{PIL image, question, answers (List of 10), multiple_choice_answer, question_type, answer_type}`
matching the `lmms-lab/VQAv2` 8-col schema, so the agent + scoring run with **no network and no model download**. Full spec in §7.

### Licensing summary (decision)
- **Commercially safe (declared):** `lmms-lab/VQAv2` (cc-by-4.0), `facebook/textvqa` (cc-by-4.0), `lmms-lab/GQA` (mit), `dandelin/vilt-b32-finetuned-vqa` (apache-2.0).
- **AVOID — NON-COMMERCIAL:** `thangduong0509/daquar_vqa` (cc-by-nc-sa-4.0). Do not use. (Use `Luo-wj/DAQUAR` instead if a DAQUAR mirror is needed, but it is license-unconfirmed.)
- **FLAG — license unconfirmed (undeclared on repo, CC-BY-4.0 upstream):** `HuggingFaceM4/VQAv2`, `merve/vqav2-small`, `Multimodal-Fatima/*`, `lmms-lab/OK-VQA`, `lmms-lab/VizWiz-VQA`, `Luo-wj/DAQUAR`. Treat as "verify before commercial use."
- Also avoid `jp1924/VisualQuestionAnswering` (gated + Korean, not VQAv2).

**Default data decision:** TRAIN = `HuggingFaceM4/VQAv2` (only train mirror with 10 annotators; FLAG license, needs `trust_remote_code=True`). EVAL = `lmms-lab/VQAv2` validation (cc-by-4.0, clean parquet). DEMO = `merve/vqav2-small` (quick) or `Multimodal-Fatima/VQAv2_sample_train` (when you need real scoring offline). VOCAB = `dandelin/vilt-b32-finetuned-vqa` config (3129). CI/offline = the synthetic generator (§7).

---

## 5. The VQA accuracy metric

**Official VQA accuracy** (Antol et al. 2015; Goyal et al. 2017), robust to inter-human disagreement by sampling over the 10 annotators.

> **There is NO `evaluate-metric/vqa` space on HF Hub** (verified — returns Not found). The metric MUST be re-implemented from the canonical `VQAEval` reference (~40 lines: `processPunctuation` + `processDigitArticle` + the accuracy loop). `evaluate-metric/exact_match` and `evaluate-metric/accuracy` resolve and may be used ONLY as auxiliary strict/loose sanity checks — they are NOT the VQA soft accuracy.

### Per-answer formula
For a candidate answer `a` (after normalization):
```
acc(a) = min( 1, (# of the 10 human annotators that gave answer a) / 3 )
```
Matching >=3 humans -> 1.0; 2 -> 0.667; 1 -> 0.333; 0 -> 0.0.

### Canonical leave-one-out averaging (the official VQAEval implementation)
For the predicted answer, compare against each of the 10 GT answers 9-at-a-time:
```
acc = (1/10) * sum over the 10 leave-one-out subsets S of  min(1, (# matches of prediction within the 9 answers in S) / 3)
```
This averages out the case where the prediction itself is one of the 10 GT answers. The common simplification when the prediction is not necessarily in GT reduces to `per-question acc = min(1, matches/3)` where `matches` = count of the 10 GT answers equal to the prediction — **but implement the 10x leave-one-out average to match the reference.** BOTH prediction and all GT answers go through the SAME normalization first.

**Dataset score = mean of per-question acc over all questions, x100 (a percentage).**

### Answer normalization (single source of truth — apply IDENTICALLY to prediction and all 10 GT answers; order matters)
1. Replace `\n` and `\t` with a space; strip leading/trailing whitespace.
2. Lowercase the whole string.
3. **processPunctuation:** remove punctuation in `[ ; / [ ] " { } ( ) = + \ _ - > < @ ` , ? ! ]` — BUT do not split numbers: a **period** is removed only if NOT between two digits (so `2.5` stays); a **comma between digits** is removed without inserting a space (so `100,000` -> `100000`). Then collapse.
4. **processDigitArticle**, token by token:
   a. Number-word -> digit via `manualMap` `{none:0, zero:0, one:1, two:2, three:3, four:4, five:5, six:6, seven:7, eight:8, nine:9, ten:10}`.
   b. Drop articles `{a, an, the}`.
   c. Canonicalize contractions via the official `contractions` dict (~80 entries): `dont->don't, isnt->isn't, arent->aren't, wasnt->wasn't, wont->won't, cant->can't, couldnt->couldn't, didnt->didn't, doesnt->doesn't, hasnt->hasn't, havent->haven't, im->i'm, ive->i've, id->i'd, youre->you're, youll->you'll, theyre->they're, thats->that's, whats->what's, wheres->where's, hes->he's, shes->she's, ...`.
5. Collapse multiple spaces to one; final strip.

After normalization, an answer matches a GT answer iff the strings are **exactly equal**. Do NOT touch the period inside decimals; the comma is stripped (not spaced) so digit groups merge.

### Per-type reporting
- **ANSWER-TYPE (3 buckets — the headline breakdown):** each question is tagged `answer_type ∈ {yes/no, number, other}` in the annotations. Report accuracy within each bucket plus overall: `accuracy['overall']`, `['perAnswerType']['yes/no' | 'number' | 'other']`. Typical ordering: yes/no highest (~80-90%), other mid, number lowest. **Always report all four numbers x100.**
- **QUESTION-TYPE (65 buckets — finer):** each question has a `question_type` = canonical leading phrase ('how many', 'what color is the', 'is the', 'are there', 'what is the', ...). Report `accuracy['perQuestionType'][qt]` as a table/appendix.
- **Implementation:** re-implement the three aggregations over the per-question acc values, grouping by the annotation's `answer_type` and `question_type` fields (present in `lmms-lab/VQAv2` and `HuggingFaceM4/VQAv2`). For test-dev/test-std the GT is held out (EvalAI server); for train/val compute locally.

### Baselines (report all)
1. **Prior 'yes':** always answer "yes". Lower bound for yes/no; near-0 on number/other (VQAv2 is balanced -> ~chance on yes/no).
2. **Most-common-answer (per-type) prior:** answer the single globally most frequent training answer (overall 'yes'; count -> '2'; color -> 'white'/'red'). The standard "prior" table from the VQA papers.
3. **Blind / question-only (language-only):** classifier on the QUESTION text alone, NO image (BoW/LSTM over the question -> answer softmax). Measures dataset language bias; the gap to the full model = "how much the image actually helps." VQAv2 was specifically built to depress this baseline.
4. **Zero-shot pretrained VQA (no fine-tuning on the target split):** `dandelin/vilt-b32-finetuned-vqa` (ideal zero-shot classifier baseline), `Salesforce/blip-vqa-base` / `blip-vqa-capfilt-large` (generative), and as stronger references `Salesforce/blip2-flan-t5-xl` / `llava-hf/llava-1.5-7b-hf` (**FLAG: llama2 license**). Prefer ViLT/BLIP for permissive licensing.
5. *(Optional)* image-only / answer-prior ablation: blind to the question, predict from image features only — completes the bias-ablation set.

---

## 6. Agentic component

A **deterministic state machine** wrapping the (frozen) VQA model. The only step that touches the model is D3; everything else is deterministic logic over the model's top-k + softmax. **>=5 decision points**, each auditable and testable offline.

```
(image, question)
      │
  ┌───▼────┐  D1 INPUT GATE
  │ valid? │──bad image──► halt status='error:bad_image'
  └───┬────┘──bad question──► halt status='error:bad_question'
      │ ok
  ┌───▼─────────┐  D2 QUESTION-TYPE ROUTER (keyword/regex; defines the allowed answer set)
  │ yes_no/count│
  │ color/other │
  └───┬─────────┘
      │
  ┌───▼────┐  D3 RUN VQA (real ViLT logits | offline StubVQA) -> topk, p_max, entropy, margin
  │ model  │──exception──► halt status='error:model_failure'
  └───┬────┘
      │
  ┌───▼──────────────┐  D4 CALIBRATED ABSTENTION GATE
  │ confident enough?│──no──► answer='unsure' status='abstained:low_confidence'
  └───┬──────────────┘
      │ yes
  ┌───▼───────────────────┐  D5 TYPE-CONSISTENCY GATE + RE-RANK
  │ top1 in allowed set?  │──no match in top-k──► status='abstained:type_mismatch'
  └───┬───────────────────┘
      │  -> return {answer, status∈{ok, ok:reranked}, question_type, confidence, topk}
```

### The 5 decision points

| ID | Gates on (signal) | Branches |
|---|---|---|
| **D1 Input gate** | Validity of `(image, question)` BEFORE any model call. Image decodes to a valid nonzero RGB array and is not degenerate-blank (pixel-variance > eps); question is non-empty after strip, has >=1 alphabetic token, length in [1,128] tokens, and is interrogative (ends in '?' OR starts with wh/aux: what/which/how/where/is/are/does/do/can). PIL + regex, **no torch**. | valid image AND valid question -> D2 \| corrupt/blank image -> halt `error:bad_image` \| empty/non-question -> halt `error:bad_question`. Hard fail; never reaches the model. |
| **D2 Question-type router** | Deterministic lexical mapping to `{yes_no, count, color, object/other}`. Leading 'is/are/does/do/can/has/was' OR 'is there' -> yes_no; 'how many'/'number of'/'count' -> count; 'what color'/'which color'/'what colour' -> color; else -> object/other. Pure regex/keyword table. **This is the value-add anchor** — the chosen type defines the allowed answer set used at D5. | yes_no -> `{yes,no}` \| count -> number words/digits `{0..20,'none'}` \| color -> closed color lexicon `{red,blue,green,yellow,...}` \| object/other -> open vocab (D5 becomes a pass-through no-op). |
| **D3 Run VQA** | Execute the model and capture the FULL distribution, not just argmax. Real: `ViltForQuestionAnswering -> logits`. Offline: `StubVQA` reads the embedded scene spec (§7). Produces top-k (k=5) `(answer, prob)`, `p_max = p1`, Shannon entropy `H = -sum p_i log p_i` over the truncated/renormalized dist, and regime. Only step that touches the model. | always emits `{topk, p_max, entropy, regime}`; runtime exception -> halt `error:model_failure`; normal -> D4 with distribution attached. |
| **D4 Calibrated abstention gate** | Whether the model is confident enough to answer at all. **VQA classifiers are overconfident**, so thresholds are tuned/calibrated on a held-out split (temperature-scaled softmax), not hard-coded. Signals: `p_max` vs `tau_conf` (e.g. 0.30) AND entropy `H` vs `tau_ent` (e.g. 1.5 nats) AND top1-top2 margin `(p1-p2)` vs `tau_margin`. Optionally per-question-type thresholds (count is harder -> stricter; yes/no can use a higher tau). | `p_max>=tau_conf AND H<=tau_ent AND margin>=tau_margin` -> CONFIDENT -> D5 \| fails any -> ABSTAIN: answer='unsure' `status='abstained:low_confidence'` (or route to needs_review). |
| **D5 Type-consistency gate + re-rank** | Force the answer to be consistent with the D2 type by re-ranking WITHIN top-k. Is `topk[0]` in the D2 allowed set? If not, scan top-k for the highest-prob entry that IS. Also normalizes form (digit<->word for counts, canonical color spelling). Example: count question, top1='cat'(0.4), top2='3'(0.3) -> re-rank to '3'. | top1 already consistent -> return it `status='ok'` \| a lower-ranked top-k entry is consistent -> return it `status='ok:reranked'` \| NO top-k entry matches the type -> abstain `status='abstained:type_mismatch'`. For object/other the set is open -> pass-through. Final output: `{answer, status, question_type, confidence, topk}`. |

> Mapping question-types to VQA's 3 reported answer-types: `yes_no -> yes/no`, `count -> number`, everything else `-> other`. Final emitted answer is run through the canonical VQA normalization (§5) before scoring.

### Value-add
A stock ViLT/BLIP model only does raw argmax. The deterministic agent adds two things production needs and a bare model lacks, with **no extra training**:

1. **Calibrated abstention (D4).** VQA models are notoriously overconfident. A temperature-scaled max-prob + entropy + top1-top2-margin gate lets the system say "unsure"/needs_review instead of emitting a confident wrong answer — trading a little coverage for a large precision/reliability gain. **This is the single most defensible headline metric:** answer-when-confident accuracy and abstention rate vs a raw-argmax baseline.
2. **Type-aware answer constraint + re-rank (D2->D5).** Classify the question type and restrict/re-rank the answer to a type-consistent top-k candidate (yes/no -> {yes,no}; 'how many' -> a number; 'what color' -> a color word). Fixes the common failure where the argmax is fluent-but-type-wrong (e.g. answering 'cat' to 'how many?'). The same frozen model yields higher constrained accuracy.

Net: higher constrained accuracy + principled refuse-to-answer, all from deterministic logic over the model's own top-k + softmax. The >=5 explicit decision points make the orchestration auditable and offline-testable.

---

## 7. Offline / test design

Mirrors P15's SeedEngine pattern: the whole 5-point agent and the scorer run with **zero torch and no network**, deterministic and reproducible from a seed.

### Synthetic scene generator — `make_scene(seed)` (PIL only)
1. Sample N objects (2-6), each with `shape ∈ {square, circle, triangle}`, `color ∈ {red, green, blue, yellow, orange, purple}` (fixed 6-color lexicon), and a **non-overlapping** bbox on a WxH (e.g. 224x224) white canvas; draw with `ImageDraw` (rectangle / ellipse / polygon).
2. Build a SCENE SPEC dict: `{'objects':[{'shape','color','bbox'}...], 'counts_by_color_shape':{...}, 'colors_present':[...], 'shapes_present':[...]}`.
3. **EMBED the spec in the PNG** so the image is self-describing and eval is fully deterministic:
   ```python
   meta = PngImagePlugin.PngInfo()
   meta.add_text('scene_spec', json.dumps(spec))
   img.save(path, pnginfo=meta)
   ```
4. **Templated QA derivation** from the spec (one generator per type, to exercise D2/D5):
   - yes_no -> "is there a {shape}?" / "is there a {color} {shape}?" ; gold = yes/no from spec membership.
   - count -> "how many {color} {shape}s?" ; gold = integer from `counts_by_color_shape`.
   - color -> "what color is the {shape}?" ; gold = color (**only emit when that shape is unique** so gold is unambiguous).
   - object -> "what shape is the {color} object?" ; gold = shape.
   Each QA item = `{image_path, question, gold_answer, question_type}` (and a 10-answer list built from gold for soft scoring).

### Stub VQA model — `StubVQA` (no torch, no network)
Mirrors the OCR SeedEngine from P15. Same `predict(image, question)` signature as the real wrapper, returning `topk + probs`:
- (a) Read the spec back out of the PNG metadata: `json.loads(Image.open(path).text['scene_spec'])`.
- (b) Compute the TRUE answer from spec + question via the same template logic.
- (c) Return a **realistic distribution, not one-hot**: put ~0.7-0.9 mass on the correct answer and spread the remainder over plausible type-consistent distractors (other colors, adjacent counts), so `p_max` / entropy / margin at D3/D4 are meaningful and the abstention gate is actually exercised.
- (d) A `noise` / `difficulty` knob deliberately lowers `p_max` or injects a type-wrong top1 on a fraction of items so **D4 (abstain) and D5 (re-rank) trigger** on the eval set and can be unit-tested.

**Result:** the entire 5-point agent runs end-to-end with no torch and no network — deterministic, reproducible from the seed — yielding a real accuracy / abstention-rate / re-rank-rate report in CI and Colab-free mode. **Swapping `StubVQA` for a real `ViltForQuestionAnswering` wrapper (same `predict()` returning topk+probs) flips the system to production with NO agent changes.**

---

## 8. Question-type classifier (keyword scheme)

Lowercase + strip the question; take leading tokens. Evaluate rules **top-down, first match wins** (order is critical: 'how many' must beat 'how'; 'what color' must beat generic 'what').

1. **YES_NO** — starts with aux/copula/modal `{is, are, was, were, am, do, does, did, has, have, had, can, could, will, would, should, shall, may, might, must}`; ALSO 'is there'/'are there', or contains 'or not'. -> constrain to `{yes, no}`. answer_type = **yes/no**.
2. **NUMBER / COUNT** — 'how many', or starts 'how much', or 'what number'/'count'/'what is the number of'. -> constrain to digit / number-word (apply number-word->digit map). answer_type = **number**. (count prior = '2'.)
3. **COLOR** — contains 'what color'/'which color'/'what colour'/'color of'. -> constrain to color lexicon `{white, black, red, blue, green, yellow, brown, orange, pink, purple, gray/grey, silver, gold, tan, ...}`. answer_type = **other**. (color prior = 'white'.)
4. **WHERE** — starts 'where'. -> location/preposition-phrase. answer_type = other.
5. **WHO** — starts 'who'. -> person/agent noun. answer_type = other.
6. **WHY** — starts 'why'. -> reason phrase (hard; often low acc). answer_type = other.
7. **WHEN** — starts 'when'. -> time-of-day/season. answer_type = other.
8. **WHICH** — starts 'which'. -> selection among options. answer_type = other.
9. **WHAT / OBJECT** (wh- catch-all) — starts 'what' (not caught by color/number) or 'what kind/type/sort/brand/sport/animal/room/time'. Sub-route on head noun ('what animal' -> animal vocab, 'what sport' -> sport vocab). answer_type = other.
10. **DEFAULT / OTHER** — anything unmatched. answer_type = other; no hard constraint.

The agent's coarse router (§6 D2) uses the four buckets `{yes_no, count, color, object/other}`; this 10-rule table is the full reference and maps onto the VQAv2 65 question-type buckets for reporting.

---

## 9. Reuse map

### Reused from sibling templates (do NOT re-invent — follow P02/P15 patterns)
- **Config** — central config object / dataclass for model id, dataset id, thresholds (`tau_conf`, `tau_ent`, `tau_margin`), seed, resolution, batch size.
- **Logging** — structured logger; per-decision-point trace lines (D1..D5 status).
- **Registry** — model/dataset registry mapping role -> verified HF id + license (populate from §3/§4 tables).
- **Autoreport** — auto-generated metrics report (overall + per-answer-type + per-question-type accuracy, baselines, abstention rate, re-rank rate).
- **Monitoring** — run metrics, latency, abstention/coverage tracking.
- **Automation** — pipeline orchestration / CLI entry harness.
- **Grading** — scoring harness skeleton (here it wraps the VQA-accuracy implementation).
- **CLI** — command surface (train / eval / predict / demo).
- **API** — request/response wrapper around `agent.predict()`.

### NEW for P17
- **Multimodal VQA model wrapper** — uniform `predict(image, question) -> {topk:[(ans,prob)], p_max, entropy, margin, regime}` over `ViltForQuestionAnswering` (classification, default) and `BlipForQuestionAnswering`/generative (alternative).
- **Image handling** — RGB decode/convert, resize/normalize via the model's image processor; input validation (degenerate-blank detection).
- **Synthetic-scene generator + StubVQA** — §7: self-describing PNGs (embedded `scene_spec`) + a torch-free stub returning realistic distributions, mirroring P15's SeedEngine.
- **VQA-accuracy metric** — §5: re-implemented `VQAEval` (normalization + 10x leave-one-out soft accuracy + per-type aggregation), since no HF `evaluate-metric/vqa` exists.
- **Type-aware agent** — §6: the deterministic 5-point state machine (input gate, type router, model, calibrated abstention, type-consistency re-rank) including the question-type keyword classifier of §8.

---

## 10. Risks and gotchas

- **Overconfident VQA models.** Softmax probabilities are poorly calibrated and skew high. Do NOT threshold on raw `p_max` alone — calibrate (temperature scaling) on a held-out split and combine `p_max` + entropy + top1-top2 margin (D4). This is why abstention is a tuned gate, not a constant.
- **Language-prior bias.** Models answer from the question text alone ('what color is the banana?' -> 'yellow' regardless of image). Always run the **blind question-only baseline**; the full-model gap over it is the real signal that the image is used. VQAv2 was built to depress this but it is not zero.
- **Answer-vocab mismatch.** The classification head is fixed at the canonical **3129** answers (pulled from `dandelin/vilt-b32-finetuned-vqa` config). Any gold answer outside this vocab is unreachable by the classification model (a hard ceiling). If you fine-tune from `dandelin/vilt-b32-mlm` with a custom vocab, the metric, the answer-constraint sets, and the report buckets must all use the SAME vocab. Generative models avoid this ceiling but trade away the clean confidence signal.
- **Heavy image deps.** torch + transformers + a vision stack + multi-GB parquet/COCO images are heavy and flaky in CI/Colab-free. **Mitigation:** the synthetic-scene + StubVQA path (§7) keeps tests and the full agent torch-free and network-free; gate real-model paths behind an availability check.
- **COCO / VQAv2 licensing.** VQAv2 images are COCO 2014/2015 (CC-BY-4.0) and the annotations are CC-BY-4.0 upstream -> commercially usable in principle, but **several practical mirrors declare NO license on the repo** (`HuggingFaceM4/VQAv2`, `merve/vqav2-small`, `Multimodal-Fatima/*`, `lmms-lab/OK-VQA`, `lmms-lab/VizWiz-VQA`, `Luo-wj/DAQUAR`) — treat as "license unconfirmed," FLAG before commercial use. **`thangduong0509/daquar_vqa` is cc-by-nc-sa-4.0 (NON-COMMERCIAL) — do not use.** Prefer the declared-clean sets (`lmms-lab/VQAv2` cc-by-4.0, `facebook/textvqa` cc-by-4.0, `lmms-lab/GQA` mit) for anything that must be commercially defensible.
- **Dataset loading traps.** `HuggingFaceM4/VQAv2` is a loading-script dataset (Dataset Viewer 501, needs `trust_remote_code=True`, may fetch COCO images by URL) — fragile on newer `datasets`; reserve it strictly for the train split and pin versions. Use `lmms-lab/VQAv2` parquet for eval.
- **Non-existent / mislicensed models.** `google/pix2struct-vqav2-base` does NOT exist (404) — do not reference it. `llava-hf/llava-1.5-7b-hf` (llama2) and `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen Research, empty HF license) are NON-permissive — never default to them; use only as flagged upper-bound comparisons.
- **Generative eval is harder.** Free-form answers must pass through the exact canonical normalization (§5) before exact-match; without it, correct answers score as misses. Keep ONE normalization function as the single source of truth for both predictions and references.
