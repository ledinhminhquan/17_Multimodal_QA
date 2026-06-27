# P17 Multimodal Question Answering (VQA) — Model Selection

> **Project:** P17 `mmqa` — Multimodal Question Answering (Visual Question Answering).
> **Author:** Le Dinh Minh Quan (student 23127460).
> **Scope of this document:** justify the trainable VQA core and rank every candidate model by license, fit, and hardware. This is the FIRST multimodal-vision project in the series (P02–P15 were text/document/OCR; P16 done), so the model story is written from the ground up.
>
> Every Hugging Face id below was verified on the Hub during research. Licenses are stated explicitly; anything non-commercial or undeclared is **FLAGGED** in-line. Do not introduce model/dataset ids beyond the ones listed here.

---

## 1. What the model has to do

P17 takes an **image + a natural-language question about it** and produces a **short answer**:

- a scene of colored shapes + "how many red squares?" -> "2"
- a kitchen photo + "what color is the table?" -> "white"
- "is there a stove?" -> "yes"

The **VQA model is the single trainable artifact** in the whole project. Everything around it — image decode/resize, question tokenization, answer normalization, the agent's 5-point decision logic, the scorer — is pretrained or deterministic and is never trained.

The model is wrapped by a **deterministic agent** (`src/mmqa/agent/`) whose five decisions are: input gate (D1), question-type router (D2), run the VQA model (D3), **calibrated abstention** (D4), and **type-consistency constraint + re-rank** (D5). D3 is the only step that touches the model. This matters enormously for model selection: **the agent does not consume an argmax string — it consumes the model's full top-k distribution + softmax** so it can compute `p_max`, top1–top2 margin, and Shannon entropy. A model that cannot cheaply expose a clean per-answer distribution is a worse fit for this project regardless of raw accuracy.

So the selection criteria, in priority order, are:

1. **Clean confidence signal** — a per-answer softmax distribution the abstention gate (D4) can threshold and a margin/entropy can be computed over.
2. **Type-constrainable output** — the answer space must map onto the agent's type constraints (yes/no -> `{yes,no}`, count -> a number, color -> a color word) so D5 can re-rank within top-k.
3. **Permissive license** — Apache-2.0 / BSD / MIT preferred; llama2 / Qwen-Research / undeclared are flagged and used only as upper-bound comparisons.
4. **Trainable on the target hardware** — fine-tunes comfortably on a free-Colab T4 (16 GB) as the floor; scales up to H100 as an upgrade tier.

These four criteria are what make **ViLT classification** the default rather than a larger generative model.

---

## 2. Two answer regimes

VQA models split into two families, and the choice between them is the central decision of this document.

### Classification VQA (DEFAULT)

A **fixed answer vocabulary** (the canonical **3129-answer VQAv2 label space**) with a linear head: image+question -> logits -> `softmax` over the 3129 answers -> top-k answers, each with a probability.

- **Cleanest metric.** The prediction is a single label from a known closed set; scoring against the official VQA soft accuracy is a direct lookup-and-normalize.
- **Cleanest confidence signal.** The softmax over the vocab *is* the distribution D3 needs. `p_max`, top1–top2 margin, and entropy fall straight out of it — no proxy required. This is exactly what D4 (calibrated abstention) and D5 (type-consistent re-rank) are designed against.
- **Easiest to train and evaluate.** A classification head over 3129 labels is a standard HF-Trainer fine-tune; metrics are deterministic.
- **Maps onto the agent constraints.** The 3129 vocab already contains `yes`, `no`, the small integers, and the color words, so the type-consistency sets at D5 are literally subsets of the head's output classes.

**This is the regime the agent is built for.** It is the default.

### Generative VQA (ALTERNATIVE)

**Autoregressive decoding** of free-form answer text (open vocabulary). Handles answers outside the fixed 3129 vocab; the confidence proxy is the sequence/token logprob (or beam score), not a clean per-answer softmax.

- **Pro:** no fixed-vocab ceiling — can produce answers the classification head structurally cannot reach.
- **Con:** harder to evaluate (decode -> apply the full canonical VQAv2 normalization -> exact-match) and the confidence signal is a logprob proxy that does not slot as cleanly into D4's `p_max`/margin/entropy gate.

Generative VQA is shipped as a **complement and an upper-bound comparison**, not the trainable default.

> **Architectural key point:** classification VQA gives a per-answer softmax distribution that is the cleanest input for a calibrated abstention gate; generative VQA must fall back to logprob/beam-score as a confidence proxy. The agent targets the classification regime (and the explicit-`ViltProcessor` path that exposes raw logits).

---

## 3. The default: ViLT classification

**`dandelin/vilt-b32-finetuned-vqa`** — **apache-2.0**, ~113M params.

- **Architecture:** ViLT (Vision-and-Language Transformer) — a **single-stream** VL transformer. Image patches and question WordPiece tokens are concatenated and fed through one shared transformer; there is no heavy separate object detector or region-feature extractor, which is why ViLT is light and fast.
- **Head:** a **classification head over ~3129 answers** -> logits -> softmax. This *is* the canonical VQAv2 answer vocabulary.
- **Processor:** `ViltProcessor` = `ViltImageProcessor` (RGB convert, resize to ~384px, normalize) + `BertTokenizerFast` (`[CLS] question [SEP]`).
- **Model class:** `ViltForQuestionAnswering`.

### Why ViLT classification is the default

1. **Cleanest metric + confidence.** The softmax over 3129 answers is exactly the distribution the agent's D3 needs; `p_max`, margin, and entropy are immediate. No logprob proxy.
2. **Maps directly onto the agent's constraints.** The 3129 vocab contains `yes`/`no`, integers, and color words, so D5's type-consistency sets are subsets of the model's own classes — re-ranking within top-k is a filtered argmax.
3. **Apache-2.0 — the cleanest permissive license** of any candidate. No commercial caveat, no flag.
4. **Smallest trainable core (~113M).** Fine-tunes comfortably on a free-Colab T4 at batch 16–32 @384px — the hardware floor the project targets.
5. **HF-Trainer-friendly.** Standard classification fine-tune; no LoRA gymnastics required to fit the default tier.
6. **Already the answer-vocab source.** The canonical 3129 answers come from this model's `config.json` `id2label`/`label2id` (`len == 3129`). Using ViLT as the core means the model, the metric buckets, and the agent's constraint sets all share one vocabulary by construction.

### Two production paths the notebook demonstrates

| Path | API | What it exposes | Used by |
|---|---|---|---|
| Quick classification | `transformers.pipeline('visual-question-answering', model='dandelin/vilt-b32-finetuned-vqa')` | `[{answer, score}, ...]` top-k | demos / smoke tests |
| **Agent path** | explicit `ViltProcessor` + `ViltForQuestionAnswering` | raw `outputs.logits` -> softmax -> top-k + full distribution | **the agent (D3) — it needs the whole distribution, not just argmax** |

### Fine-tuning a custom answer vocabulary

If a different/smaller answer vocab is needed, fine-tune **from the base** `dandelin/vilt-b32-mlm` (**apache-2.0**, ~113M) via `ViltForQuestionAnswering(num_labels=your_vocab)` + `ViltProcessor`. **Caveat:** if you change the vocab, the metric, the D5 answer-constraint sets, and the report buckets must ALL use the same new vocab — the 3129-answer ceiling moves with it.

---

## 4. The generative alternative

For answers outside the fixed 3129 vocab, the project ships a generative tier as a complement and upper-bound. Ranked by fit:

| Role | id | License | Params | Class + Processor |
|---|---|---|---|---|
| **Generative default** | `Salesforce/blip-vqa-base` | bsd-3-clause | 384.7M | `BlipForQuestionAnswering` + `BlipProcessor` (`BlipImageProcessor` + `BertTokenizerFast`); `.generate()` |
| Stronger generative (A100) | `Salesforce/blip-vqa-capfilt-large` | bsd-3-clause | ~470M | `BlipForQuestionAnswering` + `BlipProcessor` |
| Lightweight generative (T4) | `microsoft/git-base-vqav2` | mit | 177.2M | `AutoModelForCausalLM` + `AutoProcessor` (GitProcessor: CLIP image proc + `BertTokenizer`) |

- **BLIP** (`blip-vqa-base`, BSD-3) is the natural open-vocab complement to ViLT: an image encoder + Q-Former / cross-attention text decoder producing free-text answers. It is the third production entry point in the notebook (`BlipProcessor` + `BlipForQuestionAnswering` with `.generate()`). It is ~3.4× the size of ViLT and trades the clean softmax for a sequence logprob.
- **`blip-vqa-capfilt-large`** (BSD-3, ~470M) is the stronger generative variant for the A100/L4 tier.
- **GIT** (`microsoft/git-base-vqav2`, **MIT**, 177M) is a small generative causal-LM contrast — light enough for T4, MIT-clean. Useful as a "small generative vs small classification" comparison against ViLT.

> Upstream reference: Salesforce LAVIS `load_model_and_preprocess('blip_vqa')` is the origin of the HF BLIP port. **Cite it; do not depend on it** — the project uses the HF `transformers` ports only.

Every generative answer must pass through the **single canonical VQAv2 normalization** (lowercase, strip punctuation, drop articles, number-word + contraction mapping) before exact-match scoring; otherwise correct answers score as misses. Keep ONE normalization function shared by predictions and references.

---

## 5. The H100 upgrade tier

When an H100 (or comparable) is available, the project can swap the core for a much larger VLM via bf16 + LoRA / Q-Former-only tuning. Ranked by fit to *this* project (constrained, type-aware decoding):

| Role | id | License | Params | Class + Processor | Tuning |
|---|---|---|---|---|---|
| **RECOMMENDED H100 upgrade** | `Salesforce/blip2-flan-t5-xl` | **mit** | ~3.94B | `Blip2ForConditionalGeneration` + `Blip2Processor` (image proc + `T5TokenizerFast`) | bf16 + LoRA / Q-Former-only |
| H100 alternative (decoder-only) | `Salesforce/blip2-opt-2.7b` | mit* | ~3.74B | `Blip2ForConditionalGeneration` + `Blip2Processor` (image proc + `GPT2TokenizerFast`) | bf16 + LoRA |
| Large permissive instruction-VLM | `Qwen/Qwen2-VL-2B-Instruct` | **apache-2.0** | ~2.21B | `Qwen2VLForConditionalGeneration` + `AutoProcessor` (+ qwen-vl-utils) | LoRA |
| Top H100 instruction-VLM | `Qwen/Qwen2-VL-7B-Instruct` | **apache-2.0** | ~8.29B | `Qwen2VLForConditionalGeneration` + `AutoProcessor` | LoRA only |

- **`Salesforce/blip2-flan-t5-xl` is the recommended upgrade** (MIT, ~3.9B). Its **encoder-decoder T5** backbone gives the cleanest **constrained / instruction-style decoding**, which is exactly what the agent's D5 type constraints want ("answer with a number", "answer yes or no"). It also brings strong zero-shot performance, useful as a reference baseline. MIT-clean.
- **`Salesforce/blip2-opt-2.7b`** (MIT in HF metadata, ~3.7B) is the decoder-only alternative. *Note:* the underlying OPT weights carry an upstream Meta research-use caveat even though HF metadata declares `mit` — keep that in mind for commercial use.
- **`Qwen/Qwen2-VL-2B-Instruct`** (**apache-2.0**, ~2.2B) is the clean modern instruction-VLM option; **`Qwen2-VL-7B-Instruct`** (also apache-2.0) is the top tier but ~8B is overkill for a fine-tunable core — reserve it for the top tier, LoRA only.

All H100-tier models are **generative**, so they revert to the logprob/beam-score confidence proxy. They are upgrades for raw accuracy and zero-shot strength, not replacements for ViLT's clean classification confidence signal.

---

## 6. T4 fallback (16 GB, free Colab — the hardware floor)

The project must run on a free-Colab T4. On that floor:

- **DEFAULT `dandelin/vilt-b32-finetuned-vqa`** (or fine-tune from `dandelin/vilt-b32-mlm`): ViLT ~113M, classification head over ~3129 answers, **trains comfortably on T4 at batch 16–32 @384px**. This is why ViLT is the default — it is the only candidate that *fully fine-tunes* on the floor hardware.
- `Salesforce/blip-vqa-base` (~385M, generative): fits T4 for **inference and LoRA / small-batch fine-tune**.
- `microsoft/git-base-vqav2` (177M, MIT): small generative contrast that also fits T4.

---

## 7. GPU-tier table

| Tier | VRAM | Default model | Precision | Batch | Tuning | Processor |
|---|---|---|---|---|---|---|
| **T4** (free Colab) | 16 GB | `dandelin/vilt-b32-finetuned-vqa` (classification, ~113M) | fp16/fp32 | 16–32 @384px | full fine-tune | `ViltProcessor` (`ViltImageProcessor` + `BertTokenizerFast`) |
| T4 (generative) | 16 GB | `Salesforce/blip-vqa-base` (~385M) / `microsoft/git-base-vqav2` (177M) | fp16 | small / LoRA | LoRA / small-batch | `BlipProcessor` / `AutoProcessor` (GitProcessor) |
| **L4 / A100** (default tier) | 24 / 40–80 GB | ViLT full-speed; `Salesforce/blip-vqa-base`, `blip-vqa-capfilt-large` | fp16/bf16 | 32–64 | full fine-tune | `ViltProcessor` / `BlipProcessor` |
| **H100** (upgrade) | 80 GB | `Salesforce/blip2-flan-t5-xl` (~3.9B, MIT) *recommended*; `blip2-opt-2.7b`; `Qwen2-VL-2B/7B` (apache) | **bf16 + LoRA** / Q-Former-only | per-LoRA | LoRA / Q-Former-only | `Blip2Processor` (image proc + `T5TokenizerFast` / `GPT2TokenizerFast`); `AutoProcessor` (+ qwen-vl-utils) |

Processor pairing summary:

- **ViLT** -> `ViltProcessor` = `ViltImageProcessor` + `BertTokenizerFast`.
- **BLIP** -> `BlipProcessor` = `BlipImageProcessor` + `BertTokenizerFast`.
- **GIT** -> `AutoProcessor` (GitProcessor) = CLIP image processor + `BertTokenizer`.
- **BLIP-2** -> `Blip2Processor` = image processor + `T5TokenizerFast` (flan-t5-xl) or `GPT2TokenizerFast` (opt-2.7b).
- **Qwen2-VL** -> `AutoProcessor` (+ `qwen-vl-utils`).

---

## 8. AVOID / FLAG

| id | Status | Reason |
|---|---|---|
| `google/pix2struct-vqav2-base` | **DOES NOT EXIST (404)** | This id is not on the Hub. Pix2Struct ships only document/infographic variants (`pix2struct-infographics-vqa-*`, `pix2struct-ai2d-base`) that target rendered text/diagrams/screenshots — **wrong fit for natural-image scene VQA.** Never reference this id. |
| `llava-hf/llava-1.5-7b-hf` | **FLAG — `llama2` license** | Custom Llama 2 Community License: restricted-use / non-commercial-ish, **not clean permissive**. Also 7B is too heavy for the trainable core. Use only as a top-tier generative upper-bound comparison, never as default. |
| `Qwen/Qwen2.5-VL-3B-Instruct` | **FLAG — Qwen Research (non-commercial); HF license field EMPTY** | Non-commercial research license; metadata undeclared. FLAG before any commercial use. (Note: the **7B** Qwen2-VL is apache-2.0 and clean; the **3B** Qwen2.5 is not.) |
| `Qwen/Qwen2-VL-7B-Instruct` | license-clean (apache-2.0) but **overkill** | Apache-clean, but ~8B params is overkill for a fine-tunable classification core. Reserve for the top tier only, LoRA only. |

---

## 9. How the choice ties back to the agent

The default model is chosen *because* of the agent, not in spite of it:

- **D3 (run VQA)** wants a full top-k + softmax. ViLT's classification head delivers exactly that with no proxy.
- **D4 (calibrated abstention)** thresholds `p_max`, top1–top2 margin, and entropy — all of which are well-defined over a softmax distribution and ill-defined over a generative logprob. VQA classifiers are **overconfident**, so D4 uses temperature-scaled thresholds tuned on a held-out split; that calibration story is much cleaner over a classification softmax.
- **D5 (type-consistency re-rank)** filters the top-k to the answer type's allowed set (yes/no -> `{yes,no}`, count -> a number, color -> a color word). Because those words are already classes in the 3129 vocab, re-ranking is a filtered argmax over the existing distribution — no decoding, no re-generation.

The generative and H100 tiers exist as upper-bound comparisons and open-vocab escape hatches, but they revert to a logprob confidence proxy, which is why they are alternatives rather than the default.

> **Offline note:** for CI / torch-free / network-free runs the model is replaced by a `SceneStubVQA` stub (reads the scene spec embedded in the synthetic PNG and returns a realistic top-k distribution, not one-hot). Swapping `SceneStubVQA` for the real `ViltForQuestionAnswering` wrapper — same `predict(image, question) -> {topk, p_max, entropy, margin, regime}` signature — flips the system to production with **no agent changes**. This is only possible because the default model's interface is a clean top-k distribution.

---

## 10. Decision summary

- **Ship `dandelin/vilt-b32-finetuned-vqa` (apache-2.0, ~113M) as the trainable core.** It is the cleanest permissive license, the smallest model that fully fine-tunes on a free T4, a classification head over the canonical 3129 answers (clearest metric and confidence), already the project's answer-vocab source, and it maps directly onto the agent's type-constraint (D5) and calibrated-abstention (D4) gates.
- **Custom-vocab base:** `dandelin/vilt-b32-mlm` (apache-2.0).
- **Generative alternative:** `Salesforce/blip-vqa-base` (bsd-3, 385M), with `blip-vqa-capfilt-large` (bsd-3) stronger and `microsoft/git-base-vqav2` (mit, 177M) lighter.
- **H100 upgrade:** `Salesforce/blip2-flan-t5-xl` (mit, ~3.9B) recommended for clean constrained T5 decoding; `blip2-opt-2.7b` (mit) and `Qwen2-VL-2B/7B-Instruct` (apache) alternatives.
- **Never default to:** `llava-hf/llava-1.5-7b-hf` (llama2 — FLAG) or `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen Research / empty license — FLAG); use only as flagged upper-bound comparisons. **Never reference** `google/pix2struct-vqav2-base` (404, does not exist).
