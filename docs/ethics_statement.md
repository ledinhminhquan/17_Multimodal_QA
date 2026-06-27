# Ethics Statement — P17 Multimodal Question Answering (VQA)

**Project:** P17 Multimodal Question Answering (Visual Question Answering, package `mmqa`)
**Author:** Le Dinh Minh Quan (student 23127460)
**Scope of this document:** the ethical commitments, known harms, and mitigations that govern how the P17 VQA system answers questions about images.

---

## 1. Why VQA needs its own ethics statement

P17 is this portfolio's **first multimodal-vision project** (P02–P16 were text, document, and OCR systems). That shift changes the risk surface in a way that text-only NLP does not face. A VQA system takes a **photograph the user supplies** plus a **natural-language question** about it and returns a short answer — "how many chairs are there?" → "2"; "what color is the table?" → "white"; "is there a stove?" → "yes". Three properties make this materially more sensitive than the earlier projects:

1. **The input is a raw photograph.** Photos carry faces, homes, screens, documents, medication labels, location cues, and bystanders who never consented. A question about an image is rarely just about the pixels the user noticed.
2. **The answer is consumed as fact.** A terse, fluent "yes" or "2" or "white" reads as confident ground truth, even when the model is guessing. There is no hedging in a one-word answer unless the system adds it.
3. **A primary, legitimate use is accessibility.** The most important real-world VQA users are blind and low-vision people (the VizWiz benchmark — `lmms-lab/VizWiz-VQA` — is built from exactly these real assistive photos). For them the answer is not a convenience; it is the only channel to the visual world, and they cannot cross-check it against the image.

The remainder of this document states how P17 responds to each of these.

---

## 2. Over-trust in high-stakes use

### 2.1 The core failure mode

A stock ViLT or BLIP model does **raw argmax**: it always emits its single most probable answer, with no notion of "I don't know." VQA classifiers are additionally **poorly calibrated and skew overconfident** — the softmax probability attached to a wrong answer is routinely high. The combination is dangerous: a confident-looking wrong answer delivered to someone who cannot verify it.

This matters most in two settings P17 explicitly anticipates:

- **Accessibility for blind / low-vision users.** "Is this medication the blue pill or the white one?", "Is the stove off?", "What does this sign say?", "Is the baby's bottle empty?". A wrong-but-confident answer here is not a UX defect — it can cause a real-world harm the user has no way to catch.
- **Medical, safety, or legal images.** Photographs of skin, X-rays, lab readouts, dosage labels, warning signs, or legal documents. A VQA model trained on COCO scenes is **out of its competence domain** on these, yet will still answer.

### 2.2 The P17 commitment: assist and abstain, never assert certainty

P17's design principle is that **the tool assists; it never asserts certainty.** This is not a slogan — it is enforced by a dedicated decision point in the agent, the **D4 Calibrated Abstention Gate**:

- D4 does **not** trust raw `p_max`. Because VQA softmax is overconfident, the gate combines three signals — maximum probability `p_max` vs `tau_conf`, Shannon entropy `H` of the answer distribution vs `tau_ent`, and the top1–top2 margin `(p1 − p2)` vs `tau_margin` — and these thresholds are **temperature-calibrated on a held-out split**, not hard-coded.
- When the model is not confident enough, the system **abstains**: it returns `answer = "unsure"` with `status = "abstained:low_confidence"` and a `needs_review` flag, instead of emitting a guess.
- Thresholds can be set **per question type** (counting is harder than yes/no, so it is held to a stricter bar).

The headline reliability metric for P17 is therefore **answer-when-confident accuracy plus abstention rate**, reported against a raw-argmax baseline — i.e. we explicitly measure the precision/coverage trade that abstention buys. A system that says "unsure" on the questions it would have gotten wrong is the intended, ethically preferable behaviour, even though it answers fewer questions.

### 2.3 Domain refusal, not just low-confidence abstention

Calibrated abstention catches *uncertainty within the model's domain*. It does **not** by itself catch *out-of-domain* high-stakes use, where the model may be confidently wrong (a medical image that superficially resembles a COCO scene). P17 therefore treats **medical, diagnostic, dosage, legal, and safety-critical questions as out of scope**, and the deployment surface (README, Gradio UI, API docs) must state plainly that:

> P17 answers everyday questions about ordinary scenes. It is **not** a medical, diagnostic, legal, or safety device. Do not rely on it for medication identification, dosage, hazard assessment, or any decision where a wrong answer causes harm.

For assistive use, the same surfaces state that the answer is a **best-effort aid, not a verified fact**, and that the abstention/confidence signal must be surfaced to the end user, never silently dropped.

---

## 3. Language-prior and demographic bias

### 3.1 The language-prior shortcut

VQA models have a well-documented failure: they answer **from the question text alone, ignoring the image**. "What color is the banana?" → "yellow" regardless of what is actually shown; "how many...?" → "2"; "is there...?" → "yes". The model has learned the *distribution of human answers conditioned on the question wording* rather than *looking at the picture*. For an accessibility user this is the worst possible bias — the entire point of the tool is that they cannot see the image, so an answer that secretly ignores the image is an answer that lies to exactly the person who depends on it most.

### 3.2 How P17 surfaces it (rather than hiding it)

P17 does not claim to eliminate the language prior — it **measures and publishes** it, so the bias is visible rather than buried:

- **Blind / question-only baseline (mandatory).** A classifier trained on the **question text with no image** is reported alongside the full model. The gap between the full model and this blind baseline is the honest measure of *how much the image actually contributes*. A small gap is a red flag that the system is reading the question, not the picture.
- **Most-common-answer (prior) baseline.** Always answering the per-type majority ("yes" / "2" / "white") is reported as the floor.
- **Per-answer-type accuracy, always, broken out.** Every evaluation reports `overall`, `yes/no`, `number`, and `other` accuracy separately (×100). This exposes the characteristic profile — yes/no high, number lowest — and prevents a strong aggregate number from masking weak counting or weak open-vocab performance.
- **Per-question-type table (65 buckets)** as an appendix, so systematic weaknesses on specific phrasings ("why...", "what color...") are auditable.

The use of **VQAv2** (`lmms-lab/VQAv2`, CC-BY-4.0) as the primary evaluation set is itself a bias mitigation: VQAv2 was deliberately constructed with complementary image pairs to *depress* the language prior. It does not remove it — hence the blind baseline is still reported every time.

### 3.3 Demographic bias from the training source

The training and evaluation images are **COCO 2014/2015 scenes**. COCO is a web-scraped, geographically and culturally skewed collection: it over-represents certain regions, household objects, and contexts, and under-represents others. Models trained on it inherit those skews — object vocabulary, what counts as a "normal" kitchen or a "normal" wedding, and answers about people. P17's commitments here:

- The system is **not used to infer protected or sensitive attributes** of people in images (race, ethnicity, gender identity, age, religion, health, sexual orientation, disability). Questions of that shape are out of scope; the agent's question-type router and out-of-scope handling treat them as refuse/flag rather than answer.
- The 3129-answer classification vocabulary (from `dandelin/vilt-b32-finetuned-vqa`) is a **fixed ceiling**: any answer outside it is unreachable. Documentation states this so users do not mistake "not in vocab" for "not true."
- Performance is reported per-type rather than as a single number precisely so that uneven competence across answer categories is not hidden.

---

## 4. Privacy and consent

User-supplied photographs are the most sensitive data in the entire P02–P17 portfolio. A single image can contain faces of non-consenting bystanders, the interior of a home, a computer or phone screen, a passport or ID, a prescription label, a medical scan, or precise location cues. P17 treats every uploaded image as **private personal data by default.**

### 4.1 Data-handling commitments

- **No raw-image retention by default.** Uploaded images are processed in memory to produce an answer and are **not persisted** to disk or logs once the request completes. Any retention (for debugging or improvement) must be explicit, opt-in, time-limited, and disclosed — never the default.
- **Local / self-hosted processing is the default deployment.** The default core is a small, permissively-licensed model (`dandelin/vilt-b32-finetuned-vqa`, Apache-2.0, ~113M) that runs locally, so images need not leave the user's machine. The optional LLM "brain" (Anthropic) is **off by default, advisory only, and never sees the raw image unless the operator explicitly enables it.**
- **Logs are about decisions, not pixels.** The structured per-decision trace (D1–D5) and metrics logging record *status codes, confidence, and question type* — not the image and not its contents. The image is never written to a log line.
- **Synthetic data for tests and CI.** All automated tests, CI, and offline demos run on the **synthetic scene generator** (`data/synth_scene.py`) and the torch-free `SceneStubVQA` — no real user photos, no network, no third-party model download. Development never requires real personal images.

### 4.2 Consent

- **The uploader must have the right to share the image.** The tool must not be used on photographs of other people taken or shared without their consent, and the deployment surface states this.
- **Bystander privacy.** A question about one object in a photo does not grant license to retain or analyse everyone else in the frame. The no-retention default is the practical backstop.
- **Dataset licensing is surfaced, not buried.** Several practical VQAv2 mirrors (`HuggingFaceM4/VQAv2`, `merve/vqav2-small`, `Multimodal-Fatima/*`, `lmms-lab/VizWiz-VQA`, `lmms-lab/OK-VQA`) declare **no license on the repository** even though the COCO/VQA upstream is CC-BY-4.0; these are **FLAGGED as license-unconfirmed** and must be verified before any commercial use. `thangduong0509/daquar_vqa` (CC-BY-NC-SA-4.0, **non-commercial**) is excluded entirely. Respecting data provenance is part of respecting the people in the data.

---

## 5. Transparency

P17's answers are designed to be **legible, not oracular.** Every answer ships with the evidence needed to judge how much to trust it.

- **Confidence is always exposed.** The response carries the calibrated confidence and the top-k candidate distribution, not just the chosen answer. A user (or a downstream UI) can see "yes (0.42)" versus "yes (0.95)" and act accordingly.
- **Abstention is a first-class, visible outcome.** When D4 fires, the status is explicitly `abstained:low_confidence` with `answer = "unsure"` and `needs_review = true`. The system communicates *"I am not sure"* as a real answer, not a silent fallback.
- **The full decision trace is auditable.** The agent is a deterministic 5-point state machine, and each decision point emits a status line: input-gate result (D1), the inferred question type (D2), the model's top-k and `p_max`/entropy/margin (D3), the abstention decision (D4), and any type-consistency re-rank (D5, `status = "ok:reranked"`). A user or reviewer can see *why* an answer was given, *what* it was constrained to, and *whether* it was re-ranked away from a fluent-but-type-wrong argmax (e.g. "cat" → "3" for a counting question).
- **Limits are stated up front.** Documentation discloses the fixed 3129-answer vocabulary ceiling, the COCO domain, the known answer-type performance profile, and the out-of-scope domains — so users calibrate their trust before they rely on the tool.
- **No anthropomorphic certainty.** The tool does not phrase answers as authoritative pronouncements. A one-word answer is always accompanied by its confidence and, when low, by the explicit "unsure."

---

## 6. Responsible-use guidance

**Intended use.** Everyday, low-stakes questions about ordinary photographs — counting common objects, identifying colors, confirming the presence of an object, describing simple scenes — including as an **assistive aid** for blind and low-vision users *with the confidence/abstention signal always surfaced*.

**Out-of-scope / prohibited use.**

- **High-stakes decisions:** medical/diagnostic interpretation, medication identification or dosage, legal or financial determinations, hazard or safety assessment. The model is a COCO-scene answerer, not a domain expert, and must not be the basis for such decisions.
- **Inferring sensitive attributes of people** (race, gender, age, health, religion, sexuality, disability) from their images.
- **Surveillance, tracking, biometric identification, or profiling** of individuals.
- **Processing images the user has no right to use,** or retaining/sharing bystanders' images.
- **Treating any single answer as verified fact,** especially an answer delivered without its confidence signal.

**Operator obligations when deploying P17.**

- Keep the **no-raw-image-retention** default; disclose and obtain consent for any retention.
- **Surface confidence and the abstention/`needs_review` flag to the end user** — never strip them to present a cleaner single-word answer.
- Display the **scope and limits notice** (Section 2.3 / Section 5) at the point of use.
- Honour the **license flags** in `docs/DESIGN_BRIEF.md` §4 before any commercial deployment of flagged datasets or models (e.g. `llava-hf/llava-1.5-7b-hf` llama2, `Qwen/Qwen2.5-VL-3B-Instruct` Qwen-Research are non-permissive and must not be shipped as the default core).

---

## 7. Human-in-the-loop for flagged answers

P17 is built so that uncertainty **routes to a person**, not into a silent guess.

- **Abstention produces a review signal, not a dead end.** Whenever D4 abstains (`status = "abstained:low_confidence"`, `needs_review = true`) or D5 cannot find a type-consistent candidate (`status = "abstained:type_mismatch"`), the item is flagged for human attention rather than answered. In an assistive context the appropriate human-in-the-loop response is to tell the user "I'm not sure about this one" and let them seek another source (a sighted assistant, a second tool, a fresh photo) — never to fabricate confidence.
- **The trace makes human review tractable.** Because every decision point is logged, a reviewer can see exactly where and why an item was flagged: bad input (D1), an ambiguous question type (D2), low/uncertain model distribution (D4), or a type mismatch (D5).
- **The optional LLM brain is advisory only and never overrides.** If the optional Anthropic LLM brain is enabled, it can *advise* but **cannot override** the deterministic gates or force an answer where the model abstained. Automation never silently outranks the safety logic, and a human reviewer remains the final authority on flagged items.
- **High-stakes always routes to a human.** For any question that touches the out-of-scope domains in Section 6, the correct behaviour is to decline and defer to a qualified human, regardless of how confident the model happens to be.

---

## 8. Summary of commitments

| Risk | P17 mitigation |
|---|---|
| Over-trust / confident wrong answers | D4 calibrated abstention (`p_max` + entropy + margin, temperature-tuned) → returns `"unsure"` + `needs_review`; report answer-when-confident accuracy vs raw argmax |
| High-stakes / accessibility / medical misuse | Tool **assists, never asserts certainty**; medical/legal/safety **out of scope**; assistive answers always carry confidence + abstention |
| Language-prior bias (answer ignores image) | **Blind question-only baseline** + most-common-answer prior reported every run; image-contribution gap published |
| Demographic / COCO bias | Per-answer-type and per-question-type accuracy reported; no sensitive-attribute inference; fixed-vocab ceiling disclosed |
| Privacy of user photos | **No raw-image retention by default**; local processing; logs record decisions not pixels; synthetic data for CI |
| Consent / bystanders | Uploader must have rights; no-retention backstop; dataset license flags honoured |
| Opacity | Confidence, top-k, abstention status, and full D1–D5 decision trace exposed; limits stated up front |
| Uncertain answers | Human-in-the-loop: abstention → review signal; LLM brain advisory-only, never overrides; high-stakes always defers to a human |

P17's guiding principle: **a VQA system that admits "I'm not sure" is more trustworthy than one that is always confident and sometimes wrong — especially for the people who cannot check its answer for themselves.**
