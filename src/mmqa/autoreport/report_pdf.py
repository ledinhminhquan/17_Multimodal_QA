"""Generate the submission report.pdf for the mmqa (VQA) system.

A 10-15 page report covering every Section-I deliverable: problem & use cases, data
(VQAv2 + synthetic scenes), the ViLT VQA core, the agent (D1-D5: classify, abstain,
constrain), the evaluation (VQA accuracy + per-type + abstention), deployment, continual
learning & monitoring, privacy & robustness, and ethics. Live numbers come from
``run_dir()`` artifacts; missing metrics degrade to placeholders. reportlab lazy-imported;
a Markdown fallback is written if absent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso
from . import charts as charts_mod
from .artifact_loader import (agent_metric, base_model, has_eval, headline, latency, load_artifacts,
                              model_system_name, model_version, read_doc, sys_accuracy)

logger = get_logger(__name__)

_SUBTITLE = ("Visual Question Answering: given an image and a natural-language question, a TRAINABLE "
             "VQA core (ViLT) produces an answer, orchestrated by a deterministic agent (D1-D5) that "
             "classifies the question type, ABSTAINS when uncertain (VQA models are overconfident), and "
             "CONSTRAINS the answer to the question type. Reports the official VQA accuracy + per-type + "
             "the abstention/coverage trade-off.")

_SECTIONS = [
    ("1. Problem Definition & Use Cases", "problem_definition.md"),
    ("2. Data (VQAv2 + Synthetic Scenes)", "data_description.md"),
    ("3. System Architecture", "architecture.md"),
    ("4. Model Selection (the VQA core)", "model_selection.md"),
    ("5. Agent Architecture (Decisions D1-D5)", "agent_architecture.md"),
    ("6. Evaluation Methodology", "vqa_evaluation.md"),
    ("7. Deployment", "deployment.md"),
    ("8. Continual Learning & Monitoring", "continual_learning_monitoring.md"),
    ("9. Data Privacy & Robustness", "privacy_robustness.md"),
    ("10. Ethics & Responsible AI", "ethics_statement.md"),
]


def _builtin_sections(cfg: AppConfig, arts: Dict[str, Any]) -> Dict[str, str]:
    mname = model_system_name(arts) or "model"
    macc = sys_accuracy(arts, mname)
    bacc = sys_accuracy(arts, "blind_prior")
    cov = agent_metric(arts, "coverage")
    if macc is not None and bacc is not None:
        res_line = (f"In the latest eval the VQA core reaches accuracy **{macc*100:.1f}%** vs the blind "
                    f"question-only prior **{bacc*100:.1f}%**"
                    + (f"; the agent answers **{cov*100:.0f}%** of questions (abstaining on the rest)."
                       if cov is not None else "."))
    else:
        res_line = "Run `mmqa evaluate` to populate the live numbers here."
    return {
        "problem_definition.md": f"""
## What it does
Given an **image** and a **natural-language question** about it, produce a short answer
(e.g. a scene + "how many red squares?" -> "2"). The trainable core is the **VQA model**;
the question-type router, the calibrated-abstention gate and the answer constraints are
algorithmic. Default model **{base_model(arts)}** (ViLT, classification over ~3129 answers).

## The job-to-be-done
- **Accessibility** - answer a blind user's questions about a photo (a la VizWiz).
- **E-commerce / search** - "what colour is this product?", "is there a logo?".
- **Education / triage** - quick visual question answering with a confidence signal.

## Why it is more than a raw model
VQA models are **overconfident** and exploit **language priors** (answering "yes"/"2"/"white"
without really looking). The agent's value-add is **calibrated abstention** (say "unsure"
instead of guessing) + a **type-consistency constraint** (a yes/no question gets a yes/no
answer, a counting question a number) - reported alongside the blind-prior baseline so the
language bias is visible.

## Success metrics
- **Technical:** the official **VQA accuracy** (soft, 10-annotator) + per-answer-type + the
  **abstention/coverage** trade-off.
- **Business:** human review load (flagged answers), accuracy on the answered slice.
{res_line}
""",
        "model_selection.md": f"""
## The trainable VQA core
- **Default:** `{base_model(arts)}` (ViLT, **Apache-2.0**, ~113M) - a single-stream
  vision-language transformer with a **classification head over ~3129 answers**: the
  cleanest metric (VQA accuracy / top-1 over a fixed vocab), easiest to train AND eval, and
  the softmax maps directly onto the agent's confidence threshold + type constraint.
- **Generative alternative:** `Salesforce/blip-vqa-base` (BSD-3, open-vocabulary free-text);
  lightweight `microsoft/git-base-vqav2` (MIT). **H100 upgrade:** `Salesforce/blip2-flan-t5-xl`
  (MIT, ~3.9B, clean constrained decoding).
- **Baselines:** the **most-common-answer** floor ("yes") and a **blind / question-only**
  prior (answers from the question type alone) that measures the language-prior bias.

## Training
ViLT fine-tune with HF Trainer: the target is a **soft vector** where each annotator answer
in the vocab gets score min(1, count/3) (BCE-with-logits), selected on **VQA accuracy**;
bf16+tf32 on Ampere+/H100, fp16 on T4; resume-safe; early stopping.
{res_line}
""",
        "agent_architecture.md": f"""
## FSM
A deterministic finite-state machine; every tool returns a uniform dict and every transition
is traced. States: `ingest -> classify -> answer -> calibrate -> constrain`. An optional LLM
**brain** (`{cfg.agent.llm_model}`, OFF by default) only adds an advisory note; rules win and
the agent runs with **zero paid API calls**.

## Five decisions (each acts on an intermediate artifact)
- **D1 - input gate.** A valid image + a non-empty question (>= {cfg.agent.min_question_chars}
  chars); flags a blank image.
- **D2 - question-type classification.** yes_no / number / color / object / location / person /
  reason / other - routes the answer constraint (D5).
- **D3 - answer.** Run the VQA model -> top-{cfg.model.top_k} candidates + softmax confidence,
  margin, entropy.
- **D4 - calibrated abstention gate.** Max-prob < {cfg.agent.confidence_min}, or margin <
  {cfg.agent.margin_min}, or entropy > {cfg.agent.entropy_max} -> **abstain** ("unsure" +
  needs_review). VQA models are overconfident, so this is the safety valve.
- **D5 - type-consistency gate.** Constrain the answer to the question type (yes/no -> yes/no,
  count -> a number, colour -> a colour word); re-rank within the top-k to the best
  type-consistent candidate.

The agent emits `{{answer, qtype, confidence, abstained, type_constrained, candidates,
decisions[], trace[]}}`. Low-confidence answers are **flagged for human review**.
""",
        "vqa_evaluation.md": f"""
## The metric
The official **VQA accuracy**: for a prediction, the leave-one-out average over the 10
annotators of min(1, matches/3), after the standard answer normalization (lowercase, strip
punctuation, drop articles a/an/the, map number-words + contractions). Reported overall and
**per answer-type** (yes/no, number, other).

## Baselines & the abstention story
- **most-common-answer** ("yes") and the **blind question-only prior** (the language-bias floor).
- the **agent**: coverage (fraction answered), accuracy-on-answered, and the overall accuracy
  (abstentions count as wrong) - the calibrated-abstention precision/coverage trade-off.
{res_line}
""",
    }


def _esc(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b\x00").replace("</b>", "\x00/b\x00")
    s = s.replace("<font face='Courier'>", "\x00f\x00").replace("</font>", "\x00/f\x00")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = (s.replace("\x00b\x00", "<b>").replace("\x00/b\x00", "</b>")
          .replace("\x00f\x00", "<font face='Courier'>").replace("\x00/f\x00", "</font>"))
    return s


def _md_to_flowables(md: str, styles, max_lines: int = 300):
    from reportlab.platypus import Paragraph, Preformatted, Spacer
    flow, lines, in_code, code, bullet = [], md.splitlines()[:max_lines], False, [], []

    def flush():
        nonlocal bullet
        for b in bullet:
            flow.append(Paragraph("- " + _esc(b), styles["Body"]))
        bullet = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code), styles["Code"])); code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(ln); continue
        s = ln.rstrip()
        if not s:
            flush(); flow.append(Spacer(1, 5)); continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            flow.append(Paragraph(_esc(s.lstrip("#").strip()), styles["H2" if level <= 2 else "H3"]))
        elif s.lstrip().startswith(("- ", "* ")):
            bullet.append(s.lstrip()[2:])
        else:
            flush(); flow.append(Paragraph(_esc(s), styles["Body"]))
    flush()
    return flow


def _fmt_pct(v):
    return (f"{v*100:.1f}%" if isinstance(v, (int, float)) and not isinstance(v, bool) else "-")


def _results_tables(arts: Dict[str, Any], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    mname = model_system_name(arts) or "model"
    flow = [Paragraph("Results - VQA accuracy", styles["H3"])]
    rows = [["System", "VQA accuracy"],
            ["most-common ('yes') floor", _fmt_pct(sys_accuracy(arts, "most_common"))],
            ["blind question-only prior", _fmt_pct(sys_accuracy(arts, "blind_prior"))],
            [f"model ({mname})", _fmt_pct(sys_accuracy(arts, mname))]] if has_eval(arts) else \
           [["System", "VQA accuracy"], ["run `evaluate`", "-"]]
    t = Table(rows, hAlign="LEFT", colWidths=[240, 120])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    flow += [t, Spacer(1, 8), Paragraph("Results - the agent (abstention trade-off)", styles["H3"])]
    arows = [["Metric", "value"],
             ["Coverage (answered)", _fmt_pct(agent_metric(arts, "coverage"))],
             ["Accuracy on answered", _fmt_pct(agent_metric(arts, "accuracy_on_answered"))],
             ["Overall accuracy (abstain=wrong)", _fmt_pct(agent_metric(arts, "overall_accuracy"))],
             ["Abstain rate", _fmt_pct(agent_metric(arts, "abstain_rate"))]]
    at = Table(arows, hAlign="LEFT", colWidths=[260, 120])
    at.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f855a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf5ee")])]))
    flow += [at, Spacer(1, 6),
             Paragraph("The blind-prior accuracy shows the language bias; abstention buys accuracy "
                       "on the answered slice at the cost of coverage.", styles["Body"])]
    lat = latency(arts, "p50")
    if lat is not None:
        flow.append(Paragraph(f"Agent latency: per-question p50 ~ {lat:.0f} ms "
                              f"(p95 ~ {latency(arts, 'p95') or 0:.0f} ms).", styles["Body"]))
    flow.append(Spacer(1, 8))
    return flow


def generate_report(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    title = title or cfg.project_title
    author = author or cfg.author
    arts = load_artifacts(cfg)
    out = Path(out_path) if out_path else run_dir() / "report" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    builtins = _builtin_sections(cfg, arts)

    def section_md(fname: str) -> str:
        doc = read_doc(fname)
        if doc.strip():
            lines = doc.splitlines()
            return "\n".join(lines[:46]) if len(lines) > 46 else doc
        return builtins.get(fname, "")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer)
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("T", parent=base["Title"], fontSize=22, leading=26),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], textColor="#1a365d", spaceBefore=10),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], textColor="#2b6cb0"),
        "Body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13),
        "Code": ParagraphStyle("C", parent=base["Code"], fontSize=7.5, leading=9, backColor="#f4f6f8"),
        "Meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=11, leading=15),
    }
    try:
        built = dict(charts_mod.build_all(arts, out.parent / "charts"))
    except Exception as exc:
        logger.info("charts skipped (%s)", exc)
        built = {}

    story: List[Any] = [
        Spacer(1, 5 * cm), Paragraph(title, styles["Title"]), Spacer(1, 1 * cm),
        Paragraph(f"<b>{author}</b> - Student {cfg.student_id}", styles["Meta"]),
        Paragraph("NLP in Industry - Final Assignment (P17)", styles["Meta"]),
        Paragraph(_SUBTITLE, styles["Meta"]),
        Paragraph(f"Generated {utc_now_iso()}", styles["Body"]),
        Paragraph(f"VQA core: <b>{model_version(arts)}</b> (base {base_model(arts)})", styles["Body"]),
    ]
    story.append(PageBreak())
    story += _results_tables(arts, styles)
    for name in ("accuracy", "agent", "buckets"):
        if name in built:
            story += [Image(str(built[name]), width=13 * cm, height=7.0 * cm), Spacer(1, 6)]
    story.append(PageBreak())

    for heading, fname in _SECTIONS:
        story.append(Paragraph(heading, styles["H2"]))
        story += _md_to_flowables(section_md(fname), styles)
        story.append(Spacer(1, 10))

    try:
        SimpleDocTemplate(str(out), pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                          leftMargin=1.8 * cm, rightMargin=1.8 * cm, title=title, author=author).build(story)
    except Exception as exc:
        logger.warning("reportlab build failed (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)
    logger.info("Report -> %s", out)
    return str(out)


__all__ = ["generate_report"]
