"""Gradio demo UI for Visual Question Answering.

Upload an image (or use a synthetic sample scene), type a question -> the agent classifies
the question, runs the VQA model, abstains if unsure, and constrains the answer to the type.
Heavy deps (gradio/PIL) are imported lazily so the package imports without them.
"""

from __future__ import annotations

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_demo(cfg: AppConfig = None, load_model: bool = True):
    import gradio as gr
    from ..agent.vqa_agent import VqaAgent
    from ..data import samples
    from ..data.synth_scene import render_scene

    cfg = cfg or AppConfig()
    agent = VqaAgent(cfg, load_model=load_model)
    sample_scene = samples.scenes()[0]
    try:
        sample_img = render_scene(sample_scene)
    except Exception:
        sample_img = None

    def run(image, question):
        if image is None or not (question or "").strip():
            return "Please provide an image and a question.", ""
        out = agent.ask(question, image=image)
        ans = out["answer"]
        info = (f"type={out['qtype']} | confidence={out['confidence']} | "
                f"abstained={out['abstained']} | type_constrained={out['type_constrained']} | "
                f"top-k={out['candidates'][:3]}")
        return ans, info

    with gr.Blocks(title=cfg.serving.api_title) as demo:
        gr.Markdown(f"# {cfg.serving.api_title}\n"
                    "Ask a question about an image. The agent classifies the question, runs the VQA "
                    "model, **abstains** when unsure, and **constrains** the answer to the question type "
                    "(yes/no, count, colour).")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Image", value=sample_img)
                q = gr.Textbox(label="Question", value="how many shapes are there?")
                btn = gr.Button("Answer", variant="primary")
            with gr.Column():
                ans = gr.Textbox(label="Answer", lines=1)
                info = gr.Textbox(label="Pipeline trace", lines=3)
        btn.click(run, inputs=[inp, q], outputs=[ans, info])
        gr.Markdown("_Assistive VQA; low-confidence answers are returned as 'unsure' for human review._")
    return demo


def launch(cfg: AppConfig = None, share: bool = False, **kwargs):
    build_demo(cfg).launch(share=share, **kwargs)


__all__ = ["build_demo", "launch"]
