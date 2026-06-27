"""Command-line interface - the single entrypoint for the mmqa (VQA) system.

    mmqa <command> [options]

Commands: data, gen-synthetic, train-vqa, train-baseline, tune, evaluate, ask, ask-scene,
demo-agent, serve, benchmark, error-analysis, per-type, monitor-log, generate-report,
generate-slides, autopilot, grade.

All console output is ASCII-only (Windows cp1252 safe); stdout stays pipeable JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Multimodal Question Answering (VQA) System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all
    print(json.dumps(download_all(_load(args), render_synthetic=args.render), indent=2, ensure_ascii=False))


def cmd_gen_synthetic(args):
    from .data.dataset import build_synthetic_eval
    print(json.dumps(build_synthetic_eval(_load(args), n_scenes=args.n_scenes, split=args.split),
                     indent=2, ensure_ascii=False))


def cmd_train_vqa(args):
    from .training.train_vqa import train_vqa
    print(json.dumps(train_vqa(_load(args), limit=args.limit, base_model=args.base_model), indent=2))


def cmd_train_baseline(args):
    from .training.train_baseline import build_baseline
    print(json.dumps(build_baseline(_load(args), limit=args.limit), indent=2, ensure_ascii=False))


def cmd_tune(args):
    from .training.tune import tune
    print(json.dumps(tune(_load(args), load_model=not args.fast), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    rep = evaluate(_load(args), limit=args.limit, load_model=not args.fast)
    print(json.dumps(rep.get("headline", rep), indent=2, ensure_ascii=False))


def cmd_ask(args):
    from .agent.vqa_agent import VqaAgent
    agent = VqaAgent(_load(args), load_model=not args.fast)
    out = agent.ask(args.question, image_path=args.image)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_ask_scene(args):
    from .agent.vqa_agent import VqaAgent
    from .data import samples
    agent = VqaAgent(_load(args), load_model=not args.fast)
    scene = samples.scenes()[args.scene % len(samples.scenes())]
    out = agent.ask(args.question, scene=scene)
    print(json.dumps({"scene_index": args.scene, **out}, indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.vqa_agent import VqaAgent
    from .data import samples
    agent = VqaAgent(_load(args), load_model=not args.fast)
    for ex in samples.seed_examples():
        job = agent.run(scene=ex["scene"], question=ex["question"], save=False)
        sd = job.to_dict()
        ok = "OK" if sd["answer"] == ex["gold"] else f"(gold={ex['gold']})"
        print(f"[{sd['qtype']:7s}] {ex['question']:40s} -> {sd['answer']:8s} {ok} "
              f"conf={sd['confidence']} decisions={[(x['id'], x['branch']) for x in sd['decisions']]}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["MMQA_INFER_CONFIG"] = str(args.config)
    target = "mmqa.api.app_combined:app" if args.ui else "mmqa.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args)), indent=2, ensure_ascii=False))


def cmd_per_type(args):
    from .analysis.per_type import per_type_report
    print(json.dumps(per_type_report(_load(args)), indent=2, ensure_ascii=False))


def cmd_monitor_log(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmqa", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="prefetch/sanity-check the datasets (streaming probes)")
    sp.add_argument("--render", action="store_true", help="also render synthetic eval scenes")
    sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("gen-synthetic", help="render synthetic scene eval images")
    sp.add_argument("--n-scenes", type=int, default=None); sp.add_argument("--split", default="eval")
    sp.set_defaults(func=cmd_gen_synthetic)
    sp = sub.add_parser("train-vqa", help="fine-tune the ViLT VQA core (HF Trainer)")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--base-model", default=None)
    sp.set_defaults(func=cmd_train_vqa)
    sp = sub.add_parser("train-baseline", help="persist the prior baseline (no GPU)")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_baseline)
    sp = sub.add_parser("tune", help="abstention-threshold sweep (coverage vs accuracy)")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="VQA accuracy vs baselines + per-type + agent coverage")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--fast", action="store_true", help="scene stub (no model download)")
    sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("ask", help="answer a question about an image file")
    sp.add_argument("--image", required=True); sp.add_argument("--question", required=True)
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_ask)
    sp = sub.add_parser("ask-scene", help="answer a question about a synthetic seed scene (offline)")
    sp.add_argument("--question", required=True); sp.add_argument("--scene", type=int, default=0)
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_ask_scene)
    sp = sub.add_parser("demo-agent", help="run the agent on the synthetic seed scenes")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server (+ --ui for the Gradio demo)")
    sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--ui", action="store_true"); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark of the agent")
    sp.add_argument("--n", type=int, default=10); sp.add_argument("--warmup", type=int, default=2)
    sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="per-question error analysis + abstention buckets")
    sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("per-type", help="per-question-type accuracy + abstention report")
    sp.set_defaults(func=cmd_per_type)
    sp = sub.add_parser("monitor-log", help="production monitoring report from the job log")
    sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor_log)
    sp = sub.add_parser("generate-report", help="generate the PDF report")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check")
    sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
