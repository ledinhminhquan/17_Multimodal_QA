"""End-to-end (offline): evaluate() + error analysis + per-type + grading + baselines."""

from __future__ import annotations


def test_evaluate_offline(cfg):
    from mmqa.training.evaluate import evaluate
    rep = evaluate(cfg, save=False, load_model=False)
    systems = rep["systems"]
    assert "most_common" in systems and "blind_prior" in systems
    model_name = [k for k in systems if k not in ("most_common", "blind_prior")][0]
    # the scene stub (image-aware) must beat the blind language prior
    assert systems[model_name]["accuracy"] > systems["blind_prior"]["accuracy"]
    assert rep["agent"]["coverage"] is not None


def test_error_and_per_type(cfg):
    from mmqa.analysis.error_analysis import error_analysis
    from mmqa.analysis.per_type import per_type_report
    ea = error_analysis(cfg, save=False)
    assert ea["n"] >= 1
    pt = per_type_report(cfg, save=False)
    assert pt["per_type"]


def test_grading_runs(cfg):
    from pathlib import Path

    from mmqa.grading.checklist import build_checklist
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["total"] > 0
    assert res["summary"]["FAIL"] == 0


def test_baselines():
    from mmqa.models.baseline import MostCommonVQA, PriorVQA
    assert MostCommonVQA().answer(None, "anything?").top1 == "yes"
    assert PriorVQA().answer(None, "how many cats?").top1 == "2"
