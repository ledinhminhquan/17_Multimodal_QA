"""VQA metric: answer normalization + the official soft accuracy + per-type."""

from __future__ import annotations

from mmqa.training import metrics as M


def test_normalize_answer():
    assert M.normalize_answer("The Cat.") == "cat"
    assert M.normalize_answer("a dog") == "dog"
    assert M.normalize_answer("two") == "2"
    assert M.normalize_answer("YES!") == "yes"


def test_vqa_accuracy_single_consensus():
    gts = ["yes"] * 10
    assert M.vqa_accuracy_single("yes", gts) == 1.0
    assert M.vqa_accuracy_single("no", gts) == 0.0


def test_vqa_accuracy_partial():
    # 3 of 10 said "2" -> min(1, matches/3) with leave-one-out
    gts = ["2", "2", "2"] + ["3"] * 7
    acc = M.vqa_accuracy_single("2", gts)
    assert 0.0 < acc < 1.0


def test_single_gold_exact():
    assert M.vqa_accuracy_single("blue", ["blue"]) == 1.0
    assert M.vqa_accuracy_single("red", ["blue"]) == 0.0


def test_vqa_accuracy_and_per_type():
    preds = ["yes", "2", "blue"]
    gts = [["yes"] * 10, ["2"] * 10, ["red"] * 10]
    types = ["yes/no", "number", "other"]
    assert M.vqa_accuracy(preds, gts) > 0.6
    pt = M.per_type_accuracy(preds, gts, types)
    assert pt["yes/no"]["accuracy"] == 1.0
    assert pt["other"]["accuracy"] == 0.0
