"""Question-type classification + the per-type answer constraints (D2/D5)."""

from __future__ import annotations

from mmqa.models import question_type as QT


def test_classify():
    assert QT.classify_question("How many cats are there?") == "number"
    assert QT.classify_question("What color is the car?") == "color"
    assert QT.classify_question("Is there a dog?") == "yes_no"
    assert QT.classify_question("Where is the ball?") == "location"
    assert QT.classify_question("Who is in the photo?") == "person"
    assert QT.classify_question("What is on the table?") == "object"


def test_answer_matches_type():
    assert QT.answer_matches_type("yes", "yes_no")
    assert not QT.answer_matches_type("blue", "yes_no")
    assert QT.answer_matches_type("3", "number")
    assert QT.answer_matches_type("two", "number")
    assert not QT.answer_matches_type("dog", "number")
    assert QT.answer_matches_type("red", "color")
    assert QT.answer_matches_type("anything", "object")   # no constraint


def test_constrain_candidates_reranks():
    cands = ["blue", "yes", "no"]
    keep = QT.constrain_candidates(cands, "yes_no")
    assert keep[0] in ("yes", "no")
    # never empty even if nothing matches
    assert QT.constrain_candidates(["dog", "cat"], "yes_no") == ["dog", "cat"]


def test_type_default():
    assert QT.type_default("yes_no") == "yes"
    assert QT.type_default("number") == "1"
    assert QT.type_default("object") is None
