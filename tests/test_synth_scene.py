"""Synthetic scene generator + the deterministic scene-answerer + the SceneStubVQA."""

from __future__ import annotations

from mmqa.config import ModelConfig
from mmqa.data import samples
from mmqa.data import synth_scene as S
from mmqa.models.vqa_model import SceneStubVQA
from mmqa.vision.image_utils import SceneImage, read_scene


def test_make_scene_deterministic():
    a = S.make_scene(7)
    b = S.make_scene(7)
    assert a == b
    assert 1 <= len(a["shapes"]) <= 4


def test_scene_answer_counts_and_colors():
    scene = samples.scenes()[0]   # red square, blue circle, green square
    assert S.scene_answer(scene, "how many shapes are there?") == "3"
    assert S.scene_answer(scene, "how many squares are there?") == "2"
    assert S.scene_answer(scene, "what color is the circle?") == "blue"
    assert S.scene_answer(scene, "is there a triangle?") == "no"
    assert S.scene_answer(scene, "is there a red square?") == "yes"


def test_scene_questions_have_10_annotators():
    qs = S.scene_questions(samples.scenes()[0])
    assert qs and all(len(q["answers"]) == 10 for q in qs)


def test_scene_image_carrier_no_pil():
    scene = samples.scenes()[1]
    img = SceneImage(scene)
    assert read_scene(img) == scene
    assert img.size[0] > 1


def test_stub_answers_from_scene():
    stub = SceneStubVQA(ModelConfig())
    img = SceneImage(samples.scenes()[0])
    res = stub.answer(img, "how many shapes are there?")
    assert res.top1 == "3"
    assert res.confidence > 0.5
