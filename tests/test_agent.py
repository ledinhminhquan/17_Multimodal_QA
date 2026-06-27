"""Agent: the 5-decision FSM runs offline; abstention + type-constraint behave."""

from __future__ import annotations

from dataclasses import replace

from mmqa.data import samples


def test_agent_five_decisions(agent):
    ex = samples.seed_examples()[0]
    job = agent.run(scene=ex["scene"], question=ex["question"], save=False)
    ids = [d.id for d in job.decisions]
    assert ids == ["D1", "D2", "D3", "D4", "D5"]
    assert job.status.value in ("completed", "needs_review", "abstained")


def test_agent_answers_seed_correctly(agent):
    n_correct = 0
    examples = samples.seed_examples()
    for ex in examples:
        out = agent.ask(ex["question"], scene=ex["scene"])
        if out["answer"] == ex["gold"]:
            n_correct += 1
    assert n_correct >= len(examples) - 1   # the scene stub answers (almost) all correctly


def test_no_image_fails(agent):
    job = agent.run(question="how many?", save=False)
    assert job.status.value == "failed"
    assert any(d.id == "D1" and d.branch == "no_image" for d in job.decisions)


def test_abstention_on_high_threshold(cfg):
    from mmqa.agent.vqa_agent import VqaAgent
    # force abstention: require confidence > what the stub's type-default (0.30) gives on an
    # unparseable question
    acfg = replace(cfg.agent, confidence_min=0.9)
    tcfg = replace(cfg, agent=acfg)
    agent = VqaAgent(tcfg, load_model=False)
    out = agent.ask("what is the meaning of this abstract scene?", scene=samples.scenes()[0])
    assert out["abstained"] is True
    assert out["answer"] == "unsure"


def test_type_constraint_yes_no(cfg):
    # a yes/no question must get a yes/no answer
    out = __import__("mmqa.agent.vqa_agent", fromlist=["VqaAgent"]).VqaAgent(
        cfg, load_model=False).ask("is there a circle?", scene=samples.scenes()[0])
    assert out["answer"] in ("yes", "no")


def test_deterministic(agent):
    ex = samples.seed_examples()[2]
    a = agent.run(scene=ex["scene"], question=ex["question"], save=False)
    b = agent.run(scene=ex["scene"], question=ex["question"], save=False)
    assert a.answer == b.answer
    assert [d.branch for d in a.decisions] == [d.branch for d in b.decisions]
