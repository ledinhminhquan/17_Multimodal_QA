"""Config: defaults, YAML round-trip, unknown-key tolerance, env paths."""

from __future__ import annotations


def test_defaults(cfg):
    assert cfg.model.base_model == "dandelin/vilt-b32-finetuned-vqa"
    assert cfg.model.model_type == "vilt"
    assert 0.0 <= cfg.agent.confidence_min <= 1.0
    assert cfg.data.vqa_eval_dataset == "lmms-lab/VQAv2"


def test_yaml_roundtrip(tmp_path, cfg):
    from mmqa.config import load_config, save_config
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    again = load_config(p)
    assert again.model.base_model == cfg.model.base_model
    assert again.agent.confidence_min == cfg.agent.confidence_min


def test_unknown_keys_ignored(tmp_path):
    from mmqa.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  base_model: x\n  bogus: 1\nzzz: 2\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.model.base_model == "x"


def test_env_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("MMQA_ARTIFACTS_DIR", str(tmp_path / "arts"))
    from mmqa import config
    assert str(tmp_path / "arts") in str(config.artifacts_dir())
    dirs = config.ensure_dirs()
    assert dirs["models"].exists()
