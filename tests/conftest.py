"""Pytest fixtures — force everything offline + a temp artifacts dir."""

from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="mmqa-test-")
os.environ.setdefault("MMQA_ARTIFACTS_DIR", _TMP)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def cfg():
    from mmqa.config import AppConfig
    c = AppConfig()
    c.data.use_hf = False
    return c


@pytest.fixture
def agent(cfg):
    from mmqa.agent.vqa_agent import VqaAgent
    return VqaAgent(cfg, load_model=False)
