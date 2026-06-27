"""mmqa - Multimodal Question Answering (Visual Question Answering).

An end-to-end, production-grade VQA system: given an image and a natural-language
question about it, answer the question. A trainable VQA core (ViLT by default) is
orchestrated by a deterministic agent (D1-D5) that classifies the question type,
abstains when uncertain, and constrains the answer to the question type. Built for
the "NLP in Industry" final assignment (project #17).
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import AppConfig, load_config, save_config, ensure_dirs  # noqa: E402

__all__ = ["AppConfig", "load_config", "save_config", "ensure_dirs", "__version__"]
