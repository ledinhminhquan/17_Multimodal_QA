"""Image utilities: load an image, read an embedded synthetic-scene spec, light checks.

PIL is imported lazily so the package imports without it; ``read_scene`` works on both
in-memory PIL images (``image.info``) and PNGs reloaded from disk (PNG text chunk).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)


class SceneImage:
    """A no-PIL image stand-in that carries a synthetic-scene spec.

    Lets the agent / tests run the full pipeline from a scene dict with no Pillow:
    ``read_scene`` reads ``.info``, ``is_valid_image`` reads ``.size``, and the
    SceneStubVQA answers from the embedded scene. A real run uses a real PIL image.
    """

    def __init__(self, scene: dict):
        self.scene = scene
        self.size = (int(scene.get("width", 1)), int(scene.get("height", 1)))
        self.mode = "RGB"
        self.info = {"mmqa_scene": json.dumps(scene, ensure_ascii=False)}

    def convert(self, mode: str = "RGB"):
        return self


def load_image(src: Any):
    """Accept a path, a PIL image, or raw bytes -> an RGB PIL image."""
    from PIL import Image
    if hasattr(src, "convert"):
        return src.convert("RGB") if src.mode != "RGB" else src
    if isinstance(src, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(src)).convert("RGB")
    return Image.open(src).convert("RGB")


def read_scene(image) -> Optional[dict]:
    """Pull the synthetic-scene spec embedded by the generator, if present."""
    info = getattr(image, "info", None) or {}
    raw = info.get("mmqa_scene")
    if raw is None and isinstance(getattr(image, "text", None), dict):
        raw = image.text.get("mmqa_scene")
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None


def has_scene(image) -> bool:
    return read_scene(image) is not None


def image_quality(image) -> float:
    """A cheap 0..1 quality proxy (variance of grayscale); blank images score ~0."""
    try:
        import numpy as np
        g = np.asarray(image.convert("L"), dtype="float32")
        return float(min(1.0, g.std() / 64.0))
    except Exception:
        return 1.0


def is_valid_image(image) -> bool:
    return hasattr(image, "size") and image.size[0] > 1 and image.size[1] > 1


__all__ = ["load_image", "read_scene", "has_scene", "image_quality", "is_valid_image"]
