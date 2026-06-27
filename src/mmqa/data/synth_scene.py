"""Synthetic scene generator + a scene-answerer (the offline backbone for VQA).

VQA needs an image AND a model, both heavy. To run the agent, the evaluation and the
tests with **no torch and no real VQA model**, we synthesize a controllable world:
``make_scene`` builds a scene of colored shapes; ``render_scene`` draws it with PIL and
**embeds the scene spec in the PNG metadata** (read back by ``vision.image_utils``);
``scene_questions`` derives templated (question, gold answer, type) triples from the
scene; and ``scene_answer`` answers a question deterministically from the scene spec.

The ``SceneStubVQA`` model (``models/vqa_model.py``) calls ``scene_answer`` on the scene
embedded in an image, so the whole pipeline produces meaningful, honest numbers offline
and exercises the agent's abstention (questions it cannot parse get a low score). On
Colab the real ViLT/BLIP model reads real VQAv2 images. (Mirrors P15's SeedEngine.)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)

SHAPES = ["square", "circle", "triangle"]
PALETTE = {
    "red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 170, 70),
    "yellow": (235, 205, 50), "purple": (150, 60, 190), "orange": (240, 140, 30),
}


def make_scene(seed: int, max_shapes: int = 4, image_size: int = 384) -> Dict:
    import random
    rng = random.Random(seed)
    n = rng.randint(1, max_shapes)
    colors = list(PALETTE.keys())
    shapes: List[Dict] = []
    cells = image_size // 2
    for i in range(n):
        shp = rng.choice(SHAPES)
        col = rng.choice(colors)
        size = rng.randint(image_size // 8, image_size // 4)
        x = rng.randint(8, max(9, image_size - size - 8))
        y = rng.randint(8, max(9, image_size - size - 8))
        shapes.append({"shape": shp, "color": col, "bbox": [x, y, size, size]})
    return {"width": image_size, "height": image_size, "shapes": shapes}


def render_scene(scene: Dict, *, bg: str = "white", seed: int = 0):
    """Render a scene -> PIL.Image (RGB) with the scene spec embedded in PNG info."""
    from PIL import Image, ImageDraw
    w, h = scene["width"], scene["height"]
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for s in scene["shapes"]:
        x, y, sw, sh = s["bbox"]
        col = PALETTE.get(s["color"], (0, 0, 0))
        if s["shape"] == "square":
            draw.rectangle([x, y, x + sw, y + sh], fill=col)
        elif s["shape"] == "circle":
            draw.ellipse([x, y, x + sw, y + sh], fill=col)
        else:  # triangle
            draw.polygon([(x + sw // 2, y), (x, y + sh), (x + sw, y + sh)], fill=col)
    img.info["mmqa_scene"] = json.dumps(scene, ensure_ascii=False)
    return img


def save_png_with_scene(img, scene: Dict, path: str) -> str:
    from pathlib import Path

    from PIL.PngImagePlugin import PngInfo
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = PngInfo()
    meta.add_text("mmqa_scene", json.dumps(scene, ensure_ascii=False))
    img.save(str(p), pnginfo=meta)
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Question generation + the deterministic scene-answerer
# ─────────────────────────────────────────────────────────────────────────────
def _counts(scene: Dict):
    by_shape: Dict[str, int] = {}
    by_color: Dict[str, int] = {}
    by_pair: Dict[Tuple[str, str], int] = {}
    for s in scene["shapes"]:
        by_shape[s["shape"]] = by_shape.get(s["shape"], 0) + 1
        by_color[s["color"]] = by_color.get(s["color"], 0) + 1
        key = (s["color"], s["shape"])
        by_pair[key] = by_pair.get(key, 0) + 1
    return by_shape, by_color, by_pair


def scene_questions(scene: Dict, seed: int = 0) -> List[Dict]:
    import random
    rng = random.Random(seed)
    by_shape, by_color, by_pair = _counts(scene)
    qs: List[Dict] = []

    def add(question, answer, qtype, atype):
        # 10 identical annotators: an unambiguous synthetic scene = perfect human agreement,
        # so a correct answer scores VQA-accuracy 1.0 (and a wrong one 0.0).
        qs.append({"question": question, "answers": [answer] * 10, "gold": answer,
                   "qtype": qtype, "answer_type": atype})

    # total count
    add("how many shapes are there?", str(len(scene["shapes"])), "number", "number")
    # per-shape count (one random present shape)
    shp = rng.choice(list(by_shape))
    add(f"how many {shp}s are there?", str(by_shape[shp]), "number", "number")
    # color of a unique shape
    uniq = [s for s in scene["shapes"] if by_shape[s["shape"]] == 1]
    if uniq:
        s = uniq[0]
        add(f"what color is the {s['shape']}?", s["color"], "color", "other")
    # existence (present)
    s = rng.choice(scene["shapes"])
    add(f"is there a {s['shape']}?", "yes", "yes_no", "yes/no")
    # existence (absent)
    absent = [sh for sh in SHAPES if sh not in by_shape]
    if absent:
        add(f"is there a {absent[0]}?", "no", "yes_no", "yes/no")
    # color presence
    miss_color = [c for c in PALETTE if c not in by_color]
    if miss_color:
        add(f"is there anything {miss_color[0]}?", "no", "yes_no", "yes/no")
    return qs


def scene_answer(scene: Dict, question: str) -> Optional[str]:
    """Answer a question deterministically from the scene spec (None if unparseable)."""
    q = (question or "").strip().lower()
    by_shape, by_color, by_pair = _counts(scene)
    shapes_re = "|".join(SHAPES)
    colors_re = "|".join(PALETTE)

    m = re.search(rf"how many ({colors_re})\s+({shapes_re})s?", q)
    if m:
        return str(by_pair.get((m.group(1), m.group(2)), 0))
    m = re.search(rf"how many ({shapes_re})s?", q)
    if m:
        return str(by_shape.get(m.group(1), 0))
    if q.startswith(("how many", "how much")):
        return str(len(scene["shapes"]))
    m = re.search(rf"what colou?r is the ({shapes_re})", q)
    if m:
        for s in scene["shapes"]:
            if s["shape"] == m.group(1):
                return s["color"]
        return None
    m = re.search(rf"is there (?:a |an )?({colors_re})\s+({shapes_re})", q)
    if m:
        return "yes" if by_pair.get((m.group(1), m.group(2)), 0) > 0 else "no"
    m = re.search(rf"is there (?:a |an )?({shapes_re})", q)
    if m:
        return "yes" if by_shape.get(m.group(1), 0) > 0 else "no"
    m = re.search(rf"is there anything ({colors_re})", q)
    if m:
        return "yes" if by_color.get(m.group(1), 0) > 0 else "no"
    m = re.search(rf"\b({colors_re})\b", q)
    if "color" in q and m:
        return m.group(1)
    return None


def generate_dataset(out_dir: str, *, n_scenes: int = 120, max_shapes: int = 4,
                     image_size: int = 384, seed: int = 42) -> Dict:
    """Render ``n_scenes`` scene images + a manifest.jsonl (image + its QA)."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    written = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for i in range(n_scenes):
            scene = make_scene(seed + i, max_shapes=max_shapes, image_size=image_size)
            img = render_scene(scene, seed=seed + i)
            fname = f"scene_{i:04d}.png"
            save_png_with_scene(img, scene, str(out / fname))
            for q in scene_questions(scene, seed=seed + i):
                mf.write(json.dumps({"image": fname, "scene": scene, **q}, ensure_ascii=False) + "\n")
            written += 1
    logger.info("generated %d synthetic scenes -> %s", written, out)
    return {"scenes": written, "dir": str(out), "manifest": str(manifest)}


__all__ = ["SHAPES", "PALETTE", "make_scene", "render_scene", "save_png_with_scene",
           "scene_questions", "scene_answer", "generate_dataset"]
