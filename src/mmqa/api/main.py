"""FastAPI service for the Multimodal Question Answering (VQA) system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /ask-scene``  - {question, scene} -> answer (the synthetic path; no image upload, always on)
* ``POST /ask``        - upload an image + a question -> answer (registered only when python-multipart present)

Low-confidence answers are returned as "unsure" and flagged (the calibrated-abstention gate).
"""

from __future__ import annotations

import io
from importlib.util import find_spec

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import AskResponse, AskSceneRequest, HealthResponse

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)

_HAS_MULTIPART = find_spec("multipart") is not None or find_spec("python_multipart") is not None


def _resp(out: dict) -> AskResponse:
    return AskResponse(question=out.get("question", ""), answer=out["answer"], qtype=out["qtype"],
                       confidence=out["confidence"], abstained=out["abstained"],
                       type_constrained=out["type_constrained"], needs_review=out["needs_review"],
                       candidates=out.get("candidates", []), model_version=out["model_version"],
                       status=out["status"])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", model=getattr(agent.model, "name", "?"), version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "model": getattr(agent.model, "version", "?"),
            "model_version": cfg.serving.model_version}


@app.post("/ask-scene", response_model=AskResponse)
def ask_scene(req: AskSceneRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="provide a question")
    out = get_agent().ask(req.question, scene=req.scene)
    out["question"] = req.question
    return _resp(out)


if _HAS_MULTIPART:
    from fastapi import File, Form, UploadFile

    @app.post("/ask", response_model=AskResponse)
    def ask(file: "UploadFile" = File(...), question: str = Form(...)) -> AskResponse:
        if not question.strip():
            raise HTTPException(status_code=422, detail="provide a question")
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        except Exception:
            raise HTTPException(status_code=422, detail="could not read image")
        out = get_agent().ask(question, image=img)
        out["question"] = question
        return _resp(out)
else:  # pragma: no cover
    logger.info("python-multipart not installed; /ask disabled (/ask-scene still works).")


__all__ = ["app"]
