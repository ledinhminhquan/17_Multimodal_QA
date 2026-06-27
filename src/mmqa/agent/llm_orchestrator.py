"""Optional LLM brain (anthropic), with rule fallback.

Advisory only: may write a one-line note about a low-confidence VQA answer (e.g. "the
model is unsure; a human should verify"). Disabled by default; validates its own output
and on any problem the caller keeps the rule result. Default deployment makes zero paid
API calls and is fully deterministic. **Never overrides the model's answer or the gates.**
"""

from __future__ import annotations

import os
from typing import Optional

from ..config import AgentConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        return bool(self.cfg.llm_fallback_enabled and os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic
            key = os.environ.get(self.cfg.llm_api_key_env)
            self._client = anthropic.Anthropic(api_key=key) if key else None
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def review_note(self, question: str, answer: str, confidence: Optional[float]) -> Optional[str]:
        """A one-line reviewer note for a low-confidence answer. None keeps the rule rationale."""
        if not self.available() or not question:
            return None
        client = self._get_client()
        if client is None:
            return None
        prompt = (f"A visual-QA model answered the question '{question}' with '{answer}' "
                  f"(confidence {confidence}). In ONE short sentence, note whether this looks "
                  f"plausible or should be double-checked by a human. Do NOT change the answer.")
        try:
            msg = client.messages.create(model=self.cfg.llm_model, max_tokens=80, temperature=0.0,
                                         messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") for b in msg.content).strip()
            return text or None
        except Exception as exc:
            logger.info("LLM review_note failed (%s)", exc)
            return None


__all__ = ["LLMBrain"]
