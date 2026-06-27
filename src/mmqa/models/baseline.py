"""Baselines for VQA - the honest floors the trained model must beat.

* ``MostCommonVQA`` - always predicts the global most-common answer ("yes"). The trivial floor.
* ``PriorVQA`` - a **blind / question-only** model: answers from the question TYPE alone
  (yes/no -> "yes", count -> "2", colour -> "white", else most-common). Measures the
  language-prior bias in VQA (how far you get WITHOUT looking at the image).

Both expose ``answer(image, question, top_k) -> AnswerResult`` so they slot into the same
agent and evaluation as the real model.
"""

from __future__ import annotations

from ..data.samples import MOST_COMMON_ANSWER
from . import question_type as QT
from .vqa_model import AnswerResult


class MostCommonVQA:
    name = "most_common"
    version = "prior-1.0"

    def answer(self, image, question: str, top_k: int = 5) -> AnswerResult:
        return AnswerResult([(MOST_COMMON_ANSWER, 1.0)], self.name, self.version)


class PriorVQA:
    name = "blind_prior"
    version = "prior-1.0"

    _DEFAULT = {"yes_no": "yes", "number": "2", "color": "white"}

    def answer(self, image, question: str, top_k: int = 5) -> AnswerResult:
        qtype = QT.classify_question(question)
        ans = self._DEFAULT.get(qtype, MOST_COMMON_ANSWER)
        return AnswerResult([(ans, 0.5), (MOST_COMMON_ANSWER, 0.3)], self.name, self.version)


def build_baseline(kind: str = "blind"):
    return MostCommonVQA() if kind == "most_common" else PriorVQA()


__all__ = ["MostCommonVQA", "PriorVQA", "build_baseline"]
