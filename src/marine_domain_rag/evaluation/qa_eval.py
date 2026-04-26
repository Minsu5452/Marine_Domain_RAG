"""소규모 QA 평가 — hit@k, 인용 정확도, 답변 비공백 비율."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    n: int
    hit_at_k: float
    cite_match_rate: float
    nonempty_rate: float


def load_suite(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def _coerce_state(out) -> tuple[list[dict], str]:
    """LangGraph 의 invoke 반환은 dict-like state. citations / answer 만 추출."""
    if isinstance(out, dict):
        return list(out.get("citations") or []), str(out.get("answer") or "")
    return list(getattr(out, "citations", []) or []), str(getattr(out, "answer", "") or "")


def evaluate(app, suite: list[dict], k: int = 5) -> EvalResult:
    hits = 0
    cites = 0
    nonempty = 0
    for item in suite:
        q = str(item["question"])
        gold_law = item.get("gold_law")
        gold_article = item.get("gold_article")
        try:
            out = app.invoke({"question": q})
        except Exception as e:  # noqa: BLE001
            logger.warning("invoke fail (%s): %s", q, e)
            continue
        cit_list, answer = _coerce_state(out)

        is_hit = False
        for cit in cit_list[:k]:
            law = (cit.get("law_name") or "")
            art = str(cit.get("article_no") or "")
            if gold_law and gold_law in law:
                if not gold_article or str(gold_article) in art:
                    is_hit = True
                    break
        if is_hit:
            hits += 1
        if answer.strip() and any(gold_law and gold_law in (cit.get("law_name") or "")
                                  for cit in cit_list):
            cites += 1
        if answer.strip():
            nonempty += 1
        del out, cit_list, answer
        gc.collect()
    n = max(len(suite), 1)
    return EvalResult(n=len(suite), hit_at_k=hits / n,
                      cite_match_rate=cites / n, nonempty_rate=nonempty / n)
