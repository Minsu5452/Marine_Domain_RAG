"""Query decomposition 전략.

LangGraph 워크플로우의 ``node_decompose`` 가 사용하는 여러 전략을 한 곳에 모아
ablation 평가를 쉽게 했습니다.

전략:
  raw      : 분해 없음. 원 질문 한 줄만 하위 쿼리로 사용
  noun     : 공백 분리 + 길이 ≥ 2 토큰을 명사 후보로 추출 (현재 baseline)
  llm      : LLM 에 분해를 요청. JSON 파싱 실패 / 환경에서 LLM 미가용 시
             noun 으로 자동 fallback (debug 메타에 fallback 사유 기록)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

Strategy = Callable[[str], list[str]]


def decompose_raw(question: str) -> list[str]:
    """분해 없음. 검색은 원 질문 한 번만 수행."""
    q = question.strip()
    return [q] if q else []


def decompose_noun(question: str) -> list[str]:
    """현재 baseline. 공백 분리 후 길이 ≥ 2 토큰 상위 3개를 추가."""
    toks = [t for t in question.replace("?", " ").split() if len(t) >= 2]
    return list(dict.fromkeys([question] + toks[:3]))


_DECOMPOSE_SYSTEM = (
    "당신은 한국 법령 검색 보조자입니다. 입력 질문을 검색 효율을 높이기 위한 "
    "2~4 개의 짧은 한국어 sub-query 로 분해하세요. 응답은 반드시 JSON 배열 한 줄로만 "
    "출력하세요. 예: [\"임용권자\", \"해양경찰청 임용\"]"
)


def decompose_llm(question: str, llm) -> tuple[list[str], str | None]:
    """LLM 으로 분해. (sub_queries, fallback_reason) 반환.

    ``fallback_reason`` 이 None 이면 LLM 분해 성공, 아니면 noun 결과를 대신 반환.
    """
    if llm is None or getattr(llm, "name", "") in ("mock", "skip-llm"):
        return decompose_noun(question), "llm_unavailable_or_mock"
    try:
        raw = llm.generate(_DECOMPOSE_SYSTEM, question, max_new_tokens=128)
    except Exception as e:  # noqa: BLE001
        logger.warning("decompose_llm generate fail: %s", e)
        return decompose_noun(question), f"generate_error:{type(e).__name__}"

    # JSON 배열 추출
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not m:
        return decompose_noun(question), "no_json_array"
    try:
        arr = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return decompose_noun(question), "json_parse_error"
    if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
        return decompose_noun(question), "non_string_array"
    cleaned = [s.strip() for s in arr if s.strip()]
    if not cleaned:
        return decompose_noun(question), "empty_array"
    return list(dict.fromkeys([question] + cleaned[:4])), None


def get_strategy(name: str, llm=None) -> Strategy:
    """평가 스크립트가 호출하기 좋은 closure 형태."""
    name = (name or "noun").lower()
    if name == "raw":
        return decompose_raw
    if name == "noun":
        return decompose_noun
    if name == "llm":
        def _strategy(q: str) -> list[str]:
            sub, _reason = decompose_llm(q, llm)
            return sub
        return _strategy
    logger.warning("unknown decompose strategy %s — fallback noun", name)
    return decompose_noun
