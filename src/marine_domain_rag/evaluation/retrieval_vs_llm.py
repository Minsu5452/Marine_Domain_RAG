"""Retrieval-only vs LLM-augmented 분리 평가.

기존 ``qa_eval.evaluate`` 는 hit@k(retrieval) 와 cite_match(LLM 인용 일치) 를
같은 루프 안에서 합쳐 측정합니다. RAG 컴포넌트별 기여도를 분리하기 위해
이 모듈은 다음 두 단계로 나눠 계산합니다.

  1) retrieval-only       : retriever.search() 결과만으로 hit@k 측정
  2) LLM-augmented        : LangGraph workflow 의 최종 citations / answer 평가

또한 다음 3가지 오류를 분류해 카운트합니다.

  retrieved_but_not_cited : retrieval 은 정답 조문을 top-k 안에 가져왔지만
                            LLM 의 최종 인용 리스트에는 정답이 없는 경우
  not_retrieved           : retrieval 부터 정답 조문을 못 가져온 경우
  llm_hallucination       : retrieval 은 정답을 가져왔고 LLM 인용에도 정답이
                            있으나, answer 본문에 retrieval 컨텍스트와 무관한
                            법령명이 등장하는 경우(휴리스틱)
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RetrievalLLMReport:
    n: int
    # retrieval-only
    retrieval_hit_at_k: float
    # LLM-augmented
    cite_match_rate: float
    nonempty_rate: float
    # 오류 분류 (retrieval_hit 와 cite_match 의 조합으로 결정)
    retrieved_but_not_cited: int
    not_retrieved: int
    llm_hallucination: int
    # 1행 = 1샘플
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    avg_latency_sec: float = 0.0

    def as_table(self) -> str:
        rows = [
            ("samples", self.n),
            ("retrieval_hit@k", f"{self.retrieval_hit_at_k:.3f}"),
            ("cite_match_rate", f"{self.cite_match_rate:.3f}"),
            ("nonempty_rate", f"{self.nonempty_rate:.3f}"),
            ("retrieved_but_not_cited", self.retrieved_but_not_cited),
            ("not_retrieved", self.not_retrieved),
            ("llm_hallucination", self.llm_hallucination),
            ("avg_latency_sec", f"{self.avg_latency_sec:.2f}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}} : {v}" for k, v in rows)


def load_suite(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def _coerce_state(out) -> tuple[list[dict], str]:
    if isinstance(out, dict):
        return list(out.get("citations") or []), str(out.get("answer") or "")
    return list(getattr(out, "citations", []) or []), str(getattr(out, "answer", "") or "")


def _hit_in_citations(item: dict, cit_list: list[dict], k: int) -> bool:
    """item.gold_law / gold_article (+ gold_citations alternatives) 기준으로
    cit_list 의 top-k 안에 정답 조문이 들어 있는지."""
    gold_law = item.get("gold_law")
    gold_article = item.get("gold_article")
    gold_citations = item.get("gold_citations") or []
    alts = item.get("acceptable_alternatives") or []

    targets: list[tuple[str, str | None]] = []
    if gold_law:
        targets.append((gold_law, str(gold_article) if gold_article else None))
    for c in [*gold_citations, *alts]:
        ln = c.get("law_name") or ""
        an = c.get("article_no")
        targets.append((ln, str(an) if an else None))

    for cit in cit_list[:k]:
        law = (cit.get("law_name") or "")
        art = str(cit.get("article_no") or "")
        for tl, ta in targets:
            if tl and tl in law and (not ta or ta == art):
                return True
    return False


def _hit_in_search_df(item: dict, df: pd.DataFrame, k: int) -> bool:
    """retriever.search() 결과 dataframe 기준 hit@k."""
    if df is None or df.empty:
        return False
    head = df.head(k)
    gold_law = item.get("gold_law")
    gold_article = item.get("gold_article")
    gold_citations = item.get("gold_citations") or []
    alts = item.get("acceptable_alternatives") or []
    targets: list[tuple[str, str | None]] = []
    if gold_law:
        targets.append((gold_law, str(gold_article) if gold_article else None))
    for c in [*gold_citations, *alts]:
        ln = c.get("law_name") or ""
        an = c.get("article_no")
        targets.append((ln, str(an) if an else None))
    for _, r in head.iterrows():
        law = str(r.get("law_name", "") or "")
        art = str(r.get("article_no", "") or "")
        for tl, ta in targets:
            if tl and tl in law and (not ta or ta == art):
                return True
    return False


def _looks_hallucinated(answer: str, cit_list: list[dict]) -> bool:
    """가벼운 휴리스틱: 답변 본문에 컨텍스트(citations) 의 어떤 법령명도 들어가
    있지 않으면 LLM 이 외부 지식만 사용한 것으로 간주."""
    if not answer.strip() or not cit_list:
        return False
    law_names = [(c.get("law_name") or "").strip() for c in cit_list]
    law_names = [ln for ln in law_names if ln]
    if not law_names:
        return False
    return not any(ln in answer for ln in law_names)


def evaluate_split(retriever, app, suite: list[dict], k: int = 5) -> RetrievalLLMReport:
    """retriever (HybridRetriever) 와 langgraph app 을 둘 다 받아 분리 측정."""
    n = len(suite)
    retr_hits = 0
    cite_hits = 0
    nonempty = 0
    retrieved_but_not_cited = 0
    not_retrieved = 0
    halluc = 0
    per_sample = []
    total_latency = 0.0

    for item in suite:
        q = str(item["question"])
        # 1) retrieval-only
        try:
            df = retriever.search(q, k=k)
        except Exception as e:  # noqa: BLE001
            logger.warning("retrieve fail (%s): %s", q, e)
            df = pd.DataFrame()
        retrieval_hit = _hit_in_search_df(item, df, k=k)

        # 2) LLM-augmented (LangGraph workflow)
        cit_list: list[dict] = []
        answer = ""
        latency = 0.0
        try:
            t0 = time.time()
            out = app.invoke({"question": q})
            latency = time.time() - t0
            cit_list, answer = _coerce_state(out)
        except Exception as e:  # noqa: BLE001
            logger.warning("invoke fail (%s): %s", q, e)
        cite_hit = _hit_in_citations(item, cit_list, k=k)
        is_nonempty = bool(answer.strip())
        is_halluc = _looks_hallucinated(answer, cit_list)

        if retrieval_hit:
            retr_hits += 1
        if cite_hit:
            cite_hits += 1
        if is_nonempty:
            nonempty += 1
        # 오류 분류
        if not retrieval_hit:
            not_retrieved += 1
        elif retrieval_hit and not cite_hit:
            retrieved_but_not_cited += 1
        if is_halluc:
            halluc += 1
        total_latency += latency

        per_sample.append({
            "question": q,
            "retrieval_hit": retrieval_hit,
            "cite_hit": cite_hit,
            "nonempty": is_nonempty,
            "hallucination": is_halluc,
            "latency_sec": round(latency, 3),
            "top_citations": [
                f"{c.get('law_name','')} 제{c.get('article_no','')}조" for c in cit_list[:3]
            ],
        })
        gc.collect()

    denom = max(n, 1)
    return RetrievalLLMReport(
        n=n,
        retrieval_hit_at_k=retr_hits / denom,
        cite_match_rate=cite_hits / denom,
        nonempty_rate=nonempty / denom,
        retrieved_but_not_cited=retrieved_but_not_cited,
        not_retrieved=not_retrieved,
        llm_hallucination=halluc,
        per_sample=per_sample,
        avg_latency_sec=total_latency / denom,
    )
