"""LangGraph 다단계 워크플로우.

States:
  query_decompose       — 사용자 질문을 keyword/하위질문으로 분해 (LLM)
  hybrid_retrieve       — BM25 + 임베딩 hybrid retriever 호출
  graph_expand          — GraphRAG 로 retrieve 결과 확장
  rerank_filter         — 점수 기반 dedup + top-k 컷오프
  cite_compose          — 인용 문자열 생성
  answer_generate       — LLM 으로 최종 답변 생성

이 노드 그래프는 langgraph.graph.StateGraph 로 구성한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


@dataclass
class RAGState:
    question: str
    sub_queries: list[str] = field(default_factory=list)
    hits: pd.DataFrame | None = None
    expanded: pd.DataFrame | None = None
    citations: list[dict] = field(default_factory=list)
    answer: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


SYSTEM_PROMPT = (
    "당신은 한국 해양수산부·해양경찰청 소관 법령에 정통한 어시스턴트입니다. "
    "반드시 제공된 조문(컨텍스트) 안에서만 근거를 찾고, 인용한 조문 번호와 법령명을 함께 표기하세요. "
    "컨텍스트에 없는 내용은 모른다고 답하세요."
)


def _format_context(hits: pd.DataFrame, top_k: int = 5) -> str:
    rows = hits.head(top_k).to_dict(orient="records")
    parts = []
    for i, r in enumerate(rows, 1):
        parts.append(f"[{i}] {r.get('law_name','')} 제{r.get('article_no','')}조 "
                     f"{r.get('article_title','')}\n{r.get('text','')[:500]}")
    return "\n\n".join(parts)


def build_app(retriever, graph, llm, *, top_k: int = 5,
              release_retriever_to_cpu: bool = False):
    """retriever (HybridRetriever), graph (nx.MultiDiGraph), llm (load_llm 결과) 주입.

    release_retriever_to_cpu=True 면 LLM 호출 직전에 sentence-transformers 모델을
    CPU 로 이동시켜 GPU/MPS 메모리 압박을 회피한다.
    """

    def node_decompose(state: RAGState) -> dict:
        # 가벼운 휴리스틱: 질문에서 명사/구를 추출 (실프로젝트는 LLM 분해)
        toks = [t for t in state.question.replace("?", " ").split() if len(t) >= 2]
        sub = list(dict.fromkeys([state.question] + toks[:3]))
        return {"sub_queries": sub, "debug": {**state.debug, "decompose": sub}}

    def node_retrieve(state: RAGState) -> dict:
        frames = []
        for q in state.sub_queries:
            frames.append(retriever.search(q, k=top_k))
        merged = pd.concat(frames, ignore_index=True).drop_duplicates("doc_id")
        merged = merged.sort_values("score", ascending=False).head(top_k * 2)
        return {"hits": merged}

    def node_graph_expand(state: RAGState) -> dict:
        if state.hits is None or state.hits.empty or graph is None:
            return {"expanded": state.hits}
        from ..graph.builder import expand_via_graph
        seed_ids = state.hits["doc_id"].tolist()
        ext_ids = expand_via_graph(graph, seed_ids, hops=2, max_articles=top_k * 2)
        ext = retriever.articles[retriever.articles["doc_id"].isin(ext_ids)].copy()
        ext["score"] = 0.0
        merged = pd.concat([state.hits, ext], ignore_index=True).drop_duplicates("doc_id")
        return {"expanded": merged.head(top_k * 2)}

    def node_rerank(state: RAGState) -> dict:
        df = (state.expanded if state.expanded is not None else state.hits)
        if df is None or df.empty:
            return {"hits": df}
        df = df.sort_values("score", ascending=False).head(top_k)
        return {"hits": df}

    def node_cite(state: RAGState) -> dict:
        cites = []
        if state.hits is not None:
            for _, r in state.hits.iterrows():
                cites.append({
                    "law_name": r.get("law_name", ""),
                    "article_no": r.get("article_no", ""),
                    "article_title": r.get("article_title", ""),
                    "score": float(r.get("score", 0.0)),
                })
        return {"citations": cites}

    def node_answer(state: RAGState) -> dict:
        if release_retriever_to_cpu:
            try:
                retriever.embedder.to("cpu")
            except Exception:  # noqa: BLE001
                pass
        ctx = _format_context(state.hits, top_k=top_k) if state.hits is not None else ""
        user_prompt = (
            f"질문: {state.question}\n\n"
            f"[컨텍스트]\n{ctx}\n\n"
            "위 컨텍스트만 사용해 한국어로 답변하세요. 답변 끝에 '인용:' 다음 인용한 항목 번호를 적으세요."
        )
        ans = llm.generate(SYSTEM_PROMPT, user_prompt)
        return {"answer": ans}

    sg = StateGraph(RAGState)
    sg.add_node("decompose", node_decompose)
    sg.add_node("retrieve", node_retrieve)
    sg.add_node("graph_expand", node_graph_expand)
    sg.add_node("rerank", node_rerank)
    sg.add_node("cite", node_cite)
    sg.add_node("answer", node_answer)

    sg.set_entry_point("decompose")
    sg.add_edge("decompose", "retrieve")
    sg.add_edge("retrieve", "graph_expand")
    sg.add_edge("graph_expand", "rerank")
    sg.add_edge("rerank", "cite")
    sg.add_edge("cite", "answer")
    sg.add_edge("answer", END)
    return sg.compile()


def ask(app, question: str) -> RAGState:
    out = app.invoke(RAGState(question=question))
    if isinstance(out, dict):
        return RAGState(**out)
    return out
