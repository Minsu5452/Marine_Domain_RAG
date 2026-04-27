"""Retrieval-only vs LLM-augmented 분리 평가 실행 스크립트.

사용법:
    python scripts/eval_retrieval_vs_llm.py \
        --config configs/default.yaml \
        --report reports/retrieval_vs_llm.json

옵션:
    --top-k          retriever 와 평가의 top-k (기본: configs.retrieval.top_k)
    --skip-llm       LLM 로드 없이 retrieval-only 만 측정
                     (cite_match / hallucination 카운트는 0 으로 남김)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from marine_domain_rag.config import Config
from marine_domain_rag.evaluation.retrieval_vs_llm import (
    RetrievalLLMReport,
    evaluate_split,
    load_suite,
)
from marine_domain_rag.graph.builder import load_graph
from marine_domain_rag.indexing.embed_index import load_index
from marine_domain_rag.langgraph_app.workflow import build_app
from marine_domain_rag.llm.exaone_loader import MockLLM, load_llm

logger = logging.getLogger("eval_retrieval_vs_llm")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--report", default="reports/retrieval_vs_llm.json")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--skip-llm", action="store_true",
                   help="LLM 로드 생략 (retrieval-only 만 측정)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)

    retriever = load_index(cfg.get("paths", "faiss_dir"))
    graph_path = cfg.get("paths", "graph_path")
    g = load_graph(graph_path) if Path(graph_path).exists() else None

    if args.skip_llm:
        llm = MockLLM("skip-llm")
    else:
        llm = load_llm(
            provider=cfg.get("llm", "provider", default="mock"),
            model_name=cfg.get("llm", "model_name"),
            device=cfg.get("llm", "device", default="auto"),
            dtype=cfg.get("llm", "dtype", default="float16"),
            gguf_repo_id=cfg.get("llm", "gguf_repo_id"),
            gguf_filename=cfg.get("llm", "gguf_filename"),
        )

    top_k = args.top_k or int(cfg.get("retrieval", "top_k", default=5))
    app = build_app(
        retriever, g, llm, top_k=top_k,
        release_retriever_to_cpu=bool(
            cfg.get("llm", "release_retriever_to_cpu", default=False)
        ),
    )

    suite = load_suite(cfg.get("paths", "eval_suite"))
    report: RetrievalLLMReport = evaluate_split(retriever, app, suite, k=top_k)

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n": report.n,
            "top_k": top_k,
            "llm_provider": (cfg.get("llm", "provider", default="mock")
                             if not args.skip_llm else "skip"),
            "llm_name": getattr(llm, "name", "unknown"),
            "retrieval_hit_at_k": report.retrieval_hit_at_k,
            "cite_match_rate": report.cite_match_rate,
            "nonempty_rate": report.nonempty_rate,
            "retrieved_but_not_cited": report.retrieved_but_not_cited,
            "not_retrieved": report.not_retrieved,
            "llm_hallucination": report.llm_hallucination,
            "avg_latency_sec": report.avg_latency_sec,
            "per_sample": report.per_sample,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Retrieval vs LLM split report ===")
    print(report.as_table())
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
