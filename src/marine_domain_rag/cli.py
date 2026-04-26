"""Marine Domain RAG CLI.

Subcommands:
  collect       국가법령정보 OPEN API → 원본 JSON 저장
  parse         원본 JSON → 조문 parquet
  index         조문 parquet → FAISS + BM25
  build-graph   조문 parquet → GraphRAG 그래프
  query         질의 → LangGraph 워크플로우 실행
  eval          QA 스위트 평가
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .collectors.law_client import LawAPIClient
from .config import Config
from .graph.builder import build_graph, load_graph, save_graph
from .indexing.embed_index import build_index, load_index
from .langgraph_app.workflow import build_app
from .llm.exaone_loader import load_llm
from .parsing.article_parser import parse_laws

logger = logging.getLogger("marine_rag")


def cmd_collect(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    client = LawAPIClient()
    out_dir = Path(cfg.get("paths", "raw_laws_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in client.iter_laws(
        cfg.get("law_api", "ministries"),
        max_per_ministry=int(cfg.get("law_api", "max_per_ministry", default=30)),
        fetch_body=True,
    ):
        mst = rec.get("법령일련번호") or "unknown"
        with open(out_dir / f"{mst}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        logger.info("saved %s — %s", mst, rec.get("법령명한글"))


def cmd_parse(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    raw_dir = Path(cfg.get("paths", "raw_laws_dir"))
    records: list[dict] = []
    for jf in raw_dir.glob("*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    df = parse_laws(records)
    out = Path(cfg.get("paths", "articles_parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    logger.info("parsed %d articles from %d laws -> %s", len(df), len(records), out)


def cmd_index(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    df = pd.read_parquet(cfg.get("paths", "articles_parquet"))
    build_index(
        df,
        model_name=cfg.get("embedding", "model_name"),
        batch_size=int(cfg.get("embedding", "batch_size")),
        max_seq_length=int(cfg.get("embedding", "max_seq_length")),
        out_dir=cfg.get("paths", "faiss_dir"),
        bm25_weight=float(cfg.get("retrieval", "bm25_weight")),
        embed_weight=float(cfg.get("retrieval", "embed_weight")),
    )
    logger.info("index built -> %s", cfg.get("paths", "faiss_dir"))


def cmd_build_graph(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    df = pd.read_parquet(cfg.get("paths", "articles_parquet"))
    g = build_graph(
        df,
        min_term_freq=int(cfg.get("graph", "min_term_frequency")),
        top_terms_per_article=int(cfg.get("graph", "top_terms_per_article")),
    )
    save_graph(g, cfg.get("paths", "graph_path"))
    logger.info("graph nodes=%d edges=%d -> %s",
                g.number_of_nodes(), g.number_of_edges(), cfg.get("paths", "graph_path"))


def _build_app_from_cfg(cfg: Config):
    retriever = load_index(cfg.get("paths", "faiss_dir"))
    graph_path = cfg.get("paths", "graph_path")
    g = load_graph(graph_path) if Path(graph_path).exists() else None
    llm = load_llm(
        provider=cfg.get("llm", "provider", default="mock"),
        model_name=cfg.get("llm", "model_name"),
        device=cfg.get("llm", "device", default="auto"),
        dtype=cfg.get("llm", "dtype", default="float16"),
        gguf_repo_id=cfg.get("llm", "gguf_repo_id"),
        gguf_filename=cfg.get("llm", "gguf_filename"),
    )
    return build_app(
        retriever, g, llm,
        top_k=int(cfg.get("retrieval", "top_k", default=5)),
        release_retriever_to_cpu=bool(cfg.get("llm", "release_retriever_to_cpu", default=False)),
    )


def cmd_query(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    app = _build_app_from_cfg(cfg)
    out = app.invoke({"question": args.q})
    cites = out.get("citations") if isinstance(out, dict) else getattr(out, "citations", [])
    answer = out.get("answer") if isinstance(out, dict) else getattr(out, "answer", "")
    print("\n=== ANSWER ===\n" + (answer or ""))
    print("\n=== CITATIONS ===")
    for c in (cites or []):
        print(f"- {c.get('law_name')} 제{c.get('article_no')}조 {c.get('article_title')} "
              f"(score={c.get('score', 0):.3f})")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    from .evaluation.qa_eval import evaluate, load_suite
    suite = load_suite(cfg.get("paths", "eval_suite"))
    app = _build_app_from_cfg(cfg)
    res = evaluate(app, suite, k=int(cfg.get("retrieval", "top_k", default=5)))
    print(f"n={res.n} hit@k={res.hit_at_k:.3f} "
          f"cite_match={res.cite_match_rate:.3f} nonempty={res.nonempty_rate:.3f}")


def _check_env_warning() -> None:
    """llama-cpp + sentence-transformers 동시 사용 시 OMP/BLAS 충돌 회피 환경변수 확인."""
    import os
    missing = []
    if os.environ.get("KMP_DUPLICATE_LIB_OK") != "TRUE":
        missing.append("KMP_DUPLICATE_LIB_OK=TRUE")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        missing.append("OMP_NUM_THREADS=1")
    if missing:
        print("⚠️  EXAONE GGUF + sentence-transformers 동시 사용 시 SIGSEGV 가 발생할 수 있습니다. "
              "다음 환경변수를 prefix 로 붙여 다시 실행하세요:\n"
              f"   {' '.join(missing)} python -m marine_domain_rag.cli ...\n",
              flush=True)


def main() -> None:
    _check_env_warning()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="marine-rag")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect").set_defaults(func=cmd_collect)
    sub.add_parser("parse").set_defaults(func=cmd_parse)
    sub.add_parser("index").set_defaults(func=cmd_index)
    sub.add_parser("build-graph").set_defaults(func=cmd_build_graph)

    p_q = sub.add_parser("query")
    p_q.add_argument("--q", required=True)
    p_q.set_defaults(func=cmd_query)

    sub.add_parser("eval").set_defaults(func=cmd_eval)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
