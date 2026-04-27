"""Query decomposition 전략 ablation.

세 전략을 동일 retriever / 동일 QA 스위트로 비교해 hit@k 변화를 측정합니다.

  raw   : 분해 없음 (원 질문 한 번)
  noun  : 명사 추출 (현재 baseline)
  llm   : LLM 분해 — 환경에서 LLM 미가용이면 mock fallback (리포트에 표시)

사용법:
    python scripts/eval_decompose_ablation.py \
        --config configs/default.yaml \
        --report reports/decompose_ablation.json

옵션 --skip-llm 을 주면 LLM 로드를 건너뛰고 llm 전략은 자동 fallback 됩니다
(여전히 noun 으로 동작하지만 그 사유를 리포트에 명시합니다).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from marine_domain_rag.config import Config
from marine_domain_rag.evaluation.retrieval_vs_llm import _hit_in_search_df, load_suite
from marine_domain_rag.indexing.embed_index import load_index
from marine_domain_rag.langgraph_app.decompose import (
    decompose_llm,
    decompose_noun,
    decompose_raw,
)
from marine_domain_rag.llm.exaone_loader import MockLLM, load_llm

logger = logging.getLogger("eval_decompose_ablation")


def _run_strategy(name: str, retriever, suite, k: int, llm=None) -> dict:
    """retriever 단계만 사용하는 빠른 ablation. sub_queries 의 hits 를
    score 기준으로 합쳐 head(k) 와 정답을 비교."""
    hits_at_k = 0
    fallback_reasons: list[str] = []
    per_sample = []
    for item in suite:
        q = str(item["question"])
        if name == "raw":
            sub = decompose_raw(q)
            reason = None
        elif name == "noun":
            sub = decompose_noun(q)
            reason = None
        elif name == "llm":
            sub, reason = decompose_llm(q, llm)
            if reason:
                fallback_reasons.append(reason)
        else:
            raise ValueError(f"unknown strategy {name}")

        frames = []
        for sq in sub:
            frames.append(retriever.search(sq, k=k))
        merged = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("doc_id")
            .sort_values("score", ascending=False)
            .head(k)
        )
        is_hit = _hit_in_search_df(item, merged, k=k)
        if is_hit:
            hits_at_k += 1
        per_sample.append({
            "question": q,
            "sub_queries": sub,
            "fallback_reason": reason,
            "retrieval_hit": is_hit,
        })
    n = max(len(suite), 1)
    return {
        "strategy": name,
        "hit_at_k": hits_at_k / n,
        "n": len(suite),
        "fallback_reasons": fallback_reasons,
        "per_sample": per_sample,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--report", default="reports/decompose_ablation.json")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--skip-llm", action="store_true",
                   help="LLM 로드 생략. llm 전략은 mock fallback 으로 표시됨")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)

    retriever = load_index(cfg.get("paths", "faiss_dir"))
    top_k = args.top_k or int(cfg.get("retrieval", "top_k", default=5))
    suite = load_suite(cfg.get("paths", "eval_suite"))

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

    results = []
    for strat in ("raw", "noun", "llm"):
        results.append(_run_strategy(strat, retriever, suite, top_k, llm=llm))

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "top_k": top_k,
            "llm_name": getattr(llm, "name", "unknown"),
            "skip_llm": args.skip_llm,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Decompose strategy ablation ===")
    print(f"{'strategy':<8} {'n':>3} {'hit@k':>7}  notes")
    for r in results:
        note = ""
        if r["fallback_reasons"]:
            uniq = sorted(set(r["fallback_reasons"]))
            note = "fallback: " + ", ".join(uniq)
        print(f"{r['strategy']:<8} {r['n']:>3} {r['hit_at_k']:>7.3f}  {note}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
