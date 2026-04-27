"""Multi-LLM 비용·품질 비교.

같은 retriever / 같은 QA 스위트로 여러 LLM provider 를 돌려
질문당 token 사용량 + cite_match_rate + latency 를 한 표에 모읍니다.

비교 후보(스크립트 인자 ``--candidates`` 로 추가/제외 가능):

  mock                : MockLLM (echo)
  exaone_gguf_q4      : LGAI EXAONE-3.5-2.4B-Instruct-GGUF Q4_K_M (현재 default)
  exaone_gguf_q8      : 같은 모델 Q8_0 quant (다운로드 가능 시)
  exaone              : transformers AutoModel (MPS/cuda)

환경상 모델 다운로드/로드가 불가하면 해당 후보는 ``status="load_failed"`` 로
기록되고 표에서 빠지지 않습니다 — JD 의 "비용 최적화" 의사결정에는 모델 로드
실패 자체도 정보입니다.

사용법:
    python scripts/eval_llm_cost_quality.py \
        --report reports/llm_cost_quality.json
    python scripts/eval_llm_cost_quality.py \
        --candidates mock,exaone_gguf_q4 \
        --report reports/llm_cost_quality.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from marine_domain_rag.config import Config
from marine_domain_rag.evaluation.retrieval_vs_llm import (
    _hit_in_citations,
    _coerce_state,
    load_suite,
)
from marine_domain_rag.graph.builder import load_graph
from marine_domain_rag.indexing.embed_index import load_index
from marine_domain_rag.langgraph_app.workflow import build_app
from marine_domain_rag.llm.exaone_loader import (
    ExaoneGGUFLLM,
    ExaoneLLM,
    MockLLM,
)

logger = logging.getLogger("eval_llm_cost_quality")


@dataclass
class Candidate:
    key: str
    label: str
    factory: callable  # () -> llm
    notes: str = ""


def _build_candidates(cfg: Config) -> list[Candidate]:
    repo = cfg.get("llm", "gguf_repo_id",
                   default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF")
    return [
        Candidate(
            key="mock",
            label="MockLLM (echo)",
            factory=lambda: MockLLM(),
            notes="실제 LLM 미적용 baseline. 비용 0.",
        ),
        Candidate(
            key="exaone_gguf_iq4xs",
            label="EXAONE-3.5-2.4B GGUF IQ4_XS",
            factory=lambda: ExaoneGGUFLLM(
                repo_id=repo,
                filename="EXAONE-3.5-2.4B-Instruct-IQ4_XS.gguf",
            ),
            notes="가장 작은 quant (~1.4GB). 비용 우선.",
        ),
        Candidate(
            key="exaone_gguf_q4",
            label="EXAONE-3.5-2.4B GGUF Q4_K_M",
            factory=lambda: ExaoneGGUFLLM(
                repo_id=repo,
                filename=cfg.get("llm", "gguf_filename",
                                 default="EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"),
            ),
            notes="현재 default. CPU 안정·균형.",
        ),
        Candidate(
            key="exaone_gguf_q5",
            label="EXAONE-3.5-2.4B GGUF Q5_K_M",
            factory=lambda: ExaoneGGUFLLM(
                repo_id=repo,
                filename="EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf",
            ),
            notes="품질 우선 후보 (~2GB). 메모리 ↑.",
        ),
        Candidate(
            key="exaone",
            label="EXAONE-3.5-2.4B transformers",
            factory=lambda: ExaoneLLM(
                cfg.get("llm", "model_name",
                        default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"),
                device=cfg.get("llm", "device", default="auto"),
                dtype=cfg.get("llm", "dtype", default="float16"),
            ),
            notes="MPS/cuda 백엔드. SBERT 동시 로드 시 SIGSEGV 위험.",
        ),
    ]


def _eval_one_candidate(cand: Candidate, retriever, graph, suite, top_k: int,
                        release_retriever_to_cpu: bool) -> dict:
    record = {
        "key": cand.key,
        "label": cand.label,
        "notes": cand.notes,
        "status": "ok",
    }
    t_load_0 = time.time()
    try:
        llm = cand.factory()
    except Exception as e:  # noqa: BLE001
        record["status"] = "load_failed"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc(limit=3)
        return record
    record["load_time_sec"] = round(time.time() - t_load_0, 2)
    record["llm_name"] = getattr(llm, "name", cand.key)

    app = build_app(retriever, graph, llm, top_k=top_k,
                    release_retriever_to_cpu=release_retriever_to_cpu)

    cite_hits = 0
    nonempty = 0
    total_latency = 0.0
    total_prompt = 0
    total_completion = 0
    per_sample = []
    for item in suite:
        q = str(item["question"])
        try:
            t0 = time.time()
            out = app.invoke({"question": q})
            latency = time.time() - t0
        except Exception as e:  # noqa: BLE001
            logger.warning("invoke fail (%s): %s", q, e)
            per_sample.append({"question": q, "error": str(e)})
            continue
        cit, ans = _coerce_state(out)
        usage = getattr(llm, "last_usage", {}) or {}
        prompt_tok = int(usage.get("prompt_tokens", 0))
        comp_tok = int(usage.get("completion_tokens", 0))
        total_prompt += prompt_tok
        total_completion += comp_tok
        total_latency += latency
        if ans.strip():
            nonempty += 1
        if _hit_in_citations(item, cit, k=top_k):
            cite_hits += 1
        per_sample.append({
            "question": q,
            "latency_sec": round(latency, 3),
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "answer_preview": ans[:160],
        })

    n = max(len(suite), 1)
    record.update({
        "n": len(suite),
        "cite_match_rate": cite_hits / n,
        "nonempty_rate": nonempty / n,
        "avg_latency_sec": total_latency / n,
        "avg_prompt_tokens": total_prompt / n,
        "avg_completion_tokens": total_completion / n,
        "avg_total_tokens": (total_prompt + total_completion) / n,
        "per_sample": per_sample,
    })
    return record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--report", default="reports/llm_cost_quality.json")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--candidates", default=None,
                   help="콤마 구분 키 (mock,exaone_gguf_q4,exaone_gguf_q8,exaone). "
                        "기본은 전 후보.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config)

    retriever = load_index(cfg.get("paths", "faiss_dir"))
    graph_path = cfg.get("paths", "graph_path")
    g = load_graph(graph_path) if Path(graph_path).exists() else None
    top_k = args.top_k or int(cfg.get("retrieval", "top_k", default=5))
    suite = load_suite(cfg.get("paths", "eval_suite"))

    cands = _build_candidates(cfg)
    if args.candidates:
        keep = {x.strip() for x in args.candidates.split(",") if x.strip()}
        cands = [c for c in cands if c.key in keep]

    results = []
    for c in cands:
        logger.info("--- evaluating %s (%s) ---", c.key, c.label)
        results.append(_eval_one_candidate(
            c, retriever, g, suite, top_k,
            release_retriever_to_cpu=bool(
                cfg.get("llm", "release_retriever_to_cpu", default=False)),
        ))

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "top_k": top_k,
            "n": len(suite),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Multi-LLM cost vs quality ===")
    header = f"{'key':<18} {'status':<13} {'cite':>6} {'tok/q':>7} {'lat(s)':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['key']:<18} {r.get('status','?'):<13} {'-':>6} {'-':>7} {'-':>7}")
            continue
        print(f"{r['key']:<18} {r['status']:<13} "
              f"{r['cite_match_rate']:>6.3f} "
              f"{r['avg_total_tokens']:>7.1f} "
              f"{r['avg_latency_sec']:>7.2f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
