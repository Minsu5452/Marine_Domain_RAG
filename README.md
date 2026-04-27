# 해양 도메인 RAG

공개 국가법령정보 API에서 해양 관련 법령을 수집하고, 조문 단위로 파싱한 뒤 RAG, GraphRAG, LangGraph 기반 질의응답 흐름으로 연결한 데모 프로젝트입니다. 슈어소프트테크 인턴십 중 해양경찰청 CDX 과제로 수행한 작업을, 회사 보안상 사내 코드·데이터를 외부에 공개할 수 없어 공개 데이터 기반으로 가볍게 재현했습니다.

## 개요

| 항목 | 내용 |
| --- | --- |
| 분야 | 해양 법령 질의응답 |
| 데이터 | 국가법령정보 Open API |
| 검색 | 한국어 embedding, FAISS, text search, graph retrieval |
| 워크플로우 | LangGraph 기반 다단계 검색·답변 흐름 |
| LLM | local/open-source model wrapper, mock fallback |
| 역할 | 슈어소프트테크 인턴십 중 해양경찰청 CDX 과제로 수행한 작업을, 회사 보안상 사내 코드·데이터를 외부에 공개할 수 없어 공개 데이터 기반으로 가볍게 재현한 데모 |

## 파이프라인

```text
법령 Open API
  -> raw JSON / HTML 수집
  -> 조문 / 항 / 호 단위 파싱
  -> embedding index + graph index 생성
  -> LangGraph 검색 workflow
  -> 조문 인용 기반 답변 생성
  -> 소규모 QA 평가
```

## 저장소 구성

```text
.
├── configs/
│   └── default.yaml
├── notebooks/
│   └── 01_demo_pipeline.ipynb
├── src/marine_domain_rag/
│   ├── collectors/      # 법령 API client
│   ├── parsing/         # 조문 parser
│   ├── indexing/        # embedding / FAISS index
│   ├── graph/           # 조문 / 용어 / 법령 graph
│   ├── langgraph_app/   # 검색 workflow
│   ├── llm/             # open-source LLM wrapper
│   └── evaluation/      # QA 평가 utility
├── tests/
│   └── qa_pairs.yaml
└── pyproject.toml
```

## 실행

```bash
pip install -e .

export LAW_OC="<law-api-oc-key>"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python -m marine_domain_rag.cli collect
python -m marine_domain_rag.cli parse
python -m marine_domain_rag.cli index
python -m marine_domain_rag.cli build-graph
python -m marine_domain_rag.cli query --q "해양경찰청 소속 경찰공무원의 임용권자는?"
python -m marine_domain_rag.cli eval
```

LLM을 로드하지 않고 검색 품질만 확인할 때는 `configs/default.yaml`에서 mock provider를 사용합니다.

## Retrieval vs LLM contribution

기존 `eval` 명령은 hit@k(retrieval 기여)와 cite_match(LLM의 인용 충실도)를 한 번에 측정해, 성능이 떨어졌을 때 어느 컴포넌트가 원인인지 분리하기 어려웠습니다. 컴포넌트별 기여도를 가시화하기 위해 두 단계로 나눠 측정하는 스크립트를 추가했습니다.

```bash
# retrieval-only 빠른 측정 (LLM 로드 없이 hybrid retriever만 평가)
python scripts/eval_retrieval_vs_llm.py --skip-llm \
    --report reports/retrieval_vs_llm_skip_llm.json

# LLM-augmented 전체 측정 (configs/default.yaml의 llm.provider 사용)
python scripts/eval_retrieval_vs_llm.py \
    --report reports/retrieval_vs_llm.json
```

`tests/qa_pairs.yaml` 각 샘플에 `gold_citations`(우선 정답 인용)와 `acceptable_alternatives`(허용 대안)를 추가해, 정답이 한 조문으로 좁혀지지 않는 사례도 평가에 반영했습니다.

리포트는 다음 4개 지표 + 3가지 오류 분류를 함께 출력합니다.

| 지표 | 의미 |
| --- | --- |
| retrieval_hit@k | retriever 단독으로 정답 조문을 top-k 안에 가져온 비율 |
| cite_match_rate | LangGraph 워크플로우 최종 citations에 정답이 포함된 비율 |
| nonempty_rate | LLM이 비공백 답변을 생성한 비율 |
| avg_latency_sec | 샘플당 평균 응답 시간 |
| retrieved_but_not_cited | retrieval은 정답을 가져왔지만 최종 인용에 빠진 사례 수 |
| not_retrieved | retrieval부터 정답을 못 가져온 사례 수 |
| llm_hallucination | 답변 본문에 컨텍스트의 어떤 법령명도 등장하지 않은 사례 수 (휴리스틱) |

데모 스위트(5개 샘플) 기준 retrieval-only 모드 측정값은 다음과 같습니다(LLM은 로드하지 않음, mock fallback).

| samples | retrieval_hit@k | cite_match_rate | nonempty | retrieved_but_not_cited | not_retrieved | llm_hallucination | avg_latency_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.800 | 0.600 | 1.000 | 1 | 1 | 0 | 0.07 |

해석: top-5 안에 정답을 못 가져온 1건은 retriever 단계에서 개선해야 하고, 가져왔지만 인용으로 안 넘어간 1건은 rerank/cite 단계의 점수 컷오프 또는 graph_expand 가중치 문제로 좁혀집니다. LLM-augmented 모드를 추가로 돌리면 같은 5건 중 LLM이 외부 지식만으로 답해버린 케이스(`llm_hallucination`)가 별도로 카운트되어, "검색은 잘 됐는데 LLM이 무시했다" 와 "검색이 못 따라갔다" 를 분리해 볼 수 있습니다.

## Query decomposition 전략 ablation

LangGraph 워크플로우의 `node_decompose` 가 사용하는 분해 전략을 추상화해 (`src/marine_domain_rag/langgraph_app/decompose.py`), 같은 retriever / 같은 QA 스위트로 hit@k 변화를 비교했습니다.

```bash
python scripts/eval_decompose_ablation.py \
    --report reports/decompose_ablation.json
# LLM 미가용 환경에서는 --skip-llm 으로 raw / noun 만 실측, llm 은 fallback 표시
python scripts/eval_decompose_ablation.py --skip-llm \
    --report reports/decompose_ablation_skip_llm.json
```

| strategy | 설명 | hit@k |
| --- | --- | --- |
| raw | 분해 없이 원 질문 한 번 검색 | 0.800 |
| noun | 길이 ≥ 2 토큰 상위 3개 + 원 질문 (현재 baseline) | 0.600 |
| llm | LLM 분해, mock/미가용 환경에서는 noun fallback (사유 기록) | 0.600 (fallback) |

데모 환경(EXAONE GGUF 로드 불가 가정)에서는 `llm` 전략이 `noun` 으로 fallback 되어 동일 점수가 나옵니다. 흥미로운 관찰은 이 작은 스위트에서 `raw` 가 `noun` 보다 +0.2 더 높다는 점인데, 명사 추출이 조사·어미가 붙은 토큰("임용에", "어떻게")을 sub_query 로 추가해 BM25 노이즈로 작용한 결과로 해석됩니다. 실 운영에서는 KoNLPy 같은 형태소 분석기로 명사만 추출하거나 LLM 분해를 사용해야 하며, 본 ablation 스크립트가 그 비교 인프라를 제공합니다.

## Multi-LLM 비용·품질 비교

같은 retriever / 같은 QA 스위트로 여러 LLM 후보를 돌려 질문당 token 사용량 + cite_match_rate + latency 를 비교하는 스크립트입니다 (`scripts/eval_llm_cost_quality.py`). 비교 후보는 GGUF quant 단계와 transformers 백엔드를 포함하며, 환경상 다운로드/로드가 불가한 후보는 `status="load_failed"` 로 기록되어 의사결정 근거로 남습니다.

```bash
python scripts/eval_llm_cost_quality.py \
    --candidates mock,exaone_gguf_iq4xs,exaone_gguf_q4 \
    --report reports/llm_cost_quality.json
```

데모 스위트(5개 샘플) 실측치는 다음과 같습니다 (Mac M-series CPU, n_ctx=4096).

| 후보 | load(s) | avg prompt tok | avg completion tok | cite_match | avg latency (s) | 비고 |
| --- | --- | --- | --- | --- | --- | --- |
| MockLLM (echo) | 0.00 | 165.4 | 57.2 | 0.600 | 0.07 | 비용 0 baseline. retrieval 결과 노출 전용. |
| EXAONE-3.5-2.4B GGUF IQ4_XS | 48.12 | 475.8 | 149.6 | 0.600 | 5.86 | 가장 작은 quant. 첫 로드 시 다운로드 포함. |
| EXAONE-3.5-2.4B GGUF Q4_K_M | 2.24 | 475.8 | 116.6 | 0.600 | 5.43 | 현재 default. 캐시 적중. |

두 GGUF quant 의 prompt 토큰은 동일 컨텍스트 사용으로 같으며, completion 토큰과 latency 가 약간 다릅니다. 본 5개 스위트 기준 cite_match 는 동률이라 retrieval 단계가 이미 정답 조문을 top-k 안에 넣어주면 quant 차이는 답변 품질에 결정적이지 않다는 신호로 해석됩니다.

### 비용 효율적 모델 선택 가이드

해당 데모 규모에서는 retrieval 정확도가 cite_match 의 상한을 결정하므로, LLM은 컨텍스트를 충실히 따라갈 수 있는 가장 작은 모델 (현 데모에서는 EXAONE-3.5-2.4B GGUF Q4_K_M, ~1.5GB) 부터 시작하는 것이 비용 효율적입니다. 답변의 완결성·문체·지시 준수가 더 중요한 사용 사례에서는 동일 모델의 Q5_K_M / Q6_K 로 단계적으로 키우며 cite_match 와 nonempty_rate, 그리고 사용자 만족도(별도 평가 필요)의 기울기를 측정해 멈출 지점을 찾는 것을 권장합니다. 본 스크립트가 그 단계별 비교 인프라(load 시간, 토큰 평균, cite_match, latency)를 한 표로 제공합니다.
