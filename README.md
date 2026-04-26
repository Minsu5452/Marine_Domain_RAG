# Marine Domain RAG

> 해양수산부·해양경찰청 소관 법령을 OPEN API 로 수집·파싱·인덱싱하고, RAG / GraphRAG / LangGraph 기반 워크플로우로 도메인 질의에 답하는 데모.

## ⚠️ 공개 범위 (Disclaimer)

본 저장소는 **재직 중 수행한 CDX 프로젝트** (한국 오픈소스 LLM 기반 해양 도메인 LLM 개발) 의 일부 흐름을 **본인이 별도로 재현한 데모**입니다.

- 회사·고객사 자산(원본 코드, 내부 데이터, 자체 LLM 가중치, fine-tuning 결과, 내부 평가 수치)은 보안 정책상 **공개하지 않습니다**.
- 본 레포에서 사용하는 데이터·코드는 모두 **공개 국가법령정보 OPEN API** 만으로 구성된 재현 환경입니다.
- 실제 프로젝트는 더 큰 규모(LoRA/QLoRA fine-tuning 실험, OCR 비정형 문서 처리, 자체 LLM 평가, 대규모 GraphRAG 인덱싱 등)이며, 본 레포는 그중 데이터 수집 → 파싱 → RAG/GraphRAG/LangGraph 흐름만 일부 시연합니다.

## Snapshot

| 항목 | 내용 |
| --- | --- |
| 도메인 | 해양 법률 (해양수산부·해양경찰청 소관) |
| 데이터 | 국가법령정보 OPEN API (JSON/HTML) |
| 인덱싱 | 한국어 임베딩 + FAISS / GraphRAG (조문↔용어↔법령 그래프) |
| 워크플로우 | LangGraph 기반 다단계 검색·답변 그래프 |
| LLM | 한국어 오픈소스 LLM (Hugging Face) — 사용자가 키 발급 시 swap |
| 평가 | 조문 인용 정확도, 검색 hit@k, 답변 일관성 (소규모 데모) |
| 원본 프로젝트 | SureSoftTech AX응용기술팀 / CDX (해양수산부·해양경찰청 전용 LLM) |
| 본인 역할 (원본) | 법률 데이터 수집·파싱, RAG·GraphRAG·LangGraph 파이프라인, 오픈소스 LLM fine-tuning 실험 (LoRA/QLoRA) |

## 핵심 흐름

```
[국가법령정보 OPEN API]
        │
   법령 검색 → 본문 수집 (JSON/HTML)
        │
   조문 단위 파싱 + 메타 (소관부처, 시행일, 법령ID)
        │
   ┌────────────────┬─────────────────┐
   │                │                 │
 임베딩+FAISS    GraphRAG 빌드    BM25 retrieval
   │                │                 │
   └────── LangGraph 워크플로우 ───────┘
        (질의 분해 → 검색 → 인용 → 답변 생성)
        │
   소규모 평가 (hit@k, 인용 정확도)
```

## Repository Structure

```
src/marine_domain_rag/
  collectors/         # 국가법령정보 OPEN API 클라이언트 (lawSearch / lawService)
  parsing/            # 법령 본문 → 조·항·호 구조화
  indexing/           # 임베딩 + FAISS / BM25 빌드
  graph/              # 조문↔용어↔법령 그래프 (GraphRAG)
  langgraph_app/      # LangGraph 기반 다단계 검색·답변 그래프
  llm/                # HF 오픈소스 LLM 래퍼 (key 없으면 mock 응답)
  evaluation/         # hit@k, 인용 정확도, 답변 일관성 (소규모)
notebooks/            # 단계별 데모 (수집 → 파싱 → 인덱싱 → 질의)
configs/              # YAML (소관부처 필터, 임베딩 모델명, retriever 파라미터)
scripts/              # CLI 진입점 (collect / parse / build / query / eval)
data/                 # raw / processed / external (gitignored)
```

## 실행 (재현 시)

```bash
pip install -e .

export LAW_OC="<국가법령정보 OC 키>"   # 예: daro98
# (옵션) HuggingFace 토큰 — 익명도 EXAONE GGUF 다운로드 가능하나 rate limit 회피용
export HF_TOKEN="<HF 토큰>"

# ⚠️ EXAONE GGUF (llama-cpp) + sentence-transformers 의 OMP/BLAS 라이브러리 충돌 회피
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# 1) 해수부·해경 소관 법령 수집
python -m marine_domain_rag.cli collect

# 2) 조문 단위 파싱
python -m marine_domain_rag.cli parse

# 3) 임베딩 + BM25 + FAISS 인덱스
python -m marine_domain_rag.cli index

# 4) GraphRAG 빌드
python -m marine_domain_rag.cli build-graph

# 5) LangGraph + EXAONE 질의
python -m marine_domain_rag.cli query --q "해양경찰청 소속 경찰공무원의 임용권자는?"

# 6) 소규모 평가
python -m marine_domain_rag.cli eval
```

**LLM 백엔드 선택** — `configs/default.yaml` 의 `llm.provider`:
- `exaone_gguf` (권장, llama.cpp + Q4_K_M GGUF, CPU 안정)
- `exaone` (transformers + MPS — Mac 에서 SBERT 와 동시 로드 시 SIGSEGV 가능)
- `mock` (LLM 없이 retrieve / citation 만 검증)

## Public Scope / Out of Scope

**포함**
- 국가법령정보 OPEN API 기반 법령 수집/파싱
- 한국어 임베딩 + FAISS RAG
- 조문 그래프 + GraphRAG
- LangGraph 기반 다단계 워크플로우
- HF 오픈소스 LLM 래퍼 (mock fallback)
- 소규모 평가 스위트 (hit@k, 인용 정확도)

**제외 (보안/규모)**
- 회사 자산 코드, 내부 데이터, 자체 LLM 가중치, 내부 평가 수치
- LoRA/QLoRA fine-tuning 실험 (원본은 수행, 본 레포는 미포함 — fine-tuning 후 성능 저하 사실만 본문 disclaimer 에 반영)
- OCR 기반 비정형 문서 처리 (실제 사용, 본 레포는 미포함)

## Links
- 원본 활동: SureSoftTech AX응용기술팀 / CDX (2025.06–2025.11)
- 사용 OPEN API
  - 국가법령정보 공동활용 (open.law.go.kr)
