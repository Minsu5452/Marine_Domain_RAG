# 해양 도메인 RAG

공개 국가법령정보 API에서 해양 관련 법령을 수집하고, 조문 단위로 파싱한 뒤 RAG, GraphRAG, LangGraph 기반 질의응답 흐름으로 연결한 데모 프로젝트입니다.

## 개요

| 항목 | 내용 |
| --- | --- |
| 분야 | 해양 법령 질의응답 |
| 데이터 | 국가법령정보 Open API |
| 검색 | 한국어 embedding, FAISS, text search, graph retrieval |
| 워크플로우 | LangGraph 기반 다단계 검색·답변 흐름 |
| LLM | local/open-source model wrapper, mock fallback |
| 역할 | 재직 중 수행한 해양 도메인 LLM/RAG 경험을 공개 데이터 기준으로 재현 |

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
