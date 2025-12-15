# RAG Integrity Lab 개요

## 프로젝트 목적
AWS EC2 문서 기반의 RAG(Retrieval-Augmented Generation) 시스템을 구축하고, **로컬(GPU) 및 서버리스(API)** 환경 모두에서 동작하는 하이브리드 파이프라인을 검증합니다.

## 핵심 기능 (Phase 4 완료)
- **Dual Indexing**: 로컬(`BAAI/bge-m3`)과 서버리스(`OpenAI/text-embedding-3-small`) 환경에 맞는 독립적인 벡터 인덱스 운용.
- **Smart Router**: LLM 기반의 의도 분류(Intent Classification)를 통해 잡담(Chitchat)과 검색질문(Search Query)을 지능적으로 분기.
- **Hybrid Search**: FAISS(Vector) + BM25(Keyword) + CrossEncoder(Reranking)의 3단계 검색 파이프라인.
- **Multi-Provider Support**: `Local(Llama-3)`, `OpenAI(GPT-4o)`, `Gemini(Pro)` 간의 유연한 전환 지원.

## 프로젝트 구조
```
rag-integrity-lab/
├── data/             # 원본 문서, 전처리 청크, 인덱스 파일
├── ingest/           # 청크/임베딩 설정 및 스크립트
├── service/          # FastAPI 서버, Router, LLM Service Factory
├── evaluation/       # 정량적 평가 (Accuracy, Faithfulness)
├── docs/             # 프로젝트 문서 및 태스크 로그
├── scripts/          # 유틸리티 (인덱스 빌드, 청킹 등)
└── logs/             # 런타임 로그 (Router, QP, HTTP)
```

## 시스템 요구사항
- **Local Mode**: NVIDIA GPU (VRAM 12GB+ 권장), CUDA 12.x
- **Serverless Mode**: CPU 2 Core, RAM 4GB (GPU 불필요, OpenAI/Gemini API Key 필요)
