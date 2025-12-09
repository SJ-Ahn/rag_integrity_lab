# rag_integrity_lab 개요

## 프로젝트 목적
동일한 AWS EC2 및 DBMS 문서 세트를 기반으로 RAG 파이프라인의 질문·답변·출처 매핑의 정확도를 반복 검증하고 정량화합니다.

## 핵심 목표
- **정확성(Accuracy):** 질문-답변-인용 일치율 92% 이상
- **근거 충실성(Faithfulness):** 원문 대비 일치율 85% 이상
- **재현성(Reproducibility):** 동일 조건에서 결과 동일
- **자동 평가(Automation):** CI 파이프라인으로 테스트 반복 가능

## 구성 구조
```

rag-integrity-lab/
├── data/             # 원본·전처리 문서
├── ingest/           # 청크/임베딩 스크립트
├── service/          # RAG API(FastAPI)
├── evaluation/       # 평가 세트·채점 스크립트
├── docs/             # 보고서·실험노트
├── logs/             # 로그 및 환경정보
└── tools/            # 유틸리티 스크립트

```

## 장비 사양
- RAM: 32GB
- GPU: RTX 5060 Ti (CUDA 12.8)
- Python 3.10+, Ubuntu 24.04
