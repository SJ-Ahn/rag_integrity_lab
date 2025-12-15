# 인덱스 & 리트리버 (Dual Indexing Strategy)

## 1. 개요 (Overview)
`LLM_PROVIDER` 설정에 따라 서로 다른 임베딩 모델과 벡터 인덱스를 자동으로 로드하여 최적의 검색 성능을 제공합니다.

## 2. 모드별 구성 (Configuration)

### A. Local Mode (GPU Required)
- **Embedding Model**: `BAAI/bge-m3` (Dense + Sparse)
- **Index Path**: `data/index/local`
- **Features**: 
    - FAISS HNSW (M=32, efSearch=96)
    - BM25 (형태소 분석 기반 키워드 검색)
    - Hybrid Weights: Vector(0.7) + BM25(0.3)

### B. Serverless Mode (OpenAI/Gemini)
- **Embedding Model**: `text-embedding-3-small` (OpenAI API)
- **Index Path**: `data/index/openai`
- **Features**: 
    - FAISS Flat/HNSW (API 임베딩 사용)
    - **Optimization**: `tiktoken`을 사용하여 8192 토큰 초과 청크 자동 Truncation.
    - **Resilience**: API Rate Limit (429) 발생 시 Exponential Backoff 적용.

## 3. 재랭킹 (Reranking)
- **Model**: `BAAI/bge-reranker-base` (Cross-Encoder)
- **Process**: 
    1. 1차 검색(Hybrid)으로 Top-50 추출
    2. Reranker로 Query-Document 연관성 정밀 채점
    3. 최종 Top-8 생성 및 LLM 전달
