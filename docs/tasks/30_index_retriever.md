# 인덱스 & 리트리버 (정확도 우선 / P95 10s)

- 임베딩: BAAI/bge-m3
- 인덱스: FAISS HNSW (M=32, efSearch=96)
- 하이브리드: FAISS + BM25 (0.65:0.35)
- 후보: Top-50 → Rerank → Top-8
