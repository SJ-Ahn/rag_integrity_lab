
# QuickStart

```bash
# 환경 준비
source .venv/bin/activate
uv pip install -r requirements.txt

# 정규화
python scripts/normalize_docs.py

# 청크
python scripts/make_chunks.py

# 인덱스
python scripts/build_faiss.py

# 서비스
python service/app.py

# 평가
python evaluation/run_eval.py
```

기본 설정:

* 임베딩: bge-m3
* 리트리버: FAISS+BM25 (α=0.65:0.35)
* Reranker: ON (Top-50→Top-8)
