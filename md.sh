#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# RAG Integrity Lab – Markdown 자동 생성 스크립트 (v2)
# 00_overview.md ~ 99_quickstart.md 전부 생성
# ==========================================================
# 실행 방법:
#   chmod +x generate_all_md.sh
#   ./generate_all_md.sh
# ==========================================================

TARGET_DIR="docs/tasks"
mkdir -p "$TARGET_DIR"

echo "📂 생성 경로: $TARGET_DIR"
echo "---------------------------------------------"

# ---------- 00_overview.md ----------
cat <<'EOF' > "$TARGET_DIR/00_overview.md"
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
EOF

# ---------- 01_environment.md ----------
cat <<'EOF' > "$TARGET_DIR/01_environment.md"
# 운영 환경 세팅 (Ubuntu 24.04, CUDA 12.8, Python 3.10+)

## 하드웨어
- RAM: 32GB
- GPU: RTX 5060 Ti (CUDA 12.8)

## Python 환경 관리
- 권장: **uv + venv** (빠름/가볍고 충돌 적음)
- 대안: **micromamba** (conda 대체)
EOF

# ---------- 10_data_prepare.md ----------
cat <<'EOF' > "$TARGET_DIR/10_data_prepare.md"
# 데이터 준비 & 정규화

## 입력
- HTML 10,268개, PDF 1개
- 경로: `data/source/html/`, `data/source/pdf/`
- 무시: css/js/jsp 등 비텍스트 파일

## 출력
- `data/working/normalized/*.jsonl`
- `data/working/meta/*.json`
EOF

# ---------- 20_chunking.md ----------
cat <<'EOF' > "$TARGET_DIR/20_chunking.md"
# 청크 전략

## 목표
인용 정확도 + 긴 문맥 보존의 균형 유지

## 설정
- `chunk_size`: 900 tokens
- `chunk_overlap`: 150
- 헤더 기반 분할 (h2~h4)
- `<a name>` 앵커 유지
EOF

# ---------- 30_index_retriever.md ----------
cat <<'EOF' > "$TARGET_DIR/30_index_retriever.md"
# 인덱스 & 리트리버 (정확도 우선 / P95 10s)

- 임베딩: BAAI/bge-m3
- 인덱스: FAISS HNSW (M=32, efSearch=96)
- 하이브리드: FAISS + BM25 (0.65:0.35)
- 후보: Top-50 → Rerank → Top-8
EOF

# ---------- 35_reranker.md ----------
cat <<'EOF' > "$TARGET_DIR/35_reranker.md"
# Reranker (BAAI/bge-reranker-base)

1) Hybrid 검색 Top-50
2) Cross-Encoder rerank → Top-8
3) 중복 anchor 제거
4) 긴 답변 컨텍스트 생성
EOF

# ---------- 40_system_prompt_and_citation.md ----------
cat <<'EOF' > "$TARGET_DIR/40_system_prompt_and_citation.md"
# 시스템 프롬프트 & 인용 규칙

```

당신은 AWS EC2 및 DBMS 공식 문서를 근거로 한국어로만 답합니다.
근거가 부족하면 "충분한 근거를 찾지 못했습니다."라고 답하고,
추가로 필요한 정보를 제안하세요.

[인용 규칙]

* 각 단락마다 최소 1개 [n]
* [n]은 doc_id#anchor와 일치
* 원문 인용은 영문 그대로 유지

```
EOF

# ---------- 41_korean_answer_style.md ----------
cat <<'EOF' > "$TARGET_DIR/41_korean_answer_style.md"
# 한국어 답변 스타일 가이드

- 문체: 정확·절차적·명확
- 외래어 병기 (예: Realm, Policy)
- 구조: 설명 → 절차 → 주의 → 인용
EOF

# ---------- 50_service_api.md ----------
cat <<'EOF' > "$TARGET_DIR/50_service_api.md"
# 서비스(API/CLI)

## /ask
입력: { query_ko }
출력: { answer_ko, citations[], latency_ms }

단계:
1. retrieve
2. rerank
3. LLM 생성
4. cite & 포맷팅
EOF

# ---------- 60_evaluation.md ----------
cat <<'EOF' > "$TARGET_DIR/60_evaluation.md"
# 자동 평가 (골든셋 기반)

- 질문 ≥30개
- 지표:
  - Citation ≥92%
  - Recall@5 ≥95%
  - Faithfulness ≥85%
  - Hallucination ≤4%
EOF

# ---------- 70_logging_monitoring.md ----------
cat <<'EOF' > "$TARGET_DIR/70_logging_monitoring.md"
# 로깅 & 모니터링

## 로그 구분
- env/: 라이브러리·GPU 정보
- http/: 요청/응답
- qa/: 질문/답변/인용
- latency/: 단계별 시간
EOF

# ---------- 80_ci_repo.md ----------
cat <<'EOF' > "$TARGET_DIR/80_ci_repo.md"
# 리포지토리 & CI

- GitHub 비공개 저장소 권장
- gitignore:
```

data/source/
data/working/
data/index/
logs/
.venv/

````
EOF

# ---------- 90_kpi_policy.md ----------
cat <<'EOF' > "$TARGET_DIR/90_kpi_policy.md"
# KPI 정책

- Citation ≥92%
- Recall@5 ≥95%
- Faithfulness ≥85%
- Hallucination ≤4%
- P95 ≤10.0s
EOF

# ---------- 95_ablation_tuning.md ----------
cat <<'EOF' > "$TARGET_DIR/95_ablation_tuning.md"
# 어블레이션 & 튜닝

변수:
- Splitter: baseline / header_hybrid
- Hybrid α: 0.65:0.35 / 0.7:0.3
- TopK: 5/8/12
- Rerank: on/off
EOF

# ---------- 96_troubleshooting.md ----------
cat <<'EOF' > "$TARGET_DIR/96_troubleshooting.md"
# 트러블슈팅

## 성능 문제
- 지연↑: reranker batch↓
- GPU OOM: chunk_size↓
- Recall↓: BM25 비중↑

## 품질 문제
- 인용 누락: anchor 강화
- Faithfulness↓: rerank on
EOF

# ---------- 97_experiment_notes.md ----------
cat <<'EOF' > "$TARGET_DIR/97_experiment_notes.md"
# 실험 노트 템플릿

| 날짜 | 설정 | 변수 | 결과 | 메모 |
|------|------|------|------|------|
| 2025-11-05 | baseline | α=0.65:0.35 | Citation 92.3% | 기준 측정 |
EOF

# ---------- 98_makefile_snippet.md ----------
cat <<'EOF' > "$TARGET_DIR/98_makefile_snippet.md"
# Makefile 예시

```make
.PHONY: prepare chunks index serve eval report
prepare:
\tpython scripts/normalize_docs.py --in data/source --out data/working/normalized
chunks:
\tpython scripts/make_chunks.py --cfg ingest/cfg/ingest_baseline.yaml
index:
\tpython scripts/build_faiss.py --cfg ingest/cfg/index_bge_m3.yaml
serve:
\tuvicorn service.app:app --port 8080
eval:
\tpython evaluation/run_eval.py --cfg evaluation/cfg/eval_grid.yaml
report:
\tpython evaluation/reporters/make_summary.py --in evaluation/results --out docs/results_summary.md
````

EOF

# ---------- 99_quickstart.md ----------

cat <<'EOF' > "$TARGET_DIR/99_quickstart.md"

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
EOF

echo "✅ 모든 Markdown 파일 생성 완료!"
ls -1 "$TARGET_DIR"/*.md
