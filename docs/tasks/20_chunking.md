# 청크 전략

## 목표
인용 정확도 + 긴 문맥 보존의 균형 유지

## 설정
- `chunk_size`: 900 tokens
- `chunk_overlap`: 150
- 헤더 기반 분할 (h2~h4)
- `<a name>` 앵커 유지
