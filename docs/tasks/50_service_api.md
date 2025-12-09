# 서비스(API/CLI)

## /ask
입력: { query_ko }
출력: { answer_ko, citations[], latency_ms }

단계:
1. retrieve
2. rerank
3. LLM 생성
4. cite & 포맷팅
