# Phase 4 (Serverless RAG & Dual Indexing) & Router Debugging Report

## 1. 구현 요약 (Implementation Summary)

### 1.1 Dual Indexing & Serverless RAG
- **목적**: GPU가 없는 환경(Serverless)에서도 외부 API(OpenAI/Gemini)를 사용하여 RAG를 수행하고, 로컬 환경(GPU)에서는 로컬 모델로 전환 가능하도록 함.
- **방식**:
    - `LLM_PROVIDER` 환경 변수에 따라 `INDEX_DIR` 및 `EMBEDDING_MODEL` 동적 변경.
    - **Local**: `data/index/local` (BAAI/bge-m3)
    - **OpenAI/Gemini**: `data/index/openai` (text-embedding-3-small)
- **성과**: 단일 코드베이스로 두 가지 운영 모드 완벽 지원.

### 1.2 채팅 라우터 (Chat Router) 최적화
- **목적**: 잡담(Chitchat)과 기술 질문(Search Query)을 정확히 분류하여 비용 절감.
- **개선 사항**:
    - **Prompt Engineering**: "너는 무엇을 할수 있니?"와 같은 역량 질문을 잡담으로 처리하도록 규칙 강화.
    - **Robustness**: 입력값 정제(Sanitization) 및 LLM 생성 중단(Stop Tokens) 로직 추가로 안정성 확보.

## 2. 주요 트러블슈팅 (Troubleshooting)

| 문제 현상 | 원인 분석 | 해결 방법 |
|:---:|:---|:---|
| **Rate Limit / 400 Bad Request** | OpenAI 임베딩 시 청크 토큰 초과 | `tiktoken` 기반 Truncation 로직 및 Exponential Backoff 구현 |
| **Router Crash / Invalid JSON** | Local Llama 모델이 JSON 생성 후에도 텍스트를 계속 생성("hallucinating interaction") | `stop=["Input:", "\n\n", "}"]` 파라미터 추가로 생성 강제 종료 및 Regex 추출 개선 |
| **Router Misclassification** | "뭐 할 수 있어?" 질문을 기술 검색(`search_query`)으로 오분류 | 프롬프트에 "역할/기능 질문은 chitchat" 규칙 명시 및 Few-shot 예제 추가 |
| **App.py 500 Error** | Router 로깅 설정 누락으로 에러가 숨겨짐 | `service.router` 로거 설정 추가 및 `chat_endpoint`에 Intent 로깅 추가 |

## 3. 검증 결과 (Verification)

### 자동화 테스트 (`test_phase4_dual_mode.py`)
- ✅ **Local Mode**: 인덱스 로딩, 검색, 라우팅 정상 확인.
- ✅ **OpenAI Mode**: 인덱스 로딩, 검색, 라우팅 정상 확인.

### 실서버 테스트 (`curl`)
- ✅ **잡담**: "너는 무엇을 할수 있니?" -> `Intent: chitchat` (Correct), 답변 생성 확인.
- ✅ **일반 질문**: "EC2 생성법" -> `Intent: search_query` -> RAG 검색 수행.

시스템이 안정화되었습니다.
