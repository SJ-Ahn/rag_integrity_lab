# 서비스 (API/CLI)

## 1. Endpoints

### `POST /api/chat` (Main Endpoint)
사용자의 자연어 질문을 받아 라우팅 후 답변을 반환합니다.

- **Request Body**:
    ```json
    {
      "question": "EC2 인스턴스 생성 방법 알려줘",
      "history": [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "안녕하세요!"}
      ]
    }
    ```

- **Logic (Smart Routing)**:
    1. **Intent Classification**: 
        - `chitchat`: 인사, 감정표현, 역할 질문 -> **즉시 응답** ("router" provider)
        - `search_query`: 기술 질문 -> **RAG 검색** 수행 ("local" or "openai" provider)
    2. **RAG Process** (if search_query):
        - Hybrid Retrieval (Vector + BM25) -> Rerank -> LLM Generation

- **Response Body**:
    ```json
    {
      "answer": "EC2 인스턴스는 콘솔에서 ...",
      "provider": "openai",
      "intent": "search_query",
      "sources": [
        {
          "rank": 1,
          "doc_id": "ec2-user-guide.pdf",
          "ref": "/docs/ec2-user-guide.pdf#page=12",
          "page": 12,
          "anchor": "..."
        }
      ]
    }
    ```

### `POST /ask` (Legacy RAG Only)
라우팅 없이 강제로 RAG 검색을 수행합니다. 디버깅 용도로 사용됩니다.

## 2. Logging & Monitoring
- **Router Logs**: `logs/router/service.log` (원시 LLM 응답 및 파싱 결과)
- **QA Logs**: `logs/qa/service.log` (질문, 의도, 답변, 인용 정보 기록)

