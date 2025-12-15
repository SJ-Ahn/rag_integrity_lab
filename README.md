# RAG Integrity Lab

**RAG Integrity Lab**은 AWS EC2 및 DBMS 문서 기반의 **서버리스/로컬 하이브리드 RAG 시스템**입니다.  
LLM 기반의 **스마트 라우터**를 통해 비용을 최적화하고, CI 파이프라인을 통해 답변의 정확도(Accuracy)와 근거 충실성(Faithfulness)을 지속적으로 검증합니다.

---

## 🚀 Key Features

### 1. Hybrid & Dual Indexing
- **Local Mode**: GPU 환경에서 Local LLM(`Llama-3`) 및 `bge-m3` 임베딩을 사용하여 완전한 로컬 RAG 수행.
- **Serverless Mode**: GPU가 없는 환경에서도 OpenAI/Gemini API를 사용하여 고품질 RAG 수행.
- **Dynamic Switching**: 환경변수 `LLM_PROVIDER` 설정 하나로 인덱스 경로와 임베딩 모델 자동 전환.

### 2. Smart Router (Cost Optimization)
- 사용자의 질문 의도를 분석하여 **Chitchat(무료/저지연)**과 **Search Query(유료/고지연)**를 분리.
- 단순 인사나 역할 질문은 RAG 파이프라인을 거치지 않고 router가 0.01초 내에 즉시 응답.

### 3. Automated Integrity Check
- **Accuracy**: 제공된 정답 셋(Golden Dataset)과 RAG 답변의 일치율 검증.
- **Faithfulness**: 답변이 검색된 문서(Context)에 기반했는지 검증(Hallucination 방지).

---

## 🛠️ Installation

```bash
# 1. Clone Repository
git clone https://github.com/SJ-Ahn/rag_integrity_lab.git
cd rag_integrity_lab

# 2. Setup Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure API Keys (for Serverless Mode)
cp .env.example .env
# Edit .env and set OPENAI_API_KEY / GEMINI_API_KEY
```

---

## 🏃 Quick Start

### 1. Web Service (Chat Interface)
웹 인터페이스(Chat UI)를 실행하여 AI와 대화할 수 있습니다.
```bash
# Run Web Server (Port 8889)
./run_web.sh
```
- Access: `http://localhost:8889`
- Logs: `logs/router/service.log` (Router Decision)

### 2. Auto Automation (Index Build -> Eval)
전체 파이프라인(문서 청킹 -> 인덱스 빌드 -> 평가)을 한번에 실행합니다.
```bash
./run_all.sh
```

---

## 📚 Documentation
- [Project Overview](docs/tasks/00_overview.md)
- [Dual Indexing Strategy](docs/tasks/30_index_retriever.md)
- [Service API & Router](docs/tasks/50_service_api.md)
- [Phase 4 Walkthrough & Debugging](docs/tasks/phase4_walkthrough.md)
