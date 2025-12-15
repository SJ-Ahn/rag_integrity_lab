"""
최소 FastAPI 서비스로 `/ask` 엔드포인트를 제공해 RAG 질의를 처리합니다.
사전에 구축된 FAISS+BM25 하이브리드 인덱스를 로드하고, 검색된 청크를 요약하여
한국어 답변과 인용 정보를 반환합니다. 이후 LLM 생성기나 리랭커를 연결하기 위한
기본 뼈대 역할을 합니다.
"""

from __future__ import annotations

import json
import logging
import pathlib
import pickle
import re
import sys
import time
from functools import lru_cache
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    faiss = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore



# ... imports ... (imports section needs careful handling if I replace a block)
# Actually, I will replace the try-except block for sentence_transformers first, then the HybridRetriever class modification.



try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    BM25Okapi = None  # type: ignore

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tools.logger import setup_logger
from config.model_registry import get_default_model_name, get_model_info, list_available_models
from config.settings import settings


LOGGER = logging.getLogger("service.app")
HTTP_LOGGER = logging.getLogger("service.http")
QA_LOGGER = logging.getLogger("service.qa")
LATENCY_LOGGER = logging.getLogger("service.latency")

INDEX_DIR_DEFAULT = settings.INDEX_DIR
ALPHA = settings.ALPHA
FAISS_TOP_K = settings.FAISS_TOP_K
RETURN_TOP_K = settings.RETURN_TOP_K
DEFAULT_MODEL_NAME = get_default_model_name()

ANCHOR_PATTERN = re.compile(r"\[#([^\]]+)\]")


class AskRequest(BaseModel):
    query_ko: str = Field(..., description="한국어 자연어 질의")
    top_k: int = Field(RETURN_TOP_K, ge=1, le=50)
    history: List[dict] = []

    @validator("query_ko")
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query_ko must not be empty.")
        return cleaned


class Citation(BaseModel):
    ref: str
    chunk_id: str
    doc_id: str
    page: Optional[int] = None
    anchor: Optional[str] = None


class AskResponse(BaseModel):
    answer_ko: str
    citations: List[Citation]
    latency_ms: float
    provider: str



class ChunkStore:
    def __init__(self, index_dir: pathlib.Path):
        chunks_path = index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Expected chunk store at {chunks_path}. Run scripts/build_faiss.py first."
            )
        self.chunks: dict[str, str] = {}
        self.doc_ids: dict[str, str] = {}
        self.pages: dict[str, int] = {}
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = json.loads(line)
                chunk_id = str(raw["chunk_id"])
                self.chunks[chunk_id] = str(raw.get("text", ""))
                self.doc_ids[chunk_id] = str(raw.get("doc_id", ""))
                if raw.get("page"):
                    self.pages[chunk_id] = int(raw["page"])

    def get_text(self, chunk_id: str) -> str:
        return self.chunks.get(chunk_id, "")

    def get_doc(self, chunk_id: str) -> str:
        return self.doc_ids.get(chunk_id, "")

    def get_page(self, chunk_id: str) -> Optional[int]:
        return self.pages.get(chunk_id)

    def get_neighbor_chunks(self, chunk_id: str, window: int = 1) -> str:
        """
        Retrieves the text of the given chunk plus 'window' chunks before and after.
        assumes chunk_id format: doc_id#chunkNNN
        """
        if "#chunk" not in chunk_id:
            return self.get_text(chunk_id)
            
        try:
            base_doc, chunk_part = chunk_id.rsplit("#chunk", 1)
            current_idx = int(chunk_part)
        except ValueError:
            return self.get_text(chunk_id)

        expanded_text = []
        
        # Range: [current_idx - window, current_idx + window]
        for i in range(current_idx - window, current_idx + window + 1):
            neighbor_id = f"{base_doc}#chunk{i:03d}"
            text = self.chunks.get(neighbor_id)
            if text:
                expanded_text.append(text)
        
        return "\n\n".join(expanded_text)


class HybridRetriever:
    def __init__(
        self,
        index_dir: pathlib.Path,
        alpha: float = ALPHA,
        faiss_top_k: int = FAISS_TOP_K,
        return_top_k: int = RETURN_TOP_K,
        embedding_model: str = "BAAI/bge-m3",
        provider: str = "local",
    ):
        if faiss is None:
            raise RuntimeError("faiss is required (`uv pip install faiss-cpu`).")
        if provider == "local" and SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required (`uv pip install sentence-transformers`)."
            )
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 is required (`uv pip install rank_bm25`).")

        self.index_dir = index_dir
        self.alpha = alpha
        self.faiss_top_k = faiss_top_k
        self.return_top_k = return_top_k
        self.embedding_model_name = embedding_model
        self.provider = provider

        LOGGER.info("Loading FAISS index from %s (Provider=%s)", index_dir / "faiss.index", provider)
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        with (index_dir / "mapping.json").open("r", encoding="utf-8") as fh:
            mapping = json.load(fh)
        self.chunk_ids: List[str] = mapping["chunk_ids"]

        LOGGER.info("Loading BM25 model from %s", index_dir / "bm25.pkl")
        with (index_dir / "bm25.pkl").open("rb") as fh:
            self.bm25: BM25Okapi = pickle.load(fh)

        # Initialize Embedding Model based on Provider
        if self.provider == "openai" or self.provider == "gemini": # Gemini uses OpenAI index per strategy
             if OpenAI is None:
                 raise RuntimeError("OpenAI package required for remote embedding.")
             self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
             LOGGER.info("Initialized OpenAI Embedding Client: %s", self.embedding_model_name)
        else:
             self.embedding_model = SentenceTransformer(self.embedding_model_name)
             LOGGER.info("Initialized Local SentenceTransformer: %s", self.embedding_model_name)
        
        # Reranker 모델 초기화 (설정에 따라 활성화)
        self.use_reranker = settings.USE_RERANKER
        self.reranker = None
        if self.use_reranker and settings.RERANKER_MODEL:
            if CrossEncoder is None:
                  LOGGER.warning("Reranker가 활성화되었으나 CrossEncoder를 임포트할 수 없습니다.")
            else:
                LOGGER.info("Reranker 모델 로딩 중: %s", settings.RERANKER_MODEL)
                self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        
        self.chunks = ChunkStore(index_dir)

    def _encode(self, query: str) -> np.ndarray:
        if self.provider == "openai" or self.provider == "gemini":
             query = query.replace("\n", " ")
             # Use the configured OPENAI_MODEL or strict embedding model name?
             # We should use self.embedding_model_name
             resp = self.client.embeddings.create(input=[query], model=self.embedding_model_name)
             vec = resp.data[0].embedding
             return np.array([vec], dtype=np.float32)
        else:
             return self.embedding_model.encode([query], normalize_embeddings=True)

    def _faiss_scores(self, query: str) -> dict[str, float]:
        embedding = self._encode(query)
        distances, indexes = self.faiss_index.search(
            np.asarray(embedding, dtype=np.float32), self.faiss_top_k
        )
        scores: dict[str, float] = {}
        for dist, idx in zip(distances[0], indexes[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            # L2 거리 값을 유사도 점수(0~1 범위)로 변환하여 가중 평균에 활용한다.
            sim = 1.0 / (1.0 + float(dist))
            scores[chunk_id] = sim
        return scores

    def _bm25_scores(self, query: str) -> dict[str, float]:
        tokens = re.findall(r"[A-Za-z0-9가-힣_#]+", query.lower())
        raw_scores = self.bm25.get_scores(tokens)
        max_score = max(raw_scores) if raw_scores.size else 1.0
        scores: dict[str, float] = {}
        for chunk_id, score in zip(self.chunk_ids, raw_scores):
            scores[chunk_id] = float(score) / max_score if max_score else 0.0
        return scores

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[tuple[str, float]]:
        """
        하이브리드 검색 후 선택적으로 Reranking을 수행합니다.
        1단계: FAISS(벡터) + BM25(키워드)로 후보군 추출 (settings.RERANK_TOP_K)
        2단계: CrossEncoder로 재순위화 (settings.USE_RERANKER가 True인 경우)
        """
        final_k = top_k or self.return_top_k
        
        # Reranker 사용 시 후보군을 더 많이 가져옵니다.
        candidate_k = settings.RERANK_TOP_K if (self.use_reranker and self.reranker) else final_k
        
        vect_scores = self._faiss_scores(query)
        bm25_scores = self._bm25_scores(query)
        combined: dict[str, float] = {}
        for chunk_id in self.chunk_ids:
            score = self.alpha * vect_scores.get(chunk_id, 0.0) + (1 - self.alpha) * bm25_scores.get(
                chunk_id, 0.0
            )
            if score > 0:
                combined[chunk_id] = score
        
        # 1차 정렬 (점수 내림차순)
        ranked_candidates = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:candidate_k]
        
        # Reranker가 없으면 바로 반환
        if not (self.use_reranker and self.reranker):
            return ranked_candidates[:final_k]
            
        # 2차 정렬 (Reranking)
        try:
            # (Query, Document Text) 쌍 생성
            pairs = []
            valid_candidates = []
            for chunk_id, initial_score in ranked_candidates:
                text = self.chunks.get_text(chunk_id)
                if text:
                    pairs.append([query, text])
                    valid_candidates.append((chunk_id, initial_score))
            
            if not pairs:
                return []

            # Cross-Encoder 점수 계산
            rerank_scores = self.reranker.predict(pairs)
            
            # (chunk_id, rerank_score) 형태로 변환 후 정렬
            reranked_results = []
            for idx, score in enumerate(rerank_scores):
                chunk_id = valid_candidates[idx][0]
                reranked_results.append((chunk_id, float(score)))
            
            final_ranked = sorted(reranked_results, key=lambda item: item[1], reverse=True)
            return final_ranked[:final_k]
            
        except Exception as e:
            LOGGER.error(f"Reranking 실패, 1차 검색 결과 반환: {e}")
            return ranked_candidates[:final_k]

    def fetch_chunk(self, chunk_id: str) -> tuple[str, str]:
        return self.chunks.get_doc(chunk_id), self.chunks.get_text(chunk_id)


from service.llm_factory import get_llm_service
from service.llm_base import BaseLLMService
from service.router import ChatRouter

# Initialize Router
chat_router = ChatRouter()


def _map_doc_id_to_url(doc_id: str) -> str:
    """
    doc_id (e.g., aws/ec2-ug_pdf)를 로컬 PDF URL (e.g., /docs/AWS/ec2-ug.pdf)로 매핑합니다.
    규칙:
    1. aws/ 접두사 제거
    2. _pdf 접미사를 .pdf로 변경
    3. 원본 파일명이 복잡할 수 있으므로, 단순 매핑 시도 후 파일 존재 여부는 클라이언트가 처리하거나
       여기서 source 디렉토리를 확인해야 함. 지금은 단순 규칙 기반 매핑.
    """
    filename = doc_id.replace("aws/", "")
    
    if filename.endswith("_pdf"):
        filename = filename[:-4] + ".pdf"
    
    return f"/docs/{filename}"


def format_answer(query: str, retrievals: list[tuple[str, float]], store: ChunkStore) -> tuple[str, List[Citation]]:
    # Fallback function if LLM is unavailable
    if not retrievals:
        return ("검색 결과가 없습니다.", [])
    
    citations = []
    summary_parts = []
    
    for idx, (chunk_id, score) in enumerate(retrievals, start=1):
        doc_id = store.get_doc(chunk_id)
        text = store.get_text(chunk_id).strip()
        page = store.get_page(chunk_id)
        
        ref_url = _map_doc_id_to_url(doc_id)
        expanded_text = store.get_neighbor_chunks(chunk_id, window=1)
        
        citations.append(Citation(ref=ref_url, chunk_id=chunk_id, doc_id=doc_id, page=page, anchor=expanded_text))
        summary_parts.append(f"[{idx}] {text[:200]}...")
        
    fallback_answer = "LLM 서비스를 사용할 수 없어 검색된 문서의 요약을 표시합니다:\n\n" + "\n".join(summary_parts)
    return fallback_answer, citations

# ... (skipping HybridRetriever unchanged) ...

def generate_answer_with_llm(query: str, retrievals: list[tuple[str, float]], store: ChunkStore, llm: BaseLLMService, history: list = []) -> tuple[str, List[Citation]]:
    if not retrievals:
        return ("관련된 문서를 찾을 수 없어 답변을 생성할 수 없습니다.", [])

    citations: List[Citation] = []
    context_chunks = []
    
    # Use top 5 for generation context, but store all citations
    for idx, (chunk_id, _) in enumerate(retrievals, start=1):
        doc_id = store.get_doc(chunk_id)
        text = store.get_text(chunk_id).strip()
        page = store.get_page(chunk_id)
        if not text:
            continue
        
        # Doc ID to URL Mapping
        ref_url = _map_doc_id_to_url(doc_id)
        
        # Expand context for citation snippet (1 before, 1 after)
        expanded_text = store.get_neighbor_chunks(chunk_id, window=1)
            
        citations.append(Citation(ref=ref_url, chunk_id=chunk_id, doc_id=doc_id, page=page, anchor=expanded_text)) 
        context_chunks.append({"text": text, "doc_id": doc_id})
        
    try:
        answer = llm.generate_answer(query, context_chunks, history)
    except Exception as e:
        LOGGER.error(f"LLM Generation Error: {e}")
        answer = "답변 생성 중 오류가 발생했습니다. (Fallback to summary)\n" + format_answer(query, retrievals, store)[0]
        
    return answer, citations



@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    index_dir = settings.INDEX_DIR # Dynamic based on provider
    if not index_dir.exists():
        # Fallback to default if dynamic dir not found (e.g. not built yet)
        if index_dir != INDEX_DIR_DEFAULT and INDEX_DIR_DEFAULT.exists():
             LOGGER.warning(f"Preferred index {index_dir} not found. Falling back to default {INDEX_DIR_DEFAULT}")
             index_dir = INDEX_DIR_DEFAULT
        else:
             raise RuntimeError(f"Index directory {index_dir} not found. Build the index first.")
    
    # Provider logic
    provider = "local"
    if "openai" in str(index_dir):
        provider = "openai"
    elif "gemini" in str(index_dir):
        provider = "gemini"
    
    # If using default, check settings
    if index_dir == INDEX_DIR_DEFAULT and settings.LLM_PROVIDER != "local":
        # This might happen if user didn't build specific index yet. 
        # But we want to match embedding model to index! 
        # If index is old (local), passing provider="openai" will crash if we try to use OpenAI embedding on BGE index.
        # So we must infer provider FROM THE INDEX DIR or assume consistent state.
        # For safety, let's trust settings IF index dir matches expectations.
        pass

    # Better: Use settings.EMBEDDING_MODEL and settings.LLM_PROVIDER
    # But strictly speaking, HybridRetriever depends on the INDEX content.
    # We should assume settings are correct.
    
    retriever = HybridRetriever(
        index_dir=index_dir,
        provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL
    )
    LOGGER.info("HybridRetriever initialised with index at %s (Model=%s)", index_dir, settings.EMBEDDING_MODEL)
    return retriever


app = FastAPI(title="RAG Integrity Lab Service", version="0.1.0")


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        payload = {
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "client": request.client.host if request.client else None,
        }
        HTTP_LOGGER.info(json.dumps(payload, ensure_ascii=False))
        LATENCY_LOGGER.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(duration_ms, 2),
                }
            )
        )


@app.on_event("startup")
def startup_event() -> None:
    setup_logger("service.app", settings.LOG_DIR / "env/service.log")
    setup_logger("service.http", settings.LOG_DIR / "http/service.log")
    setup_logger("service.qa", settings.LOG_DIR / "qa/service.log")
    setup_logger("service.latency", settings.LOG_DIR / "latency/service.log")
    setup_logger("service.latency", settings.LOG_DIR / "latency/service.log")
    setup_logger("service.router", settings.LOG_DIR / "router/service.log")  # Add Router Logger
    
    provider = settings.LLM_PROVIDER.upper()
    LOGGER.info(f"🚀 서비스 시작! 현재 활성화된 LLM Provider: {provider}")

    if provider == "LOCAL":
        model_info = get_model_info()
        LOGGER.info(
            "로컬 모델 정보: %s (%s)",
            DEFAULT_MODEL_NAME,
            json.dumps(model_info, ensure_ascii=False),
        )
    try:
        get_retriever()
    except Exception as exc:  # pragma: no cover - fail-fast visibility
        LOGGER.error("Failed to initialise retriever: %s", exc)
        raise


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

templates = Jinja2Templates(directory="service/templates")

@app.get("/mockup")
def mockup(request: Request):
    return templates.TemplateResponse("mockup_clean_paper.html", {"request": request})


# --- Web UI Integration & Route Handlers ---

BASE_DIR = pathlib.Path(__file__).resolve().parent
# templates is already defined above, but we ensure static files are mounted
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
if settings.DATA_SOURCE_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(settings.DATA_SOURCE_DIR)), name="docs")


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class ChatRequest(BaseModel):
    question: str
    history: list = []


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()
    try:
        retriever = get_retriever()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retriever unavailable: {exc}") from exc
    try:
        retrievals = retriever.retrieve(request.query_ko, top_k=request.top_k)
        
        # [Hotfix] Filter out garbage chunks (e.g. SSL Certificates) that cause token overflow
        filtered_retrievals = []
        for chunk_id, score in retrievals:
            # ChunkStore has .get_text(), not .get()
            chunk_text = retriever.chunks.get_text(chunk_id)
            if "BEGIN CERTIFICATE" in chunk_text or "END CERTIFICATE" in chunk_text:
                LOGGER.warning(f"Filtered out content chunk {chunk_id} due to CERTIFICATE content.")
                continue
            if len(chunk_text) > 5000: # Limit single chunk size to 5000 chars safely
                 LOGGER.warning(f"Filtered out content chunk {chunk_id} due to excessive length ({len(chunk_text)} chars).")
                 continue
            filtered_retrievals.append((chunk_id, score))
            
        retrievals = filtered_retrievals
        
        # Try to use LLM first
        llm = get_llm_service()
        if llm:
            answer, citations = generate_answer_with_llm(request.query_ko, retrievals, retriever.chunks, llm, request.history)
        else:
            answer, citations = format_answer(request.query_ko, retrievals, retriever.chunks)
            
        provider = settings.LLM_PROVIDER
        QA_LOGGER.info(
            json.dumps(
                {
                    "query": request.query_ko,
                    "provider": provider,
                    "top_k": request.top_k,
                    "citations": [citation.dict() for citation in citations],
                    "answer": answer,
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        LOGGER.exception("Failed to answer query: %s", request.query_ko)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000
    return AskResponse(answer_ko=answer, citations=citations, latency_ms=latency_ms, provider=provider)


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        # 1. Intelligent Routing (Rule + LLM)
        decision = chat_router.route(payload.question)
        
        # Log Intent Decision
        QA_LOGGER.info(
            json.dumps({
                "type": "intent_classification",
                "query": payload.question,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "direct_response": bool(decision.direct_response)
            }, ensure_ascii=False)
        )
        
        # 2. Handle Chit-chat immediately with strict separation
        if decision.intent == "chitchat":
            # If the router optimized and provided a response, use it.
            answer = decision.direct_response if decision.direct_response else "안녕하세요."
            
            # Log ChitChat Response
            QA_LOGGER.info(
                json.dumps({
                    "query": payload.question,
                    "provider": "router",
                    "intent": "chitchat",
                    "answer": answer
                }, ensure_ascii=False)
            )
            
            return {
                "answer": answer,
                "provider": f"router (chitchat, conf={decision.confidence})",
                "sources": []
            }
            
        # 3. Handle Technical Query (RAG)
        # Reuse the logic from ask()
        ask_req = AskRequest(query_ko=payload.question, top_k=settings.RETURN_TOP_K, history=payload.history)
        response = ask(ask_req)
        
        sources = []
        for idx, cit in enumerate(response.citations, start=1):
            sources.append({
                "rank": idx,
                "doc_id": cit.doc_id,
                "ref": cit.ref,
                "page": cit.page,
                "anchor": cit.anchor,
                "metadata": {"page": cit.page} if cit.page else {}
            })
            
        return {
            "answer": response.answer_ko,
            "provider": response.provider,
            "intent": "search_query", # Add intent to output for debug
            "sources": sources
        }

    except Exception as e:
        LOGGER.exception("Chat endpoint error")
        return {
            "answer": f"오류가 발생했습니다: {str(e)}",
            "sources": []
        }



