import pathlib
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class Settings:
    # Indexing & Retrieval
    INDEX_DIR: pathlib.Path = pathlib.Path("data/index")
    ALPHA: float = 0.65  # Weight for vector search (1-ALPHA for BM25)
    FAISS_TOP_K: int = 50
    RETURN_TOP_K: int = 8
    
    # Reranking
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"  # Best performance/latency trade-off
    RERANK_TOP_K: int = 50  # Best Recall (93.33%)
    
    # Chunking
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150
    MIN_CHUNK_SIZE: int = 200
    
    # Paths
    DATA_SOURCE_DIR: pathlib.Path = pathlib.Path("data/source/AWS")
    DATA_WORKING_DIR: pathlib.Path = pathlib.Path("data/working")
    LOG_DIR: pathlib.Path = pathlib.Path("logs")
    
    # Evaluation
    EVAL_DATASET_PATH: pathlib.Path = pathlib.Path("evaluation/datasets/v3/golden_generated.jsonl")

    # LLM
    LLM_CONTEXT_WINDOW: int = 16384
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "local")  # "local", "openai", "gemini"
    
    # Local LLM Specific
    LOCAL_MODEL_PATH: str = "models/llama3.1/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    
    # External LLM Specific
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4-turbo-preview")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-pro")

settings = Settings()
