
import os
import sys
import logging
from config.settings import settings

# Force Local Provider
os.environ["LLM_PROVIDER"] = "local"
settings.LLM_PROVIDER = "local"
# Reload settings logic if needed, but __post_init__ ran on import? 
# We might need to manually set INDEX_DIR if settings instance is already created.
import pathlib
settings.INDEX_DIR = pathlib.Path("data/index/local")
settings.EMBEDDING_MODEL = "BAAI/bge-m3"

# Setup Logger
logging.basicConfig(level=logging.INFO)

try:
    from service.app import get_retriever, HybridRetriever
    
    print(f"Testing Local Mode...")
    print(f"Settings: Provider={settings.LLM_PROVIDER}, Index={settings.INDEX_DIR}")
    
    # Initialize Retriever
    retriever = get_retriever()
    
    # Verify Provider
    assert retriever.provider == "local"
    assert "SentenceTransformer" in str(type(retriever.embedding_model))
    print("✅ Retriever initialized with Local Provider and SentenceTransformer.")
    
    # Test Retrieval
    query = "EC2 인스턴스 타입"
    print(f"Query: {query}")
    results = retriever.retrieve(query, top_k=3)
    
    for chunk_id, score in results:
        print(f" - Found: {chunk_id} (Score: {score:.4f})")
    
    if not results:
        print("❌ No results found (Local index might be empty or broken).")
        sys.exit(1)
        
    print("✅ Local Retrieval Succeeded!")

except Exception as e:
    print(f"❌ Test Failed: {e}")
    sys.exit(1)
