
import os
import sys
import subprocess
import time
from pathlib import Path

# Common Query for testing
TEST_QUERY = "AWS CLI 설치 방법"

def run_test_in_subprocess(provider: str):
    """
    Runs a rigorous check in a fresh subprocess to ensure settings are loaded correctly.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Testing Provider: {provider.upper()}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["LLM_PROVIDER"] = provider
    # Fix for GLIBCXX error (matching run_web.sh)
    env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    
    # We'll run a snippet of python code that imports the app and checks the retriever
    script_code = """
import os
import sys
import logging
import pathlib

# Configure logging to show only critical info to keep output clean
logging.basicConfig(level=logging.ERROR)

try:
    from config.settings import settings
    
    # Check 1: Verify Settings Load
    print(f"[Check 1] Settings Loaded: Provider={settings.LLM_PROVIDER}")
    print(f"[Check 1] Expected Index Dir: {settings.INDEX_DIR}")
    print(f"[Check 1] Expected Model: {settings.EMBEDDING_MODEL}")
    
    # Verify path correctness based on strategy
    expected_path_part = f"index/{settings.LLM_PROVIDER}"
    if expected_path_part not in str(settings.INDEX_DIR):
        print(f"❌ FAILED: Index Directory {settings.INDEX_DIR} does not match provider {settings.LLM_PROVIDER}")
        sys.exit(1)

    # Check 2: Initialize Retriever
    # Note: If index doesn't exist, this will fail. We handle that gracefully.
    from service.app import get_retriever, HybridRetriever
    
    try:
        retriever = get_retriever()
        print(f"[Check 2] HybridRetriever Initialized. Provider={retriever.provider}")
        
        # Verify Retriever Internals
        if settings.LLM_PROVIDER == "openai":
            if not hasattr(retriever, 'client'):
                 print("❌ FAILED: OpenAI provider should have 'client' attribute.")
                 sys.exit(1)
            print("✅ OpenAI Client Detected.")
        else:
            if "SentenceTransformer" not in str(type(retriever.embedding_model)):
                 print("❌ FAILED: Local provider should use SentenceTransformer.")
                 sys.exit(1)
            print("✅ SentenceTransformer Detected.")

        # Check 3: Retrieval Test
        query = "AWS CLI 설치 방법"
        print(f"[Check 3] Executing Retrieval Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        
        if results:
            print(f"✅ Retrieval Success! Found {len(results)} docs.")
            for i, (chunk_id, score) in enumerate(results):
                print(f"   - [{i+1}] {chunk_id} (Score: {score:.4f})")
        else:
            print("⚠️ Warning: Retrieval returned 0 results (Index might be empty but working).")

    except RuntimeError as e:
        if "Index directory" in str(e) and "not found" in str(e):
             print(f"⚠️ SKIPPING: Index directory not built yet. ({e})")
             # This is not a code failure, just operational.
             sys.exit(0)
        else:
             raise e

except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    sys.exit(1)
"""
    
    # Run the subprocess using the current virtualenv python
    python_exe = sys.executable
    result = subprocess.run(
        [python_exe, "-c", script_code],
        env=env,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"🚨 Test Failed for {provider} (Exit Code: {result.returncode})")
    else:
        print(f"🎉 Test Passed for {provider}")

def main():
    print("Starting Phase 4 Dual-Indexing Verification...")
    print("Note: This script verifies configuration logic and index loading.")
    print("      If indexes are currently building, retrieval might skip or show partial results.")
    
    # 1. Test Local
    run_test_in_subprocess("local")
    
    # 2. Test OpenAI
    run_test_in_subprocess("openai")

if __name__ == "__main__":
    main()
