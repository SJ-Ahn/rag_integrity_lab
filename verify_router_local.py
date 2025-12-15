
import os
import sys
import logging
from config.settings import settings

# Force Local Provider for this test
os.environ["LLM_PROVIDER"] = "local"
settings.LLM_PROVIDER = "local"
settings.INDEX_DIR = settings.INDEX_DIR.parent / "local" # Ensure consistent path if needed

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_router")

from service.router import ChatRouter

def test_router_local():
    print("🚀 Initializing ChatRouter with Provider: LOCAL")
    try:
        router = ChatRouter()
    except Exception as e:
        print(f"❌ Failed to initialize ChatRouter: {e}")
        return

    test_queries = [
        "너는 무엇을 할수 있니?",
        "네 소개를 해줘",
        "AWS EC2가 뭐야?",
        "안녕",
        "너는 무엇을 할수 있니?\nU"  # Noisy case from log
    ]

    print(f"\n🧪 Testing Queries...")
    for query in test_queries:
        print(f"\n[Query]: {query}")
        try:
            # We want to see if it hits Rule-based or LLM-based
            # The router logs internally, but we can verify the result.
            result = router.route(query)
            
            print(f"  -> Intent: {result.intent.upper()}")
            print(f"  -> Confidence: {result.confidence}")
            if result.direct_response:
                print(f"  -> Direct Response: {result.direct_response}")
            
            # assertions/checks
            if query in ["너는 무엇을 할수 있니?", "네 소개를 해줘", "안녕"]:
                if result.intent != "chitchat":
                    print("  ⚠️  WARNING: Expected 'chitchat' but got 'search_query'")
            elif query == "AWS EC2가 뭐야?":
                if result.intent != "search_query":
                     print("  ⚠️  WARNING: Expected 'search_query' but got 'chitchat'")

        except Exception as e:
            print(f"  ❌ Error processing query: {e}")

if __name__ == "__main__":
    test_router_local()
