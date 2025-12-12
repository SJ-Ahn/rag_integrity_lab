import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from service.router import ChatRouter

def test_router():
    router = ChatRouter()
    
    test_cases = [
        ("안녕하세요", "chitchat"),
        ("안녕", "chitchat"),
        ("누구니?", "chitchat"),
        ("자기소개 좀 해봐", "chitchat"),
        ("EC2 생성 방법 알려줘", "search_query"),
        ("RDS 모니터링은 어떻게 해?", "search_query"),
        ("고마워", "chitchat"),
        ("game server hosting", "search_query"), # English query
        ("hi", "chitchat"),
        # New cases for Smart Routing fallback (will default to search_query if LLM not mocked, 
        # but at least check it doesn't crash)
        ("기분 좋아", "search_query"), # Rule-based might miss this, defaults to search
    ]
    
    print("=== Router Logic Verification ===")
    all_passed = True
    for query, expected in test_cases:
        result = router.route(query) 
        intent = result.intent
        status = "PASSED" if intent == expected else "FAILED"
        print(f"[{status}] Query: '{query}' -> Intent: '{intent}' (Expected: '{expected}')")
        if intent != expected:
            all_passed = False

    if all_passed:
        print("\nSUCCESS: All router test cases passed!")
    else:
        print("\nFAILURE: Some test cases failed.")

if __name__ == "__main__":
    test_router()
