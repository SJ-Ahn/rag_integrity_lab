import requests
import json
import sys

BASE_URL = "http://localhost:8889"

def test_query(payload, name):
    print(f"\n--- Testing: {name} ---")
    print(f"Query: {payload['question']}")
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            print(f"[SUCCESS] Status: 200")
            print(f"Provider: {data.get('provider')}")
            print(f"Answer: {data.get('answer')[:100]}...") # Truncate answer
            return True
        else:
            print(f"[FAILURE] Status: {res.status_code}")
            print(f"Error: {res.text}")
            return False
    except Exception as e:
        print(f"[FAILURE] Exception: {e}")
        return False

def run_tests():
    # 1. Chit-chat (Should be router)
    if not test_query({"question": "안녕", "history": []}, "Chit-chat"):
        sys.exit(1)

    # 2. General QA (Should be OpenAI)
    if not test_query({"question": "EC2 인스턴스 어떻게 생성해?", "history": []}, "General QA"):
        sys.exit(1)

    # 3. Problematic QA (Should be OpenAI, no 500/429)
    # This query "네가 교육이나 인증 프로그램 정보를 갖고 있나?" retrieved SSL certs previously.
    if not test_query({"question": "네가 교육이나 인증 프로그램 정보를 갖고 있나?", "history": []}, "Problematic QA"):
        sys.exit(1)

    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    run_tests()
