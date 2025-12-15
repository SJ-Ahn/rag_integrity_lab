import requests
import json
import sys

BASE_URL = "http://localhost:8889"

def test_api():
    print("=== API Integration Verification ===")
    
    # Test 1: Chitchat
    payload_chitchat = {"question": "안녕하세요", "history": []}
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload_chitchat)
        res.raise_for_status()
        data = res.json()
        print(f"[Chitchat] Provider: {data.get('provider')}")
        if data.get("provider") == "router (chitchat)":
            print("Message:", data.get("answer"))
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL (Expected 'router (chitchat)')")
            
    except Exception as e:
        print(f"RESULT: FAIL (Error: {e})")

    print("-" * 30)

    # Test 2: Technical
    payload_tech = {"question": "EC2 인스턴스 어떻게 생성해?", "history": []}
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload_tech)
        res.raise_for_status()
        data = res.json()
        provider = data.get('provider')
        print(f"[Technical] Provider: {provider}")
        
        # Provider should be 'local', 'openai', or 'gemini' (not router)
        if provider != "router (chitchat)":
            print(f"RESULT: PASS (Provider is {provider})")
        else:
            print("RESULT: FAIL (Unexpectedly matched as chitchat)")
            
    except Exception as e:
        print(f"RESULT: FAIL (Error: {e})")

if __name__ == "__main__":
    test_api()
