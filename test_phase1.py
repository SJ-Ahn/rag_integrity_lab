import sys
import logging
from service.llm_factory import get_llm_service
from service.llm_local import LocalLLMService

logging.basicConfig(level=logging.INFO)

def test_phase1():
    print("Testing get_llm_service()...")
    service = get_llm_service()
    
    if service is None:
        print("FAIL: Service returned None")
        sys.exit(1)
        
    if not isinstance(service, LocalLLMService):
        print(f"FAIL: Expected LocalLLMService, got {type(service)}")
        sys.exit(1)
        
    print("SUCCESS: LocalLLMService loaded successfully via Factory.")
    
if __name__ == "__main__":
    test_phase1()
