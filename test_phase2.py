import os
import sys
from config.settings import settings
from service.llm_factory import get_llm_service
from service.llm_external import OpenAILLMService, GeminiLLMService

def test_phase2():
    print("Testing OpenAI Service instantiation...")
    settings.LLM_PROVIDER = "openai"
    # Mock Key for instantiation test
    settings.OPENAI_API_KEY = "test-key"
    get_llm_service.cache_clear()
    
    service = get_llm_service()
    if isinstance(service, OpenAILLMService):
        print("SUCCESS: OpenAILLMService instantiated.")
    else:
        print(f"FAIL: Expected OpenAILLMService, got {type(service)}")
        sys.exit(1)

    print("Testing Gemini Service instantiation...")
    settings.LLM_PROVIDER = "gemini"
    # Mock Key for instantiation test
    settings.GEMINI_API_KEY = "test-key"
    get_llm_service.cache_clear()
    
    service = get_llm_service()
    if isinstance(service, GeminiLLMService):
        print("SUCCESS: GeminiLLMService instantiated.")
    else:
        print(f"FAIL: Expected GeminiLLMService, got {type(service)}")
        sys.exit(1)

    # Revert settings
    settings.LLM_PROVIDER = "local"
    get_llm_service.cache_clear()

if __name__ == "__main__":
    test_phase2()
