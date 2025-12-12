import logging
from typing import Optional
from functools import lru_cache

from config.settings import settings
from service.llm_base import BaseLLMService
from service.llm_local import LocalLLMService
from service.llm_external import OpenAILLMService, GeminiLLMService

LOGGER = logging.getLogger("service.llm.factory")

@lru_cache(maxsize=1)
def get_llm_service() -> Optional[BaseLLMService]:
    provider = settings.LLM_PROVIDER.lower()
    
    try:
        if provider == "local":
            return LocalLLMService()
        elif provider == "openai":
            return OpenAILLMService()
        elif provider == "gemini":
            return GeminiLLMService()
        else:
            LOGGER.error(f"Unknown LLM provider: {provider}")
            return None
            
    except Exception as e:
        LOGGER.error(f"Failed to initialize LLM service ({provider}): {e}")
        return None
