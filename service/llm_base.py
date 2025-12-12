from abc import ABC, abstractmethod
from typing import Optional, List

class BaseLLMService(ABC):
    @abstractmethod
    def generate_answer(self, query: str, context_chunks: list[dict], history: list = []) -> str:
        """
        Generates an answer based on the query and retrieved context chunks.
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        Generates raw text based on a prompt. Optional override for flexible usage.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("generate_text not implemented for this provider")
