
import sys
import pathlib
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Mock llama_cpp before importing app
module_mock = MagicMock()
sys.modules["llama_cpp"] = module_mock

from service.app import LLMService

class TestLLMService:
    @patch("service.app.Llama")
    def test_llm_service_initialization(self, mock_llama):
        """Test if LLMService initializes Llama correctly"""
        # Create a dummy model file
        dummy_model_path = ROOT_DIR / "dummy.gguf"
        dummy_model_path.touch()
        
        try:
            service = LLMService(str(dummy_model_path))
            mock_llama.assert_called_once()
            assert service.llm is not None
        finally:
            dummy_model_path.unlink()

    def test_generate_answer_prompt(self):
        """Test if prompt is built correctly"""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test Answer"}}]
        }
        
        # Manually inject mock
        service = LLMService.__new__(LLMService)
        service.llm = mock_llm
        
        query = "EC2란 무엇인가?"
        chunks = [{"text": "EC2는 가상 서버입니다.", "doc_id": "doc1"}]
        
        answer = service.generate_answer(query, chunks)
        
        assert answer == "Test Answer"
        
        # Verify prompt structure
        args, kwargs = mock_llm.create_chat_completion.call_args
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "AWS EC2" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "EC2는 가상 서버입니다" in messages[1]["content"]
