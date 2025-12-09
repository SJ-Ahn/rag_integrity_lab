import sys
import pathlib
import pytest
from unittest.mock import MagicMock

# Add project root to sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from service.app import format_answer, ChunkStore, AskRequest

def test_ask_request_validation():
    with pytest.raises(ValueError):
        AskRequest(query_ko="   ")
    
    req = AskRequest(query_ko=" test ")
    assert req.query_ko == "test"

def test_format_answer_no_retrievals():
    store = MagicMock(spec=ChunkStore)
    answer, citations = format_answer("query", [], store)
    assert "충분한 근거를 찾지 못했습니다" in answer
    assert citations == []

def test_format_answer_with_retrievals():
    store = MagicMock(spec=ChunkStore)
    store.get_doc.return_value = "doc1"
    store.get_text.return_value = "This is a test answer. It has multiple sentences."
    
    retrievals = [("chunk1", 0.9)]
    answer, citations = format_answer("query", retrievals, store)
    
    assert "요약된 근거입니다" in answer
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk1"
    assert citations[0].doc_id == "doc1"
