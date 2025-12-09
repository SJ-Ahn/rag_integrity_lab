import sys
import pathlib
import pytest

# Add project root to sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.make_chunks import tokenize, detokenize, chunk_tokens, Chunk

def test_tokenize_detokenize():
    text = "Hello world.\n\nThis is a new paragraph."
    tokens = tokenize(text)
    assert "<PARA_BREAK>" in tokens
    reconstructed = detokenize(tokens)
    assert reconstructed == text

def test_chunk_tokens():
    tokens = ["word"] * 100
    cfg = {
        "chunk_size": 50,
        "chunk_overlap": 10,
        "min_chunk_size": 5
    }
    chunks = list(chunk_tokens("doc1", tokens, cfg))
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].doc_id == "doc1"
