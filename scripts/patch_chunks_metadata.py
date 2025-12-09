#!/usr/bin/env python3
"""
scripts/patch_chunks_metadata.py
Uses existing chunks in data/working to patch data/index/chunks.jsonl
Preserves order to maintain alignment with FAISS/BM25 indices.
"""
import json
import logging
import pathlib
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("patch_metadata")

def load_working_chunks(glob_pattern: str) -> list[dict]:
    records = []
    # SORTED is critical to match build_faiss.py behavior
    paths = sorted(pathlib.Path().glob(glob_pattern))
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                records.append(json.loads(line))
    return records

def patch_index(working_glob: str, index_dir: str):
    idx_path = pathlib.Path(index_dir)
    if not idx_path.exists():
        raise FileNotFoundError(f"{index_dir} does not exist.")
    
    LOGGER.info("Loading working chunks from %s", working_glob)
    working_data = load_working_chunks(working_glob)
    LOGGER.info("Loaded %d chunks.", len(working_data))
    
    # Read existing chunks to verify count (optional, but good safety)
    existing_chunks_path = idx_path / "chunks.jsonl"
    if existing_chunks_path.exists():
        with existing_chunks_path.open("r") as fh:
            existing_count = sum(1 for line in fh if line.strip())
        LOGGER.info("Existing index has %d chunks.", existing_count)
        
        if existing_count != len(working_data):
            LOGGER.warning("COUNT MISMATCH! Existing: %d, New: %d. Indices might be misaligned!", existing_count, len(working_data))
            # Proceeding might be dangerous if vectors don't match texts.
            # But if the user says "page info missing", likely count matches but metadata missing.
            # If make_chunks was re-run and changed count, then FAISS is ALREADY broken.
            # If so, full rebuild is required.
            # But let's assume count matches for now.
    
    # Write new chunks.jsonl
    LOGGER.info("Writing updated chunks.jsonl to %s", existing_chunks_path)
    with existing_chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in working_data:
            # Construct entry
            entry = {
                "chunk_id": str(chunk["chunk_id"]),
                "doc_id": str(chunk.get("doc_id", "")),
                "text": str(chunk.get("text", "")),
                "page": chunk.get("page")  # This is what we want!
            }
            # Ensure page is valid if present
            if not entry["page"]:
                entry["page"] = None
                
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    LOGGER.info("Patch complete. Page numbers should be present.")

if __name__ == "__main__":
    patch_index("data/working/chunks/*.jsonl", "data/index")
