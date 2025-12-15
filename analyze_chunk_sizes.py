
import json
import pathlib
import glob
import sys
import tiktoken

def analyze_chunks(chunk_dir):
    chunk_files = glob.glob(str(pathlib.Path(chunk_dir) / "*.jsonl"))
    print(f"Analyzing {len(chunk_files)} chunk files in {chunk_dir}...")
    
    max_len = 0
    max_chunk_id = None
    max_file = None
    
    # OpenAI tokenizer
    enc = tiktoken.get_encoding("cl100k_base")

    oversized_count = 0
    total_count = 0

    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    chunk_id = data.get("chunk_id")
                    
                    # Approximate token count (or exact if fast enough)
                    # For speed, let's just use tiktoken since we need to know for sure
                    token_count = len(enc.encode(text))
                    total_count += 1
                    
                    if token_count > max_len:
                        max_len = token_count
                        max_chunk_id = chunk_id
                        max_file = cf
                        
                    if token_count > 8191:
                        print(f"WARNING: Oversized Chunk! ID={chunk_id}, Tokens={token_count}, File={cf}")
                        oversized_count += 1
                        
                except Exception as e:
                    print(f"Error parsing line in {cf}: {e}")

    print("-" * 30)
    print(f"Total Chunks: {total_count}")
    print(f"Max Token Count: {max_len}")
    print(f"Max Chunk ID: {max_chunk_id}")
    print(f"Max Chunk File: {max_file}")
    print(f"Oversized (>8191 tokens) Count: {oversized_count}")

if __name__ == "__main__":
    analyze_chunks("data/working/chunks")
