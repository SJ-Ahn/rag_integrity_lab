#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Fix for GLIBCXX_3.4.32 not found (Miniconda vs System libstdc++)
# Force usage of system libstdc++ which has the newer GLIBCXX version
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Run Uvicorn server
echo "Starting RAG Web Interface at http://localhost:8889"
uvicorn service.app:app --host 0.0.0.0 --port 8889 --reload
