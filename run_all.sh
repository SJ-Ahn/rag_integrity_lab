python scripts/normalize_docs.py --in data/source --out data/working/normalized --meta-out data/working/meta
date
python scripts/make_chunks.py --cfg ingest/cfg/ingest_baseline.yaml
date
export OPENAI_API_KEY=${OPENAI_API_KEY}
python evaluation/generate_golden.py --count 30 --version v3 --log-level INFO
date
python scripts/build_faiss.py --cfg ingest/cfg/index_bge_m3.yaml
date
python evaluation/run_eval.py --cfg evaluation/cfg/eval_grid_v3.yaml --model llama3.1
date
