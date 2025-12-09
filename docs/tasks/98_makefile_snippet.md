# Makefile 예시

```make
.PHONY: prepare chunks index serve eval report
prepare:
\tpython scripts/normalize_docs.py --in data/source --out data/working/normalized
chunks:
\tpython scripts/make_chunks.py --cfg ingest/cfg/ingest_baseline.yaml
index:
\tpython scripts/build_faiss.py --cfg ingest/cfg/index_bge_m3.yaml
serve:
\tuvicorn service.app:app --port 8080
eval:
\tpython evaluation/run_eval.py --cfg evaluation/cfg/eval_grid.yaml
report:
\tpython evaluation/reporters/make_summary.py --in evaluation/results --out docs/results_summary.md
````

