#!/usr/bin/env python3
"""
run_eval.py
~~~~~~~~~~~

사전 정의한 골든셋을 기반으로 하이브리드 인덱스의 검색 성능을 정량 평가합니다.
Citation 정확도, Recall@K, 지연 시간(P95)을 계산해 docs/tasks/60_evaluation.md
요건을 만족하는지 확인합니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None  # type: ignore

from tools.logger import setup_logger
from service.app import HybridRetriever
from config.settings import settings
from config.model_registry import get_default_model_name, list_available_models


LOGGER = logging.getLogger("evaluation")
QA_LOGGER = logging.getLogger("evaluation.qa")
LATENCY_LOGGER = logging.getLogger("evaluation.latency")
SUMMARY_LOGGER = logging.getLogger("evaluation.summary")


DEFAULT_RESULTS_PATH = "evaluation/results/summary.json"

DEFAULT_CFG = {
    "dataset": str(settings.EVAL_DATASET_PATH),
    "index_dir": str(settings.INDEX_DIR),
    "top_k": settings.RETURN_TOP_K,
    "alpha": settings.ALPHA,
    "faiss_top_k": settings.FAISS_TOP_K,
    "model": get_default_model_name(),
}


@dataclass
class EvalSample:
    query: str
    expected_doc_ids: list[str]
    expected_chunk_ids: list[str]
    answer_summary: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "EvalSample":
        return cls(
            query=str(payload["query"]),
            expected_doc_ids=[str(x) for x in payload.get("expected_doc_ids", [])],
            expected_chunk_ids=[str(x) for x in payload.get("expected_chunk_ids", [])],
            answer_summary=str(payload.get("answer_summary", "")),
        )


def load_config(path: Optional[str]) -> dict[str, object]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CFG))
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {cfg_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required for evaluation configs (`uv pip install pyyaml`).")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    cfg.update(data)
    return cfg


def load_dataset(path: pathlib.Path) -> list[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(
            f"골든셋 파일을 찾을 수 없습니다: {path}. 평가 전에 데이터셋을 준비하세요."
        )
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(EvalSample.from_json(payload))
    LOGGER.info("Loaded %d evaluation samples from %s", len(samples), path)
    return samples


def recall_at_k(expected: list[str], retrieved: list[str]) -> float:
    if not expected:
        # 기대 목록이 비어 있으면 해당 질의는 이미 충족되었다고 간주한다.
        return 1.0
    hits = sum(1 for doc_id in set(retrieved) if doc_id in expected)
    return hits / len(expected)


def citation_accuracy(expected_chunks: list[str], retrieved_chunks: list[str]) -> float:
    if not expected_chunks:
        # 기대 청크가 없으면 인용 누락을 판단할 수 없으므로 정답으로 처리한다.
        return 1.0
    matched = sum(1 for chunk in retrieved_chunks if chunk in expected_chunks)
    return matched / len(expected_chunks)


def evaluate(cfg: dict[str, object]) -> dict[str, float]:
    index_dir = pathlib.Path(str(cfg["index_dir"])).expanduser().resolve()
    dataset_path = pathlib.Path(str(cfg["dataset"])).expanduser().resolve()
    top_k = int(cfg.get("top_k", settings.RETURN_TOP_K))
    model_name = str(cfg.get("model", get_default_model_name()))

    samples = load_dataset(dataset_path)

    retriever = HybridRetriever(
        index_dir=index_dir,
        alpha=float(cfg.get("alpha", 0.65)),
        faiss_top_k=int(cfg.get("faiss_top_k", 50)),
        return_top_k=top_k,
    )
    store = retriever.chunks

    recall_scores = []
    citation_scores = []
    latencies = []

    for sample in samples:
        start = time.perf_counter()
        retrieved_pairs = retriever.retrieve(sample.query, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        retrieved_chunk_ids = [chunk_id for chunk_id, _ in retrieved_pairs]
        retrieved_doc_ids = [store.get_doc(chunk_id) for chunk_id in retrieved_chunk_ids]
        recall_scores.append(recall_at_k(sample.expected_doc_ids, retrieved_doc_ids))
        citation_scores.append(citation_accuracy(sample.expected_chunk_ids, retrieved_chunk_ids))
        QA_LOGGER.info(
            json.dumps(
                {
                    "query": sample.query,
                    "expected_doc_ids": sample.expected_doc_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "expected_chunk_ids": sample.expected_chunk_ids,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "answer_summary": sample.answer_summary,
                },
                ensure_ascii=False,
            )
        )
        LATENCY_LOGGER.info(
            json.dumps(
                {
                    "query": sample.query,
                    "latency_ms": round(latency_ms, 2),
                    "top_k": top_k,
                }
            )
        )

    result = {
        "Recall@{k}".format(k=top_k): round(statistics.mean(recall_scores) * 100, 2),
        "CitationAccuracy": round(statistics.mean(citation_scores) * 100, 2),
        "LatencyP95ms": round(np.percentile(latencies, 95), 2) if latencies else 0.0,  # type: ignore[name-defined]
        "Model": model_name,
    }
    # 상위에서 numpy를 조건부로 임포트하므로, 퍼센타일 계산에 활용하는 것이 안전함을 명시한다.
    return result


def persist_results(results: dict[str, float], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Evaluation results saved to %s", output_path)
    SUMMARY_LOGGER.info(json.dumps(results, ensure_ascii=False))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retriever quality.")
    parser.add_argument("--cfg", help="Evaluation YAML config path.")
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH, help="Where to save metrics.")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity.")
    parser.add_argument(
        "--model",
        choices=list(list_available_models().keys()),
        help="평가에 사용할 LLM 모델 식별자 (config/models.yaml 참조).",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    setup_logger("evaluation", settings.LOG_DIR / "qa/evaluation_summary.log", lvl)
    setup_logger("evaluation.qa", settings.LOG_DIR / "qa/evaluation_samples.log", lvl)
    setup_logger("evaluation.latency", settings.LOG_DIR / "latency/evaluation.log", lvl)
    setup_logger("evaluation.summary", settings.LOG_DIR / "eval_summary.log", lvl)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    cfg = load_config(args.cfg)
    if args.model:
        cfg["model"] = args.model
    try:
        results = evaluate(cfg)
        results_path = pathlib.Path(args.results)
        model_slug = str(cfg.get("model", get_default_model_name()))
        if args.results == DEFAULT_RESULTS_PATH:
            results_dir = results_path.parent / model_slug
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / results_path.name
        persist_results(results, results_path)
    except Exception:
        LOGGER.exception("Evaluation failed.")
        return 1
    return 0


if __name__ == "__main__":
    import numpy as np  # 평가 단계에서만 필요한 의존성이므로 지연 임포트한다.

    raise SystemExit(main())
