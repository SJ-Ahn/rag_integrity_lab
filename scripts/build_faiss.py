#!/usr/bin/env python3
"""
build_faiss.py
~~~~~~~~~~~~~~

사전 생성된 청크 데이터를 이용해 벡터(FAISS)와 어휘(BM25)를 결합한 하이브리드 인덱스를 구축합니다.
기본 임베딩 모델은 BAAI/bge-m3이며, 긴 꼬리 질의 대응을 위해 BM25를 함께 저장합니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import numpy as np
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
import tiktoken
import time

load_dotenv()
import pickle
import platform
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tools.logging_utils import ensure_stream_logging, setup_file_logger

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    BM25Okapi = None  # type: ignore

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None  # type: ignore

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore

try:
    import nvidia_smi  # type: ignore
except ImportError:  # pragma: no cover
    nvidia_smi = None  # type: ignore


LOGGER = logging.getLogger("build_faiss")


DEFAULT_CFG = {
    "chunk_glob": "data/working/chunks/*.jsonl",
    "index_dir": "data/index",
    "embedding_model": "BAAI/bge-m3",
    "batch_size": 8,
    "faiss": {"m": 32, "ef_search": 96, "ef_construction": 200},
    "bm25": {"k1": 1.6, "b": 0.75},
}


class ResourceTracker:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._gpu_initialized = False

    def start(self) -> None:
        self._start_time = time.perf_counter()
        if nvidia_smi is not None:
            try:
                nvidia_smi.nvmlInit()
                self._gpu_initialized = True
            except Exception:  # pragma: no cover - defensive
                LOGGER.warning("NVML 초기화에 실패했습니다. GPU 모니터링을 비활성화합니다.", exc_info=True)
                self._gpu_initialized = False
        elif pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpu_initialized = True
            except Exception:  # pragma: no cover - defensive
                LOGGER.warning("NVML 초기화에 실패했습니다. GPU 모니터링을 비활성화합니다.", exc_info=True)
                self._gpu_initialized = False
        if psutil is None and not self._gpu_initialized:
            LOGGER.debug("psutil 또는 NVML을 찾을 수 없어 리소스 샘플링을 건너뜁니다.")
            return
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def log_initial_info(self, cfg: dict[str, object]) -> None:
        hw_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_model": platform.processor(),
        }
        if psutil is not None:
            hw_info.update(
                {
                    "cpu_physical": psutil.cpu_count(logical=False),
                    "cpu_logical": psutil.cpu_count(logical=True),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                }
            )
        gpu_info = self._gpu_devices()
        if gpu_info:
            hw_info["gpus"] = gpu_info
        LOGGER.info("인덱스 구축 환경 정보: %s", json.dumps(hw_info, ensure_ascii=False))

        job_info = {
            "chunk_glob": cfg.get("chunk_glob"),
            "index_dir": cfg.get("index_dir"),
            "embedding_model": cfg.get("embedding_model"),
            "batch_size": cfg.get("batch_size"),
            "faiss": cfg.get("faiss"),
            "bm25": cfg.get("bm25"),
        }
        LOGGER.info("인덱스 구축 작업 정보: %s", json.dumps(job_info, ensure_ascii=False))

    def _gpu_devices(self) -> Optional[list[str]]:
        if not self._gpu_initialized or (nvidia_smi is None and pynvml is None):
            return None
        try:
            nvml = nvidia_smi or pynvml
            count = nvml.nvmlDeviceGetCount()
            devices = []
            for idx in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(idx)
                raw_name = nvml.nvmlDeviceGetName(handle)
                name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                memory = nvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
                devices.append(f"{name} ({memory:.1f} GB)")
            return devices
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning("GPU 정보를 조회하지 못했습니다.", exc_info=True)
            return None

    def _sample_loop(self) -> None:
        while self._running:
            sample = {
                "cpu": self._cpu_percent(),
                "mem": self._memory_percent(),
                "gpu": self._gpu_percent(),
                "vram": self._vram_percent(),
            }
            with self._lock:
                self.samples.append(sample)
            time.sleep(1.0)

    def _cpu_percent(self) -> Optional[float]:
        if psutil is None:
            return None
        return psutil.cpu_percent(interval=None)

    def _memory_percent(self) -> Optional[float]:
        if psutil is None:
            return None
        return psutil.virtual_memory().percent

    def _gpu_percent(self) -> Optional[float]:
        if not self._gpu_initialized or (nvidia_smi is None and pynvml is None):
            return None
        try:
            nvml = nvidia_smi or pynvml
            count = nvml.nvmlDeviceGetCount()
            if count == 0:
                return None
            total = 0.0
            for idx in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(idx)
                util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                total += util
            return total / count
        except Exception:  # pragma: no cover
            LOGGER.debug("GPU 활용도 측정 실패", exc_info=True)
            return None

    def _vram_percent(self) -> Optional[float]:
        if not self._gpu_initialized or (nvidia_smi is None and pynvml is None):
            return None
        try:
            nvml = nvidia_smi or pynvml
            count = nvml.nvmlDeviceGetCount()
            if count == 0:
                return None
            total = 0.0
            for idx in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(idx)
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                total += (memory.used / memory.total) * 100
            return total / count
        except Exception:  # pragma: no cover
            LOGGER.debug("VRAM 활용도 측정 실패", exc_info=True)
            return None

    def stop(self) -> dict[str, Optional[float]]:
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join()
        if self._gpu_initialized:
            try:
                if nvidia_smi is not None:
                    nvidia_smi.nvmlShutdown()
                elif pynvml is not None:
                    pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                LOGGER.debug("NVML 종료 중 오류", exc_info=True)
        self._end_time = time.perf_counter()
        return self.report()

    def report(self) -> dict[str, Optional[float]]:
        duration = None
        if self._start_time is not None:
            end_time = self._end_time or time.perf_counter()
            duration = end_time - self._start_time

        def _avg_peak(key: str) -> tuple[Optional[float], Optional[float]]:
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            if not values:
                return None, None
            return statistics.mean(values), max(values)

        cpu_avg, cpu_peak = _avg_peak("cpu")
        mem_avg, mem_peak = _avg_peak("mem")
        gpu_avg, gpu_peak = _avg_peak("gpu")
        vram_avg, vram_peak = _avg_peak("vram")

        metrics = {
            "total_duration_sec": round(duration, 2) if duration is not None else None,
            "cpu_avg_percent": round(cpu_avg, 2) if cpu_avg is not None else None,
            "cpu_peak_percent": round(cpu_peak, 2) if cpu_peak is not None else None,
            "mem_avg_percent": round(mem_avg, 2) if mem_avg is not None else None,
            "mem_peak_percent": round(mem_peak, 2) if mem_peak is not None else None,
            "gpu_avg_percent": round(gpu_avg, 2) if gpu_avg is not None else None,
            "gpu_peak_percent": round(gpu_peak, 2) if gpu_peak is not None else None,
            "vram_avg_percent": round(vram_avg, 2) if vram_avg is not None else None,
            "vram_peak_percent": round(vram_peak, 2) if vram_peak is not None else None,
            "samples": len(self.samples),
        }
        return metrics

@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ChunkRecord":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload.get("doc_id", "")),
            text=str(payload.get("text", "")),
            page=payload.get("page") if payload.get("page") else None,
        )


# ... (skip load_config in diff, focusing on ChunkRecord above and persist_index below)

def persist_index(
    index_dir: pathlib.Path,
    faiss_index,
    bm25_model: Optional[BM25Okapi],
    chunks: list[ChunkRecord],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "doc_ids": [chunk.doc_id for chunk in chunks],
    }
    (index_dir / "mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "page": chunk.page,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_config(path: Optional[str]) -> dict[str, object]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CFG))  # 중첩된 딕셔너리를 복사하기 위한 안전한 깊은 복사
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"인덱스 구성 파일을 찾을 수 없습니다: {cfg_path}")
    if cfg_path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("구성 파일은 YAML 형식이어야 합니다.")
    if yaml is None:
        raise RuntimeError("YAML을 읽기 위해서는 PyYAML이 필요합니다. `uv pip install pyyaml` 실행")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    merged = json.loads(json.dumps(DEFAULT_CFG))
    merged.update(data)
    return merged


def load_chunks(glob_pattern: str) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for path in sorted(pathlib.Path().glob(glob_pattern)):
        LOGGER.debug("청크 파일 로딩: %s", path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                records.append(ChunkRecord.from_json(raw))
    LOGGER.debug("총 로딩된 청크 수: %d", len(records))
    LOGGER.info("총 %d개의 청크를 읽었습니다: %s", len(records), glob_pattern)
    return records


def simple_tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9가-힣_#]+", text.lower())
    return words


def build_bm25(chunks: list[ChunkRecord], k1: float, b: float) -> BM25Okapi:
    if BM25Okapi is None:
        raise RuntimeError(
            "BM25 인덱스를 만들려면 rank_bm25 패키지가 필요합니다. `uv pip install rank_bm25`"
        )
    LOGGER.debug("BM25 토큰화 시작 (청크 수=%d)", len(chunks))
    tokenized_corpus = [simple_tokenize(chunk.text) for chunk in chunks]
    LOGGER.debug("BM25 토큰화 완료. K1=%s, B=%s", k1, b)
    model = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    return model


def build_faiss_index(
    vectors,  # numpy.ndarray 형식의 임베딩 행렬
    dim: int,
    m: int,
    ef_construction: int,
    ef_search: int,
):
    if faiss is None:
        raise RuntimeError("FAISS 벡터 인덱스를 생성하려면 `faiss-cpu` 패키지가 필요합니다.")

    LOGGER.debug("FAISS 인덱스 생성: dim=%d, m=%d, ef_construction=%d, ef_search=%d", dim, m, ef_construction, ef_search)
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vectors)
    return index


def encode_chunks(
    model_name: str,
    chunks: list[ChunkRecord],
    batch_size: int,
    provider: str = "local",
):
    texts = [chunk.text for chunk in chunks]
    LOGGER.info("%d개의 청크를 임베딩합니다 (Provider=%s, Model=%s)", len(texts), provider, model_name)
    
    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("OpenAI 패키지가 필요합니다.")
        # Load API Key from env (managed by settings or os)
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        final_embeddings = []
        
        # Prepare Tokenizer for truncation
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except:
             enc = tiktoken.get_encoding("gpt2") # Fallback

        # Batch processing for OpenAI
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # Replace newlines with spaces for best results (recommended by OpenAI)
                batch = [t.replace("\n", " ") for t in batch]
                
                # Truncate to 8191 tokens max
                safe_batch = []
                for text in batch:
                    tokens = enc.encode(text)
                    if len(tokens) > 8191:
                        LOGGER.warning("Truncated oversized chunk (%d tokens) to 8191 tokens.", len(tokens))
                        tokens = tokens[:8191]
                        text = enc.decode(tokens)
                    safe_batch.append(text)
                
                
                # Retry loop with exponential backoff
                max_retries = 5
                wait_time = 1
                
                for attempt in range(max_retries):
                    try:
                        resp = client.embeddings.create(input=safe_batch, model=model_name)
                        break # Success
                    except RateLimitError as e:
                        if attempt == max_retries - 1:
                            LOGGER.error("OpenAI Rate Limit Exceeded after %d retries: %s", max_retries, e)
                            raise
                        LOGGER.warning("Rate limit reached. Waiting %d seconds before retry %d/%d...", wait_time, attempt + 1, max_retries)
                        time.sleep(wait_time)
                        wait_time *= 2 # Exponential backoff
                
                # Sort by index to ensure order
                data = sorted(resp.data, key=lambda x: x.index)
                vecs = [d.embedding for d in data]
                final_embeddings.extend(vecs)
                
                # Adaptive sleep based on batch size to respect TPM
                # 1M TPM / 60 sec = ~16k tokens/sec. 
                # If batch is ~8k tokens, we can do 2 req/sec.
                # Adding base latency.
                time.sleep(0.5) 
            except Exception as e:
                LOGGER.error("OpenAI Embedding 실패: %s", e)
                raise
        
        return np.array(final_embeddings, dtype=np.float32)

    elif provider == "local":
        if SentenceTransformer is None:
             raise RuntimeError(
                "임베딩 계산을 위해 sentence-transformers 설치가 필요합니다. `uv pip install sentence-transformers`"
            )
        model = SentenceTransformer(model_name)
        LOGGER.debug("임베딩 배치 크기: %d", batch_size)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        LOGGER.debug("임베딩 완료. shape=%s", getattr(embeddings, "shape", None))
        return embeddings
    
    else:
        raise ValueError(f"지원하지 않는 Provider입니다: {provider}")


def persist_index(
    index_dir: pathlib.Path,
    faiss_index,
    bm25_model: Optional[BM25Okapi],
    chunks: list[ChunkRecord],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "doc_ids": [chunk.doc_id for chunk in chunks],
    }
    (index_dir / "mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if faiss is None:
        raise RuntimeError("인덱스 저장 중 FAISS 모듈을 찾을 수 없습니다.")
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    LOGGER.info("FAISS 인덱스를 저장했습니다: %s", index_dir / "faiss.index")

    if bm25_model is not None:
        with (index_dir / "bm25.pkl").open("wb") as fh:
            pickle.dump(bm25_model, fh)
        LOGGER.info("BM25 모델을 저장했습니다: %s", index_dir / "bm25.pkl")


def run(cfg: dict[str, object]) -> None:
    tracker = ResourceTracker()
    tracker.start()
    tracker.log_initial_info(cfg)
    try:
        chunks = load_chunks(str(cfg["chunk_glob"]))
        if not chunks:
            raise RuntimeError("청크 파일을 찾을 수 없습니다. make_chunks.py를 먼저 실행하세요.")

        embeddings = encode_chunks(
            model_name=str(cfg["embedding_model"]),
            chunks=chunks,
            batch_size=int(cfg["batch_size"]),
            provider=str(cfg.get("provider", "local")),
        )

        dim = embeddings.shape[1]
        faiss_cfg = cfg.get("faiss", {})
        index = build_faiss_index(
            vectors=embeddings,
            dim=dim,
            m=int(faiss_cfg.get("m", 32)),
            ef_construction=int(faiss_cfg.get("ef_construction", 200)),
            ef_search=int(faiss_cfg.get("ef_search", 96)),
        )

        bm25_cfg = cfg.get("bm25", {})
        bm25_model = build_bm25(
            chunks,
            k1=float(bm25_cfg.get("k1", 1.6)),
            b=float(bm25_cfg.get("b", 0.75)),
        )

        index_dir = pathlib.Path(str(cfg["index_dir"])).expanduser().resolve()
        persist_index(index_dir, index, bm25_model, chunks)
        LOGGER.info("하이브리드 인덱스 구축이 완료되었습니다.")
    finally:
        metrics = tracker.stop()
        LOGGER.info("인덱스 구축 리소스 통계: %s", json.dumps(metrics, ensure_ascii=False))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAISS + BM25 하이브리드 인덱스를 생성합니다.")
    parser.add_argument("--cfg", help="인덱스 YAML 구성 파일 경로")
    parser.add_argument("--log-level", default="INFO", help="로깅 레벨 설정")
    parser.add_argument("--provider", help="임베딩 제공자 (local, openai, gemini)")
    parser.add_argument("--model", help="임베딩 모델 이름 (provider에 맞게 설정)")
    parser.add_argument("--index-dir", help="출력 인덱스 디렉토리")
    parser.add_argument("--batch-size", type=int, help="배치 크기")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    ensure_stream_logging(level)
    setup_file_logger(LOGGER, pathlib.Path("logs/env/index_build.log"), level)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    cfg = load_config(args.cfg)
    
    # CLI Overrides
    if args.provider:
        cfg["provider"] = args.provider
    if args.model:
        cfg["embedding_model"] = args.model
    if args.index_dir:
        cfg["index_dir"] = args.index_dir
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    
    # Auto-configure index dir if utilizing Strategy 2 conventions
    if args.provider and not args.index_dir and cfg["index_dir"] == "data/index":
         cfg["index_dir"] = f"data/index/{args.provider}"
    
    LOGGER.info("설정 확인: Provider=%s, Model=%s, Dir=%s", cfg.get("provider"), cfg.get("embedding_model"), cfg.get("index_dir"))

    try:
        run(cfg)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
