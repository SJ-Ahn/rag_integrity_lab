#!/usr/bin/env python3
"""
make_chunks.py
~~~~~~~~~~~~~~

정규화된 문서를 토큰 기준으로 슬라이딩 윈도우 분할하여 검색용 청크를 생성합니다.
chunk_size와 overlap을 설정할 수 있으며, 인용 추적을 위해 #[anchor] 표기를 유지합니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tools.logger import setup_logger
from config.settings import settings

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None  # type: ignore


LOGGER = logging.getLogger("make_chunks")


DEFAULT_CFG = {
    "input": str(settings.DATA_WORKING_DIR / "normalized/normalized.jsonl"),
    "output_dir": str(settings.DATA_WORKING_DIR / "chunks"),
    "chunk_size": settings.CHUNK_SIZE,
    "chunk_overlap": settings.CHUNK_OVERLAP,
    "min_chunk_size": settings.MIN_CHUNK_SIZE,
}


ANCHOR_PATTERN = re.compile(r"\[#([^\]]+)\]")


@dataclass
class NormalizedRecord:
    doc_id: str
    title: str
    text: str
    anchors: list[str]

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "NormalizedRecord":
        return cls(
            doc_id=str(raw["doc_id"]),
            title=str(raw.get("title", "")),
            text=str(raw.get("text", "")),
            anchors=list(raw.get("anchors", [])),
        )


PAGE_PATTERN = re.compile(r"\[\[PAGE:(\d+)\]\]")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    anchors: list[str]
    token_start: int
    token_end: int
    page: Optional[int] = None

    def to_json_line(self) -> str:
        payload = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "anchors": self.anchors,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "page": self.page,
        }
        return json.dumps(payload, ensure_ascii=False)


def load_config(path: Optional[str]) -> dict[str, object]:
    if path is None:
        return DEFAULT_CFG.copy()

    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"구성 파일을 찾을 수 없습니다: {cfg_path}")
    if cfg_path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("구성 파일은 YAML 형식이어야 합니다.")
    if yaml is None:
        raise RuntimeError("YAML 구성을 읽으려면 PyYAML이 필요합니다. `uv pip install pyyaml` 실행")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    merged = DEFAULT_CFG.copy()
    merged.update(data)
    return merged


def iter_normalized(path: pathlib.Path) -> Iterable[NormalizedRecord]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                LOGGER.debug("빈 라인 건너뜀: %s:%d", path, line_no)
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON on line %s: %s", line_no, path)
                continue
            yield NormalizedRecord.from_json(raw)


def tokenize(text: str) -> list[str]:
    # 단락 경계를 보존하기 위해 빈 줄(\n\n)을 특수 토큰으로 치환한 뒤 단어 단위 토큰을 만든다.
    tokens = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        words = paragraph.split()
        LOGGER.debug("단락 토큰화 (길이=%d): %.40s", len(words), paragraph)
        tokens.extend(words)
        tokens.append("<PARA_BREAK>")
    if tokens and tokens[-1] == "<PARA_BREAK>":
        LOGGER.debug("마지막 단락 구분자 제거")
        tokens.pop()
    return tokens


def detokenize(tokens: list[str]) -> str:
    parts = []
    paragraph = []
    for token in tokens:
        if token == "<PARA_BREAK>":
            parts.append(" ".join(paragraph))
            parts.append("")  # 단락 사이에 빈 줄을 삽입하여 원문 구조를 보존한다.
            paragraph = []
            continue
        paragraph.append(token)
    if paragraph:
        parts.append(" ".join(paragraph))
    # 인접한 빈 줄이 다수 발생하는 경우 2줄로 축약해 응답 생성 품질을 유지한다.
    text = "\n".join(parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def chunk_tokens(doc_id: str, tokens: list[str], cfg: dict[str, object]) -> Iterable[Chunk]:
    size = int(cfg["chunk_size"])
    overlap = int(cfg["chunk_overlap"])
    min_size = int(cfg["min_chunk_size"])
    if size <= overlap:
        raise ValueError("chunk_size는 chunk_overlap보다 커야 합니다.")

    step = size - overlap
    chunk_index = 0
    total = len(tokens)
    if total == 0:
        return []

    global_page_counter = 1

    for start in range(0, total, step):
        end = min(start + size, total)
        window = tokens[start:end]
        if len(window) < min_size and start != 0:
            LOGGER.debug(
                "잔여 토큰 수가 너무 작아 마지막 청크를 제외했습니다: %s (tokens=%s)",
                doc_id,
                len(window),
            )
            break
        
        # Reconstruct text first to find page markers
        raw_text = detokenize(window)
        
        # Find page numbers in this chunk
        pages_found = [int(p) for p in PAGE_PATTERN.findall(raw_text)]
        if pages_found:
            global_page_counter = pages_found[0] # Use the first page found in this chunk
        
        # Clean text for embedding/display (remove page markers)
        clean_text = PAGE_PATTERN.sub("", raw_text).strip()
        # Also clean up potential double newlines left by removing page marker
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

        anchors = sorted(set(ANCHOR_PATTERN.findall(clean_text)))
        
        LOGGER.debug(
            "청크 생성: %s tokens=%d page=%d",
            doc_id,
            len(window),
            global_page_counter,
        )
        chunk_index += 1
        chunk_id = f"{doc_id}#chunk{chunk_index:03d}"
        yield Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=clean_text,
            anchors=anchors,
            token_start=start,
            token_end=end,
            page=global_page_counter,
        )


def write_chunks(chunks: Iterable[Chunk], output_dir: pathlib.Path, doc_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_doc = doc_id.replace("/", "_")
    output_path = output_dir / f"{safe_doc}.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.to_json_line() + "\n")
    LOGGER.debug("Wrote chunks → %s", output_path)


def run(cfg: dict[str, object]) -> None:
    input_path = pathlib.Path(str(cfg["input"])).expanduser().resolve()
    output_dir = pathlib.Path(str(cfg["output_dir"])).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Normalized input not found: {input_path}")

    LOGGER.info("정규화된 문서를 로딩합니다: %s", input_path)
    docs = list(iter_normalized(input_path))
    LOGGER.info("총 %d개의 문서를 읽었습니다", len(docs))

    for record in docs:
        tokens = tokenize(record.text)
        chunks = list(chunk_tokens(record.doc_id, tokens, cfg))
        if not chunks:
            LOGGER.warning("청크가 생성되지 않았습니다: %s", record.doc_id)
            continue
        write_chunks(chunks, output_dir, record.doc_id)
        LOGGER.debug("생성된 청크 수: %s → %d", record.doc_id, len(chunks))
    LOGGER.info("청크 생성 완료. 출력 경로: %s", output_dir)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정규화된 문서에서 검색용 청크를 생성합니다.")
    parser.add_argument(
        "--cfg",
        help="YAML 구성 파일 경로. 생략 시 docs/tasks/20_chunking.md 기본값 사용",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="로깅 레벨 (DEBUG, INFO 등)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    setup_logger("make_chunks", settings.LOG_DIR / "env/chunking.log", lvl)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    cfg = load_config(args.cfg)
    try:
        run(cfg)
    except KeyboardInterrupt:
        LOGGER.warning("사용자에 의해 작업이 중단되었습니다.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
