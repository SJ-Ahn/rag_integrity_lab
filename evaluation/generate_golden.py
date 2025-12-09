#!/usr/bin/env python3
"""
generate_golden.py
~~~~~~~~~~~~~~~~~~

OpenAI API를 활용해 청크 기반 질문·정답·인용 골든셋을 자동 생성합니다.
생성된 항목은 evaluation/datasets 하위에 JSONL 형태로 저장되며, RAG 파이프라인
평가에 사용할 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import random
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from openai import OpenAI

from tools.logging_utils import ensure_stream_logging, setup_file_logger


LOGGER = logging.getLogger("generate_golden")
QA_LOGGER = logging.getLogger("generate_golden.qa")

SYSTEM_PROMPT = (
    "당신은 AWS EC2 및 DBMS 기술 문서를 기반으로 한국어 질의·응답을 설계하는 도우미입니다. "
    "주어진 chunk_id와 doc_id를 반드시 인용에 포함시키고, 입력 텍스트에 존재하지 않는 "
    "사실은 절대 추가하지 마세요."
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "Chunk":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload.get("doc_id", "")),
            text=str(payload.get("text", "")),
        )


def load_chunks(glob_pattern: str, limit: Optional[int] = None) -> list[Chunk]:
    records: list[Chunk] = []
    paths = sorted(pathlib.Path().glob(glob_pattern))
    for path in paths:
        LOGGER.debug("청크 파일 로딩: %s", path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(Chunk.from_json(payload))
    if limit:
        LOGGER.debug("청크 제한 적용: %d → %d", len(records), limit)
        records = records[:limit]
    LOGGER.info("총 %d개의 청크를 로드했습니다 (%s)", len(records), glob_pattern)
    return records


def build_user_prompt(chunk: Chunk) -> str:
    # 토큰 비용을 줄이기 위해 텍스트를 2,000자 수준으로 절단한다.
    trimmed_text = chunk.text[:2000]
    return (
        "다음 AWS EC2 및 DBMS 문서 청크를 바탕으로 QA 골든셋 항목 1개를 만들어 주세요.\n"
        f"doc_id: {chunk.doc_id}\n"
        f"chunk_id: {chunk.chunk_id}\n"
        "출력 형식(JSON):\n"
        '{\n  "query": "...",\n  "expected_doc_ids": ["..."],\n  '
        '"expected_chunk_ids": ["..."],\n  "answer_summary": "..." \n}\n'
        "조건:\n"
        "1. query는 한국어 자연어 질문이어야 합니다.\n"
        "2. expected_doc_ids에는 위 doc_id만 포함하세요.\n"
        "3. expected_chunk_ids에는 위 chunk_id만 포함하세요.\n"
        "4. answer_summary는 해당 청크 내용만을 근거로 2~3문장 요약을 작성하세요.\n"
        "5. 질문은 실제 실무 시나리오를 상정하여 작성하세요.\n\n"
        f"[청크 내용]\n{trimmed_text}"
    )


def request_completion(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_json_block(content: str) -> str:
    """
    OpenAI 응답이 ```json ... ``` 코드 블록 형태로 감싸져 있을 경우 본문만 추출합니다.
    """

    trimmed = content.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```[a-zA-Z]*\s*", "", trimmed)
        trimmed = re.sub(r"\s*```$", "", trimmed)
    return trimmed


def generate_golden_items(
    chunks: Iterable[Chunk],
    client: OpenAI,
    model: str,
    count: int,
    temperature: float,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    chunk_list = list(chunks)
    if not chunk_list:
        return items
    sample_size = min(count, len(chunk_list))
    # 무작위 샘플링으로 다양한 문서 영역을 골고루 다룰 수 있도록 한다.
    LOGGER.debug("샘플링 대상 청크 수: %d", len(chunk_list))
    selected_chunks = random.sample(chunk_list, k=sample_size)
    for chunk in selected_chunks:
        user_prompt = build_user_prompt(chunk)
        LOGGER.debug("OpenAI 프롬프트 전송: %s", chunk.chunk_id)
        content = request_completion(client, model, user_prompt, temperature)
        try:
            item = json.loads(_extract_json_block(content))
        except json.JSONDecodeError:
            LOGGER.error("JSON 파싱 실패(%s): %s", chunk.chunk_id, content)
            continue

        # 모델이 누락했을 경우 대비하여 기본 인용 정보를 보강한다.
        item.setdefault("expected_doc_ids", [chunk.doc_id])
        item.setdefault("expected_chunk_ids", [chunk.chunk_id])
        items.append(item)
        QA_LOGGER.info(
            json.dumps(
                {
                    "query": item.get("query"),
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "answer_summary": item.get("answer_summary"),
                },
                ensure_ascii=False,
            )
        )
        LOGGER.debug("골든 항목 생성: %s", chunk.chunk_id)
    LOGGER.info("총 %d개의 골든셋 항목을 생성했습니다", len(items))
    return items


def persist_items(items: list[dict[str, object]], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    LOGGER.info("생성된 골든셋을 저장했습니다: %s", output_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI API로 QA 골든셋을 생성합니다.")
    parser.add_argument("--chunk-glob", default="data/working/chunks/*.jsonl", help="청크 파일 glob 경로")
    parser.add_argument("--count", type=int, default=5, help="생성할 골든셋 항목 수")
    parser.add_argument("--model", default="gpt-4o-mini", help="사용할 OpenAI 모델 이름")
    parser.add_argument("--temperature", type=float, default=0.2, help="생성 다양성(temperature)")
    parser.add_argument(
        "--version",
        default="v1",
        help="골든셋 버전 폴더 이름 (예: v1, v2). 기본값은 v1.",
    )
    parser.add_argument(
        "--output",
        help="출력 JSONL 경로. 지정하지 않으면 evaluation/datasets/<version>/golden_generated.jsonl 로 저장됩니다.",
    )
    parser.add_argument("--log-level", default="INFO", help="로깅 레벨")
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    ensure_stream_logging(level)
    setup_file_logger(LOGGER, pathlib.Path("logs/qa/golden_generation.log"), level)
    setup_file_logger(QA_LOGGER, pathlib.Path("logs/qa/golden_items.log"), level)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        return 1

    client = OpenAI(api_key=api_key)
    chunks = load_chunks(args.chunk_glob)
    if not chunks:
        LOGGER.error("청크 데이터를 찾을 수 없습니다. make_chunks.py를 먼저 실행하세요.")
        return 1

    items = generate_golden_items(
        chunks=chunks,
        client=client,
        model=args.model,
        count=args.count,
        temperature=args.temperature,
    )
    if not items:
        LOGGER.error("골든셋을 생성하지 못했습니다.")
        return 1

    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        output_path = pathlib.Path(f"evaluation/datasets/{args.version}/golden_generated.jsonl")
    LOGGER.debug("골든셋 출력 경로: %s", output_path)
    persist_items(items, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
