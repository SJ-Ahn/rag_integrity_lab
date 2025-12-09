"""
logging_utils.py
~~~~~~~~~~~~~~~~

RAG 파이프라인의 각 구성 요소가 공통된 로깅 포맷과 파일 구조를 사용하도록 돕는
보조 함수 모음입니다. docs/tasks/70_logging_monitoring.md에서 정의한 디렉터리
구조(env/http/qa/latency)에 맞춰 INFO 로그를 파일로 남길 수 있습니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def _normalize_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper(), logging.INFO)


def setup_file_logger(
    logger: logging.Logger,
    log_path: Union[str, Path],
    level: Union[int, str] = logging.INFO,
) -> logging.Logger:
    """
    지정된 로거에 파일 핸들러를 추가하고 INFO 이상 로그를 기록하도록 설정합니다.
    동일 경로의 핸들러가 이미 존재하면 중복 추가를 피합니다.
    """

    resolved_path = Path(log_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(_normalize_level(level))
    logger.propagate = True

    target = resolved_path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = Path(handler.baseFilename).resolve()
            if existing == target:
                handler.setLevel(_normalize_level(level))
                return logger

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.setLevel(_normalize_level(level))
    logger.addHandler(file_handler)
    return logger


def ensure_stream_logging(level: Union[int, str] = logging.INFO) -> None:
    """
    루트 로거에 스트림 핸들러가 없을 경우 기본 콘솔 로깅을 설정합니다.
    """

    root_logger = logging.getLogger()
    normalized = _normalize_level(level)
    has_stream = any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers)

    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        stream_handler.setLevel(normalized)
        root_logger.addHandler(stream_handler)

    root_logger.setLevel(normalized)
