"""
model_registry.py
~~~~~~~~~~~~~~~~~

config/models.yaml에 정의된 LLM 정보를 읽어 서비스와 평가 코드가 동일한 기본값을
사용하도록 돕는 유틸리티 모듈입니다.
"""

from __future__ import annotations

import functools
import pathlib
from typing import Dict

import yaml

CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "models.yaml"


@functools.lru_cache(maxsize=1)
def load_model_config() -> Dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_default_model_name() -> str:
    cfg = load_model_config()
    return str(cfg.get("default", "llama3.1"))


def get_model_info(model_name: str | None = None) -> Dict[str, object]:
    cfg = load_model_config()
    models = cfg.get("models", {})
    target = model_name or get_default_model_name()
    if target not in models:
        raise KeyError(f"모델 '{target}'을(를) models.yaml에서 찾을 수 없습니다.")
    return models[target]


def list_available_models() -> Dict[str, Dict[str, object]]:
    cfg = load_model_config()
    return cfg.get("models", {})
