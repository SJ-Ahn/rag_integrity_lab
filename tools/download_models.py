#!/usr/bin/env python3
"""
download_models.py
~~~~~~~~~~~~~~~~~~

config/models.yaml에 정의된 모델 아티팩트를 내려받아 예상 디렉터리 구조에 맞게
배치하는 도구 스크립트입니다. 모델 제공자에 따라 huggingface_hub 등의 SDK가 필요합니다.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Iterable

from huggingface_hub import snapshot_download

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "models.yaml"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config.model_registry import get_model_info, list_available_models  # noqa: E402

LOGGER = logging.getLogger("download_models")


def download_hf_repo(model_id: str, target_dir: pathlib.Path, allow_patterns: Iterable[str]) -> None:
    LOGGER.info("Downloading %s into %s", model_id, target_dir)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        allow_patterns=list(allow_patterns),
        local_dir_use_symlinks=False,
    )
    LOGGER.info("Completed download for %s", model_id)


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run(model_name: str) -> None:
    info = get_model_info(model_name)
    provider = info.get("provider", "local")
    target_path = ROOT_DIR / info.get("path", f"models/{model_name}")
    ensure_dir(target_path)

    if provider == "huggingface":
        model_id = info["hf_repo"]
        patterns = info.get("allow_patterns", ["*.gguf", "*.safetensors", "*.json"])
        download_hf_repo(model_id, target_path, patterns)
    else:
        LOGGER.warning("Provider '%s' not supported for automated download. Please place files manually.", provider)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models defined in config/models.yaml")
    parser.add_argument("--model", help="Model key to download (default: all models)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    available = list_available_models()
    if args.model:
        models = {args.model: available[args.model]}
    else:
        models = available

    if not models:
        LOGGER.error("No models defined in %s", CONFIG_PATH)
        return 1

    for name in models:
        try:
            run(name)
        except KeyError:
            LOGGER.error("Unknown model '%s'", name)
            return 1
        except Exception:
            LOGGER.exception("Failed to download model '%s'", name)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
