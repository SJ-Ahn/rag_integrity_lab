#!/usr/bin/env python3
"""
normalize_docs.py
~~~~~~~~~~~~~~~~~

Keycloak 원본 문서를 정규화(JSONL)하여 이후 청크 작업에 투입하기 위한 스크립트입니다.
HTML 본문에서 불필요한 태그를 제거하고, 인용용 앵커는 보존하여 추후 RAG 파이프라인에서
문헌 근거를 정확하게 추적할 수 있도록 합니다.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

from tools.logging_utils import ensure_stream_logging, setup_file_logger


try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    BeautifulSoup = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as extract_pdf_text  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    extract_pdf_text = None  # type: ignore


LOGGER = logging.getLogger("normalize_docs")


HTML_EXTENSIONS = {".html", ".htm"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".md", ".txt"}


@dataclass
class NormalizedDocument:
    doc_id: str
    source_path: pathlib.Path
    title: str
    text: str
    anchors: list[str]
    modified_at: dt.datetime

    def to_json_line(self) -> str:
        payload = {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "anchors": self.anchors,
            "source_path": str(self.source_path.as_posix()),
            "modified_at": self.modified_at.isoformat(),
        }
        return json.dumps(payload, ensure_ascii=False)

    def meta(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "source_path": str(self.source_path.as_posix()),
            "modified_at": self.modified_at.isoformat(),
            "anchor_count": len(self.anchors),
            "length_chars": len(self.text),
        }


def discover_sources(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in HTML_EXTENSIONS | TEXT_EXTENSIONS | PDF_EXTENSIONS:
            yield path
        else:
            LOGGER.debug("텍스트가 아닌 자원 건너뜀: %s", path)


def sanitize_doc_id(path: pathlib.Path, root: pathlib.Path) -> str:
    relative = path.relative_to(root).as_posix()
    candidate = re.sub(r"[^A-Za-z0-9/_-]+", "_", relative)
    candidate = re.sub(r"/+", "/", candidate)
    return candidate.lower()


def extract_html(path: pathlib.Path) -> tuple[str, str, list[str]]:
    if BeautifulSoup is None:
        raise RuntimeError(
            "HTML 정규화를 위해서는 BeautifulSoup(bs4)가 필요합니다. "
            "`uv pip install beautifulsoup4`로 설치해 주세요."
        )
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        # 검색 품질과 속도에 영향을 주지 않는 스크립트·스타일 태그는 제거한다.
        tag.decompose()

    anchors = []
    for anchor in soup.find_all(attrs={"name": True}):
        anchors.append(anchor["name"])

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else path.stem

    # 앵커 태그 뒤에 #[anchor] 표기를 주입하여 청크 생성 시 인용 지점을 추적 가능하게 한다.
    for anchor in soup.find_all("a", attrs={"name": True}):
        anchor.insert_after(soup.new_string(f" [#{anchor['name']}]"))

    body = soup.body or soup
    text = body.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return title, text, anchors


def extract_text(path: pathlib.Path) -> tuple[str, str, list[str]]:
    suffix = path.suffix.lower()
    if suffix in HTML_EXTENSIONS:
        return extract_html(path)
    if suffix in TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = text.splitlines()[0].strip() if text.splitlines() else path.stem
        return title, text, []
    if suffix in PDF_EXTENSIONS:
        return extract_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


try:
    from pdfminer.high_level import extract_pages  # type: ignore
    from pdfminer.layout import LTTextContainer  # type: ignore
except ImportError:  # pragma: no cover
    extract_pages = None  # type: ignore


def extract_pdf(path: pathlib.Path) -> tuple[str, str, list[str]]:
    if extract_pages is None:
        raise RuntimeError(
            "PDF 정규화를 위해서는 pdfminer.six가 필요합니다. `uv pip install pdfminer.six`를 먼저 실행하세요."
        )
    LOGGER.debug("PDF 텍스트 추출 시작 (페이지 별): %s", path)
    
    text_parts = []
    page_count = 0
    
    try:
        for i, page_layout in enumerate(extract_pages(str(path)), start=1):
            page_count = i
            # Insert Page Marker: [[PAGE:1]]
            text_parts.append(f"\n[[PAGE:{i}]]\n")
            
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text_parts.append(element.get_text())
    except Exception as e:
        LOGGER.error(f"PDF Parsing Error on {path}: {e}")
        # Fallback if extract_pages fails? No, simpler to just raise/log
        raise e

    full_text = "".join(text_parts)
    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    
    title = path.stem # Title extraction from text is flaky, use filename
    LOGGER.debug("PDF 텍스트 추출 완료: %s (페이지=%d, 길이=%d)", path, page_count, len(full_text))
    return title, full_text, []


def normalize_document(path: pathlib.Path, root: pathlib.Path) -> Optional[NormalizedDocument]:
    try:
        title, text, anchors = extract_text(path)
    except NotImplementedError as exc:
        LOGGER.warning(str(exc))
        return None
    except Exception:
        LOGGER.exception("정규화 실패: %s", path)
        return None

    stat = path.stat()
    doc_id = sanitize_doc_id(path, root=root)
    modified_at = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc)
    return NormalizedDocument(
        doc_id=doc_id,
        source_path=path,
        title=title,
        text=text,
        anchors=anchors,
        modified_at=modified_at,
    )


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> None:
    input_root = pathlib.Path(args.input).expanduser().resolve()
    normalized_dir = pathlib.Path(args.output).expanduser().resolve()
    meta_dir = pathlib.Path(args.meta_output).expanduser().resolve()

    ensure_dir(normalized_dir)
    ensure_dir(meta_dir)

    normalized_path = normalized_dir / "normalized.jsonl"
    LOGGER.info("원본 문서를 정규화합니다: %s", input_root)
    with normalized_path.open("w", encoding="utf-8") as normalized_file:
        for source_path in discover_sources(input_root):
            LOGGER.debug("정규화 대상 파일: %s", source_path)
            doc = normalize_document(source_path, root=input_root)
            if doc is None:
                continue
            normalized_file.write(doc.to_json_line() + "\n")

            meta_path = meta_dir / f"{doc.doc_id.replace('/', '_')}.json"
            meta_path.write_text(
                json.dumps(doc.meta(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            LOGGER.debug("정규화 완료: %s → %s", source_path, doc.doc_id)

    LOGGER.info("정규화 결과 파일: %s", normalized_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원본 문서를 JSONL 포맷으로 정규화합니다.")
    parser.add_argument("--input", "--in", default="data/source", help="원본 문서 디렉터리 경로")
    parser.add_argument(
        "--output",
        "--out",
        default="data/working/normalized",
        help="정규화된 JSONL을 저장할 디렉터리",
    )
    parser.add_argument(
        "--meta-output",
        "--meta-out",
        default="data/working/meta",
        help="문서 단위 메타데이터 JSON을 보관할 디렉터리",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="로깅 레벨 (DEBUG, INFO, WARNING 등)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    ensure_stream_logging(level)
    setup_file_logger(LOGGER, pathlib.Path("logs/env/normalize.log"), level)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        run(args)
    except KeyboardInterrupt:
        LOGGER.warning("사용자에 의해 중단되었습니다.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
