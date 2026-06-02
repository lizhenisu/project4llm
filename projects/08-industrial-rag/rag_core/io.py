from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from rag_core.types import ImageDocument, SourceDocument


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_documents(path: Path) -> list[SourceDocument]:
    return [SourceDocument(**row) for row in read_jsonl(path)]


def load_image_documents(path: Path) -> list[ImageDocument]:
    return [ImageDocument(**row) for row in read_jsonl(path)]


class _VisibleTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._hidden_depth = 0
        self._title: str | None = None
        self._in_title = False
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1
        if tag.lower() == "title":
            self._in_title = False
            title = _compact_whitespace(" ".join(self._title_parts))
            if title:
                self._title = title

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        self._parts.append(data)

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def text(self) -> str:
        return _compact_whitespace(" ".join(self._parts))


def load_file_documents(
    input_dir: Path,
    *,
    tenant_id: str,
    doc_version: int,
    acl_groups: list[str],
    language: str = "zh",
    recursive: bool = True,
) -> list[SourceDocument]:
    """Load common office/web text files into the canonical SourceDocument shape."""
    patterns = ("*.pdf", "*.html", "*.htm", "*.md", "*.txt")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))

    docs: list[SourceDocument] = []
    for path in sorted(set(paths)):
        text, title = extract_file_text(path)
        if not text.strip():
            continue
        relative_path = path.relative_to(input_dir).as_posix()
        docs.append(
            SourceDocument(
                tenant_id=tenant_id,
                doc_id=path.relative_to(input_dir).with_suffix("").as_posix(),
                doc_version=doc_version,
                source_type=path.suffix.lower().lstrip(".") or "file",
                source_uri=str(path),
                title=title or path.stem,
                text=text,
                language=language,
                acl_groups=acl_groups,
                metadata={
                    "relative_path": relative_path,
                    "file_size_bytes": path.stat().st_size,
                },
            )
        )
    return docs


def extract_file_text(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path), path.stem
    if suffix in {".html", ".htm"}:
        return extract_html_text(path)
    text = path.read_text(encoding="utf-8")
    return text, extract_markdown_title(text) if suffix == ".md" else path.stem


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Run `uv add pypdf`.") from exc

    reader = PdfReader(str(path))
    page_text = [page.extract_text() or "" for page in reader.pages]
    return _compact_whitespace("\n".join(page_text))


def extract_html_text(path: Path) -> tuple[str, str | None]:
    parser = _VisibleTextHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.text, parser.title or path.stem


def extract_markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
