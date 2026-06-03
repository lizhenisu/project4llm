from __future__ import annotations

import csv
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
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        self._heading_path: list[str] = []
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "nav", "header", "footer", "aside"}:
            self._hidden_depth += 1
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_tag = tag.lower()
            self._heading_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "nav", "header", "footer", "aside"} and self._hidden_depth:
            self._hidden_depth -= 1
        if tag.lower() == "title":
            self._in_title = False
            title = _compact_whitespace(" ".join(self._title_parts))
            if title:
                self._title = title
        if tag.lower() == self._heading_tag:
            heading = _compact_whitespace(" ".join(self._heading_parts))
            if heading:
                level = int(self._heading_tag[1])
                self._heading_path = self._heading_path[: level - 1] + [heading]
                self._parts.append(" > ".join(self._heading_path))
            self._heading_tag = None
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._heading_tag:
            self._heading_parts.append(data)
            return
        self._parts.append(data)

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def text(self) -> str:
        return _compact_whitespace(" ".join(self._parts))

    @property
    def heading_path(self) -> list[str]:
        return self._heading_path


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
        if path.suffix.lower() == ".pdf":
            docs.extend(
                load_pdf_page_documents(
                    path,
                    input_dir=input_dir,
                    tenant_id=tenant_id,
                    doc_version=doc_version,
                    acl_groups=acl_groups,
                    language=language,
                )
            )
            continue

        if path.suffix.lower() == ".md":
            docs.extend(
                load_markdown_section_documents(
                    path,
                    input_dir=input_dir,
                    tenant_id=tenant_id,
                    doc_version=doc_version,
                    acl_groups=acl_groups,
                    language=language,
                )
            )
            continue

        text, title = extract_file_text(path)
        if not text.strip():
            continue
        relative_path = path.relative_to(input_dir).as_posix()
        metadata = {
            "relative_path": relative_path,
            "file_size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() in {".html", ".htm"}:
            _, _, heading_path = extract_html_document(path)
            metadata["heading_path"] = heading_path
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
                metadata=metadata,
            )
        )
    return docs


def load_markdown_section_documents(
    path: Path,
    *,
    input_dir: Path,
    tenant_id: str,
    doc_version: int,
    acl_groups: list[str],
    language: str,
) -> list[SourceDocument]:
    text = path.read_text(encoding="utf-8")
    sections = split_markdown_sections(text)
    relative_path = path.relative_to(input_dir).as_posix()
    docs: list[SourceDocument] = []
    for section_index, section in enumerate(sections):
        if not section["text"].strip():
            continue
        heading_path = section["heading_path"] or [extract_markdown_title(text) or path.stem]
        title = " > ".join(heading_path)
        suffix = "" if len(sections) == 1 else f"/section-{section_index:03d}"
        docs.append(
            SourceDocument(
                tenant_id=tenant_id,
                doc_id=f"{path.relative_to(input_dir).with_suffix('').as_posix()}{suffix}",
                doc_version=doc_version,
                source_type="md",
                source_uri=str(path),
                title=title,
                text=section["text"],
                language=language,
                acl_groups=acl_groups,
                metadata={
                    "relative_path": relative_path,
                    "file_size_bytes": path.stat().st_size,
                    "heading_path": heading_path,
                    "heading_level": section["heading_level"],
                    "section_index": section_index,
                },
            )
        )
    return docs


def load_pdf_page_documents(
    path: Path,
    *,
    input_dir: Path,
    tenant_id: str,
    doc_version: int,
    acl_groups: list[str],
    language: str,
) -> list[SourceDocument]:
    pages = extract_pdf_pages(path)
    relative_path = path.relative_to(input_dir).as_posix()
    page_count = len(pages)
    docs: list[SourceDocument] = []
    for page_no, text in pages:
        if not text.strip():
            continue
        docs.append(
            SourceDocument(
                tenant_id=tenant_id,
                doc_id=f"{path.relative_to(input_dir).with_suffix('').as_posix()}/page-{page_no}",
                doc_version=doc_version,
                source_type="pdf",
                source_uri=str(path),
                title=f"{path.stem} p{page_no}",
                text=text,
                language=language,
                acl_groups=acl_groups,
                metadata={
                    "relative_path": relative_path,
                    "file_size_bytes": path.stat().st_size,
                    "page_no": page_no,
                    "page_start": page_no,
                    "page_end": page_no,
                    "page_count": page_count,
                },
            )
        )
    return docs


def split_markdown_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    current_level = 0
    section_started = False

    def flush() -> None:
        nonlocal current_lines, current_path, current_level, section_started
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                {
                    "heading_path": current_path.copy(),
                    "heading_level": current_level,
                    "text": body,
                }
            )
        current_lines = []
        section_started = False

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if section_started:
                flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [heading]
            current_path = heading_stack.copy()
            current_level = level
            current_lines = [line]
            section_started = True
            continue
        current_lines.append(line)
        if line.strip():
            section_started = True

    flush()
    if sections:
        return sections
    stripped = text.strip()
    return [{"heading_path": [], "heading_level": 0, "text": stripped}] if stripped else []


def load_table_documents(
    input_dir: Path,
    *,
    tenant_id: str,
    doc_version: int,
    acl_groups: list[str],
    language: str = "zh",
    recursive: bool = True,
    rows_per_document: int = 200,
) -> list[SourceDocument]:
    """Load CSV/TSV tables as compact markdown table documents with metadata."""
    if rows_per_document <= 0:
        raise ValueError("rows_per_document must be greater than 0")

    paths: list[Path] = []
    for pattern in ("*.csv", "*.tsv"):
        paths.extend(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))

    docs: list[SourceDocument] = []
    for path in sorted(set(paths)):
        table = read_delimited_table(path)
        if not table.rows:
            continue
        relative_path = path.relative_to(input_dir).as_posix()
        total_rows = len(table.rows)
        for part_index, start in enumerate(range(0, total_rows, rows_per_document)):
            rows = table.rows[start : start + rows_per_document]
            end = start + len(rows)
            markdown = table_to_markdown(table.columns, rows)
            suffix = "" if total_rows <= rows_per_document else f"/part-{part_index:03d}"
            docs.append(
                SourceDocument(
                    tenant_id=tenant_id,
                    doc_id=f"{path.relative_to(input_dir).with_suffix('').as_posix()}{suffix}",
                    doc_version=doc_version,
                    source_type=path.suffix.lower().lstrip(".") or "table",
                    source_uri=str(path),
                    title=path.stem if not suffix else f"{path.stem} part {part_index + 1}",
                    text=markdown,
                    language=language,
                    acl_groups=acl_groups,
                    metadata={
                        "relative_path": relative_path,
                        "file_size_bytes": path.stat().st_size,
                        "table_format": path.suffix.lower().lstrip("."),
                        "delimiter": table.delimiter,
                        "columns": table.columns,
                        "column_count": len(table.columns),
                        "row_count_total": total_rows,
                        "row_start": start + 1,
                        "row_end": end,
                        "row_count": len(rows),
                    },
                )
            )
    return docs


class DelimitedTable:
    def __init__(self, columns: list[str], rows: list[list[str]], delimiter: str) -> None:
        self.columns = columns
        self.rows = rows
        self.delimiter = delimiter


def read_delimited_table(path: Path) -> DelimitedTable:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            [_compact_whitespace(cell) for cell in row]
            for row in csv.reader(file, delimiter=delimiter)
            if any(cell.strip() for cell in row)
        ]

    if not rows:
        return DelimitedTable(columns=[], rows=[], delimiter=delimiter)

    header = rows[0]
    body = rows[1:]
    max_columns = max(len(row) for row in rows)
    columns = normalize_table_columns(header, max_columns)
    normalized_rows = [normalize_table_row(row, max_columns) for row in body]
    return DelimitedTable(columns=columns, rows=normalized_rows, delimiter=delimiter)


def normalize_table_columns(header: list[str], column_count: int) -> list[str]:
    columns: list[str] = []
    seen: dict[str, int] = {}
    for index in range(column_count):
        raw = header[index].strip() if index < len(header) else ""
        name = raw or f"column_{index + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        columns.append(name if count == 0 else f"{name}_{count + 1}")
    return columns


def normalize_table_row(row: list[str], column_count: int) -> list[str]:
    padded = row[:column_count] + [""] * max(0, column_count - len(row))
    return padded


def table_to_markdown(columns: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(markdown_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def markdown_cell(value: str) -> str:
    return _compact_whitespace(value).replace("\\", "\\\\").replace("|", "\\|")


def extract_file_text(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path), path.stem
    if suffix in {".html", ".htm"}:
        text, title, _ = extract_html_document(path)
        return text, title
    text = path.read_text(encoding="utf-8")
    return text, extract_markdown_title(text) if suffix == ".md" else path.stem


def extract_pdf_text(path: Path) -> str:
    return _compact_whitespace("\n".join(text for _, text in extract_pdf_pages(path)))


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Run `uv add pypdf`.") from exc

    reader = PdfReader(str(path))
    return [
        (page_no, _compact_whitespace(page.extract_text() or ""))
        for page_no, page in enumerate(reader.pages, start=1)
    ]


def extract_html_text(path: Path) -> tuple[str, str | None]:
    text, title, _ = extract_html_document(path)
    return text, title


def extract_html_document(path: Path) -> tuple[str, str | None, list[str]]:
    parser = _VisibleTextHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.text, parser.title or path.stem, parser.heading_path


def extract_markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
