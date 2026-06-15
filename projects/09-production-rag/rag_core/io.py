from __future__ import annotations

import base64
import csv
import json
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from rag_core.embeddings import post_json, siliconflow_url
from rag_core.types import ImageDocument, SourceDocument


@dataclass(frozen=True)
class PdfPage:
    page_no: int
    text: str
    display_text: str
    display_blocks: list[dict[str, str]]

    def __iter__(self):
        yield self.page_no
        yield self.text

    def __getitem__(self, index: int):
        return (self.page_no, self.text)[index]


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
    for raw_page in pages:
        page = normalize_pdf_page(raw_page)
        if not page.text.strip() and not page.display_blocks:
            continue
        docs.append(
            SourceDocument(
                tenant_id=tenant_id,
                doc_id=f"{path.relative_to(input_dir).with_suffix('').as_posix()}/page-{page.page_no}",
                doc_version=doc_version,
                source_type="pdf",
                source_uri=str(path),
                title=f"{path.stem} p{page.page_no}",
                text=page.text,
                language=language,
                acl_groups=acl_groups,
                metadata={
                    "relative_path": relative_path,
                    "file_size_bytes": path.stat().st_size,
                    "page_no": page.page_no,
                    "page_start": page.page_no,
                    "page_end": page.page_no,
                    "page_count": page_count,
                    "display_text": page.display_text,
                    "display_blocks": page.display_blocks,
                },
            )
        )
    return docs


def normalize_pdf_page(page: PdfPage | tuple[int, str]) -> PdfPage:
    if isinstance(page, PdfPage):
        return page
    page_no, text = page
    return PdfPage(
        page_no=page_no,
        text=text,
        display_text=text,
        display_blocks=[],
    )


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
    return _compact_whitespace("\n".join(page.text for page in extract_pdf_pages(path)))


def extract_pdf_pages(path: Path) -> list[PdfPage]:
    try:
        return extract_pdf_pages_with_pymupdf(path)
    except ImportError:
        return extract_pdf_pages_with_pypdf(path)


def extract_pdf_pages_with_pypdf(path: Path) -> list[PdfPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf. Run `uv add pypdf`.") from exc

    reader = PdfReader(str(path))
    return [
        PdfPage(
            page_no=page_no,
            text=_compact_whitespace(page.extract_text() or ""),
            display_text=_compact_whitespace(page.extract_text() or ""),
            display_blocks=[],
        )
        for page_no, page in enumerate(reader.pages, start=1)
    ]


def extract_pdf_pages_with_pymupdf(path: Path) -> list[PdfPage]:
    try:
        import fitz
    except ImportError as exc:
        raise exc

    image_captioner = PdfImageCaptioner.from_env()
    max_captioned_images = int(os.environ.get("RAG_PDF_CAPTION_MAX_IMAGES", "24"))
    captioned_images = 0
    pages: list[PdfPage] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            text_parts = [
                block
                for block in extract_pymupdf_text_blocks(page)
                if block.strip()
            ]
            table_parts = extract_pymupdf_tables(page)
            if table_parts:
                text_parts.extend(table_parts)
            retrieval_parts = list(text_parts)
            display_blocks: list[dict[str, str]] = []
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                image_data_url = pdf_image_data_url(document=document, image=image)
                if image_data_url:
                    display_blocks.append(
                        {
                            "type": "image",
                            "title": f"Image {image_index}",
                            "url": image_data_url,
                        }
                    )
                image_note = f"Image {image_index}: embedded image on PDF page {page_index}."
                if image_captioner and captioned_images < max_captioned_images:
                    caption = image_captioner.caption_pdf_image(
                        document=document,
                        image=image,
                        label=f"{path.name} page {page_index} image {image_index}",
                        language_hint=detect_text_language("\n".join(text_parts)),
                    )
                    captioned_images += 1
                    if caption:
                        image_note = f"Image {image_index} caption: {caption}"
                retrieval_parts.append(image_note)
            text = "\n\n".join(retrieval_parts)
            display_text = "\n\n".join(text_parts)
            pages.append(
                PdfPage(
                    page_no=page_index,
                    text=_compact_whitespace_preserve_lines(text),
                    display_text=_compact_whitespace_preserve_lines(display_text),
                    display_blocks=display_blocks,
                )
            )
    return pages


def extract_pymupdf_text_blocks(page) -> list[str]:
    blocks = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        text = _compact_whitespace_preserve_lines(str(block[4]))
        if text:
            blocks.append(text)
    return blocks


def extract_pymupdf_tables(page) -> list[str]:
    if not hasattr(page, "find_tables"):
        return []
    try:
        tables = page.find_tables()
    except Exception:
        return []
    table_parts: list[str] = []
    for table_index, table in enumerate(getattr(tables, "tables", []), start=1):
        try:
            rows = table.extract()
        except Exception:
            continue
        markdown = table_rows_to_markdown(rows)
        if markdown:
            table_parts.append(f"表格 {table_index}:\n{markdown}")
    return table_parts


def table_rows_to_markdown(rows: list[list[Any]]) -> str:
    cleaned = [
        [_compact_whitespace(str(cell or "")) for cell in row]
        for row in rows
        if any(str(cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    normalized = [row[:width] + [""] * max(0, width - len(row)) for row in cleaned]
    header = normalized[0]
    body = normalized[1:]
    return table_to_markdown(header, body)


class PdfImageCaptioner:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls) -> "PdfImageCaptioner | None":
        backend = os.environ.get("RAG_PDF_IMAGE_CAPTION_BACKEND", "siliconflow").lower()
        if backend in {"", "none", "off", "disabled"}:
            return None
        api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            return None
        return cls(
            base_url=os.environ.get("SILICONFLOW_URL", "https://api.siliconflow.cn"),
            api_key=api_key,
            model=os.environ.get("RAG_PDF_IMAGE_CAPTION_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        )

    def caption_pdf_image(self, *, document, image, label: str, language_hint: str = "en") -> str:
        xref = image[0]
        try:
            payload = document.extract_image(xref)
        except Exception:
            return ""
        image_bytes = payload.get("image")
        extension = payload.get("ext") or "png"
        if not image_bytes:
            return ""
        prompt = (
            "Describe the key information in this paper or source image concisely in English. "
            "If it is a chart, describe axes, process, structure, or conclusion. "
            f"Image source: {label}"
            if language_hint == "en"
            else (
                "请用中文简洁描述这张论文或资料图片中的关键信息。"
                "如果是图表，说明坐标、流程、结构或结论。"
                f"图片来源: {label}"
            )
        )
        try:
            response = post_json(
                siliconflow_url(self.base_url, "/chat/completions"),
                api_key=self.api_key,
                payload={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url_from_parts(extension=extension, image_bytes=image_bytes),
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 256,
                },
            )
        except Exception:
            return ""
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        return _compact_whitespace(str(content))


def pdf_image_data_url(*, document, image) -> str:
    xref = image[0]
    try:
        payload = document.extract_image(xref)
    except Exception:
        return ""
    image_bytes = payload.get("image")
    extension = payload.get("ext") or "png"
    if not image_bytes:
        return ""
    return image_data_url_from_parts(extension=extension, image_bytes=image_bytes)


def image_data_url_from_parts(*, extension: str, image_bytes: bytes) -> str:
    media_type = f"image/{'jpeg' if extension.lower() == 'jpg' else extension.lower()}"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def detect_text_language(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "en"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    alpha = len(re.findall(r"[A-Za-z]", stripped))
    return "zh" if cjk > max(8, alpha // 3) else "en"


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


def _compact_whitespace_preserve_lines(text: str) -> str:
    lines = [_compact_whitespace(line) for line in text.splitlines()]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and compacted:
                compacted.append("")
            previous_blank = True
            continue
        compacted.append(line)
        previous_blank = False
    return "\n".join(compacted).strip()
