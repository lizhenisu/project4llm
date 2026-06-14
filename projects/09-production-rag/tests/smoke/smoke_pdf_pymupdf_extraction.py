from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.io import extract_pdf_pages


def main() -> None:
    old_backend = os.environ.get("RAG_PDF_IMAGE_CAPTION_BACKEND")
    os.environ["RAG_PDF_IMAGE_CAPTION_BACKEND"] = "none"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "layout.pdf"
            build_fixture_pdf(pdf_path)
            pages = extract_pdf_pages(pdf_path)
    finally:
        if old_backend is None:
            os.environ.pop("RAG_PDF_IMAGE_CAPTION_BACKEND", None)
        else:
            os.environ["RAG_PDF_IMAGE_CAPTION_BACKEND"] = old_backend

    assert len(pages) == 1
    text = pages[0][1]
    assert "Attention fixture" in text
    assert "表格" in text
    assert "| Layer | Value |" in text
    assert "图片信息" in text
    print("smoke_pdf_pymupdf_extraction=ok")


def build_fixture_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=420, height=360)
    page.insert_text((36, 36), "Attention fixture")
    table = [["Layer", "Value"], ["Encoder", "6"], ["Decoder", "6"]]
    x0, y0 = 36, 80
    col_widths = [130, 90]
    row_height = 24
    for row_index, row in enumerate(table):
        y = y0 + row_index * row_height
        x = x0
        for col_index, cell in enumerate(row):
            rect = fitz.Rect(x, y, x + col_widths[col_index], y + row_height)
            page.draw_rect(rect)
            page.insert_text((x + 5, y + 16), cell)
            x += col_widths[col_index]
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), 0)
    pixmap.clear_with(0x88CCFF)
    page.insert_image(fitz.Rect(36, 170, 96, 230), pixmap=pixmap)
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    main()
