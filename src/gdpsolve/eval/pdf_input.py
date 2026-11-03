"""PDF to model input (docs/m1-baseline.md, step 1): text-layer pages as text, scanned pages as images."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import pypdfium2 as pdfium

from gdpsolve.schemas import DocumentMetadata


def metadata(path: Path) -> DocumentMetadata:
    pdf = pdfium.PdfDocument(path)
    info = pdf.get_metadata_dict(skip_empty=True)
    return DocumentMetadata(
        filename=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        page_count=len(pdf),
        encrypted=False,  # pdfium raises on encrypted PDFs without a password
        has_text_layer=any(_text(page) for page in pdf),
        title=info.get("Title"),
        author=info.get("Author"),
        producer=info.get("Producer"),
    )


def content(path: Path, count_tokens: Callable[[list[dict]], int], budget: int = 120_000,
            dpi: int = 100) -> tuple[list[dict], int | None]:
    """Content parts for the pages, in order, until `budget` tokens.

    Returns (parts, truncated_at_page): the first page left out, or None when every page fits.
    """
    parts: list[dict] = []
    used = 0
    for n, page in enumerate(pdfium.PdfDocument(path), start=1):
        text = _text(page)
        if text:
            page_parts = [{"type": "text", "text": f"--- page {n} ---\n{text}"}]
        else:
            image = page.render(scale=dpi / 72).to_pil()
            page_parts = [{"type": "text", "text": f"--- page {n} ---"}, {"type": "image", "image": image}]
        cost = count_tokens(page_parts)
        if used + cost > budget:
            return parts, n
        parts += page_parts
        used += cost
    return parts, None


def _text(page: pdfium.PdfPage) -> str:
    return page.get_textpage().get_text_bounded().strip()
