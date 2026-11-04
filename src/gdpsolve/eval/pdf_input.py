"""PDF to model input (docs/m1-baseline.md, step 1): text-layer pages as text, scanned pages as images."""

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium

from gdpsolve.schemas import DocumentMetadata

if TYPE_CHECKING:
    from PIL.Image import Image


def metadata(path: Path, password: str | None = None) -> DocumentMetadata:
    pdf = pdfium.PdfDocument(path, password=password)
    info = pdf.get_metadata_dict(skip_empty=True)
    return DocumentMetadata(
        filename=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        page_count=len(pdf),
        encrypted=password is not None,  # pdfium raises on encrypted PDFs without a password
        has_text_layer=any(page_text(page) for page in pdf),
        title=info.get("Title"),
        author=info.get("Author"),
        producer=info.get("Producer"),
    )


def content(path: Path, count_tokens: Callable[[list[dict]], int], budget: int = 120_000,
            dpi: int = 100) -> tuple[list[dict], int | None]:
    """Content parts for the pages, in order, until `budget` tokens.

    Returns (parts, truncated_at_page): the first page left out, or None when every page fits.
    """
    def pages():
        for n, page in enumerate(pdfium.PdfDocument(path), start=1):
            text = page_text(page)
            yield n, text, None if text else page.render(scale=dpi / 72).to_pil()
    return parts(pages(), count_tokens, budget)


def parts(pages: Iterable[tuple[int, str, "Image | None"]], count_tokens: Callable[[list[dict]], int],
          budget: int) -> tuple[list[dict], int | None]:
    """Content parts for (page number, text, image or None) pages, in order, until `budget` tokens.

    A page with an image becomes its header and the image; a page without one becomes its header and text.
    Returns (parts, truncated_at_page) as `content` does.
    """
    out: list[dict] = []
    used = 0
    for n, text, image in pages:
        if image is None:
            page_parts = [{"type": "text", "text": f"--- page {n} ---\n{text}"}]
        else:
            page_parts = [{"type": "text", "text": f"--- page {n} ---"}, {"type": "image", "image": image}]
        cost = count_tokens(page_parts)
        if used + cost > budget:
            return out, n
        out += page_parts
        used += cost
    return out, None


def page_text(page: pdfium.PdfPage) -> str:
    return page.get_textpage().get_text_bounded().strip()
