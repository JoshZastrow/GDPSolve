"""process() v0: ExtractionRequest -> ExtractionResult (docs/process-spec.md).

A router gives jobs to worker threads. A worker uses tools to read the document and sends a report.
The router merges the reports into the result.
Each task changes only its sections. tinker and tinker_cookbook are imported inside functions.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

from gdpsolve.eval import pdf_input
from gdpsolve.schemas import (
    BoundingBox,
    Crop,
    Document,
    ExtractionRequest,
    ExtractionResult,
    InspectPdf,
    Job,
    Message,
    Post,
    ReadNativeText,
    Recognize,
    Region,
    Render,
    Report,
    Split,
)

if TYPE_CHECKING:
    from PIL import Image

# --- config ---

MODEL = "thinkingmachines/Inkling-Small"
EFFORT = 0.3
MAX_TOKENS = 4096  # for each step
MAX_COST_PER_DOC = 0.15  # USD
PRICES = {"prefill": 0.58, "sample": 1.44}  # USD for each 1M tokens
OCR_URL = os.environ.get("GDPSOLVE_OCR_URL", "http://127.0.0.1:9931")
OCR_PROMPTS = {
    "text": "OCR:",
    "handwriting": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "symbol": "Spotting:",
}


@dataclass
class Run:
    """The state of one request. Use `lock` for every write to thread, reports, regions, memo, and cost."""

    doc: Document
    req: ExtractionRequest
    policy: Policy
    recognizer: Callable[[Image.Image, str], str]
    jobs: queue.Queue[Job | None] = field(default_factory=queue.Queue)
    events: queue.Queue[Message] = field(default_factory=queue.Queue)  # split messages
    thread: list[Message] = field(default_factory=list)  # the Thread
    reports: dict[str, Report] = field(default_factory=dict)  # job id -> latest report
    regions: dict[str, Region] = field(default_factory=dict)
    memo: dict[str, str] = field(default_factory=dict)  # tool cache: sha256(doc, tool, args) -> result
    waiting: list[Job] = field(default_factory=list)  # jobs with `after` set, not yet in `jobs`
    final_id: str = ""  # the job whose end ends the request: the root job, or the merge job after a split
    cost: float = 0.0  # USD
    lock: threading.Lock = field(default_factory=threading.Lock)
    deadline: float = field(init=False)  # time.time() value

    def __post_init__(self):
        self.deadline = time.time() + self.req.deadline_s

    def post(self, msg: Message) -> Message:
        """Add `msg` to the Thread with the next seq. Also put a split message on `events`."""
        with self.lock:
            msg.seq = len(self.thread)
            self.thread.append(msg)
        if msg.kind == "split":
            self.events.put(msg)
        return msg

    def charge(self, turn: Turn) -> float:
        """Add the price of one policy turn to `cost`. Returns the price."""
        price = (turn.prompt_tokens * PRICES["prefill"] + turn.completion_tokens * PRICES["sample"]) / 1e6
        with self.lock:
            self.cost += price
        return price

    def final_ended(self) -> bool:
        """True when the job `final_id` has a ready or failed report."""
        raise NotImplementedError

    def put_merge_job_when_children_ended(self) -> None:
        """Move each waiting job whose `after` jobs have all ended to `jobs`."""
        raise NotImplementedError


# --- ingest ---


def ingest(req: ExtractionRequest) -> Document | ExtractionResult:
    """Hash the file, try the passwords, read the metadata, and store the bytes; or a failed/password_required result."""
    raise NotImplementedError


# --- tools ---


# pdfium is not thread-safe; workers share this lock for every pdfium call.
_PDFIUM_LOCK = threading.Lock()


def _memo(run: Run, tool: str, args, compute: Callable[[], str]) -> str:
    """`compute()` once for each (doc, tool, canonical args); later calls read `run.memo`."""
    key = hashlib.sha256(json.dumps([run.doc.metadata.sha256, tool, args.model_dump(mode="json")],
                                    sort_keys=True).encode()).hexdigest()
    if key not in run.memo:
        result = compute()
        with run.lock:
            run.memo[key] = result
    return run.memo[key]


def inspect_pdf(run: Run, args: InspectPdf) -> str:
    """Page size, rotation, text layer, and image cover of one page, as text."""
    def compute() -> str:
        import pypdfium2.raw as pdfium_c

        with _PDFIUM_LOCK:
            page = _open(run)[args.page - 1]
            w, h = page.get_size()
            has_text = bool(pdf_input.page_text(page))
            area = 0.0
            for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
                x0, y0, x1, y1 = obj.get_bounds()
                area += max(0.0, min(x1, w) - max(x0, 0)) * max(0.0, min(y1, h) - max(y0, 0))
            rotation = page.get_rotation()
        return (f"page {args.page}: {w:g} x {h:g} pt, rotation {rotation}, has_text_layer={has_text}, "
                f"image_cover={min(1.0, area / (w * h)):.2f}")
    return _memo(run, "inspect_pdf", args, compute)


def read_native_text(run: Run, args: ReadNativeText) -> str:
    """Text layer of one page, or of one box (points from the top left of the page)."""
    def compute() -> str:
        with _PDFIUM_LOCK:
            page = _open(run)[args.page - 1]
            if args.bbox is None:
                return pdf_input.page_text(page)
            h = page.get_height()
            b = args.bbox
            return page.get_textpage().get_text_bounded(left=b.x0, bottom=h - b.y1, right=b.x1, top=h - b.y0).strip()
    return _memo(run, "read_native_text", args, compute)


def render(run: Run, args: Render) -> Region:
    """PNG of one page at `dpi`, in the store and in `run.regions`."""
    return _region(run, "render", args, args.page, None, args.dpi)


def crop(run: Run, args: Crop) -> Region:
    """PNG of one box at `dpi`, in the store and in `run.regions`. The box gets `padding` points on each side."""
    return _region(run, "crop", args, args.page, args.bbox, args.dpi, args.padding)


def _open(run: Run):
    """The document as a pdfium PdfDocument; call while holding _PDFIUM_LOCK."""
    import pypdfium2 as pdfium

    from gdpsolve.verifier import store

    return pdfium.PdfDocument(store.get(run.doc.pdf))


def _region(run: Run, tool: str, args, page_no: int, bbox: BoundingBox | None, dpi: int, padding: float = 0.0) -> Region:
    """Render a page, or a box of it (points from the top left), to a PNG Region; memoized."""
    def compute() -> str:
        from gdpsolve.verifier import store

        with _PDFIUM_LOCK:
            page = _open(run)[page_no - 1]
            w, h = page.get_size()
            box = None
            cut = (0, 0, 0, 0)
            if bbox is not None:
                box = BoundingBox(page=page_no, x0=max(0.0, bbox.x0 - padding), y0=max(0.0, bbox.y0 - padding),
                                  x1=min(w, bbox.x1 + padding), y1=min(h, bbox.y1 + padding))
                cut = (box.x0, h - box.y1, w - box.x1, box.y0)  # left, bottom, right, top
            image = page.render(scale=dpi / 72, crop=cut).to_pil()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        ref = store.put(buf.getvalue(), "image/png")
        sha = run.doc.metadata.sha256
        rid = hashlib.sha256(json.dumps([sha, page_no, box and box.model_dump(), dpi, []]).encode()).hexdigest()
        return Region(id=rid, doc_sha256=sha, page=page_no, bbox=box, dpi=dpi, image=ref).model_dump_json()
    region = Region.model_validate_json(_memo(run, tool, args, compute))
    with run.lock:
        run.regions.setdefault(region.id, region)
    return region


def recognize(run: Run, args: Recognize) -> str:
    """`run.recognizer` on one Region: text, HTML, or LaTeX."""
    raise NotImplementedError


def run_tool(run: Run, job: Job, tool: str, args: dict) -> list[dict]:
    """Run one of the 8 tools for `job`; the observation as content parts ({"type": "text"|"image", ...})."""
    from PIL import Image

    from gdpsolve.verifier import store

    texts = {"inspect_pdf": (InspectPdf, inspect_pdf), "read_native_text": (ReadNativeText, read_native_text),
             "recognize": (Recognize, recognize)}
    images = {"render": (Render, render), "crop": (Crop, crop)}
    if tool in texts:
        model, fn = texts[tool]
        return [{"type": "text", "text": fn(run, model.model_validate(args))}]
    if tool in images:
        model, fn = images[tool]
        region = fn(run, model.model_validate(args))
        image = Image.open(io.BytesIO(store.get(region.image)))
        return [{"type": "text", "text": f"region {region.id}: page {region.page}, {region.dpi} dpi, "
                                         f"{image.width}x{image.height} px"},
                {"type": "image", "image": image}]
    if tool == "split":
        a = Split.model_validate(args)
        run.post(Message(thread=run.req.id, job_id=job.id, kind="split", pages=a.pages or [],
                         pages_per_job=a.pages_per_job))
        return [{"type": "text", "text": "split requested"}]
    if tool == "post":
        a = Post.model_validate(args)
        run.post(Message(thread=run.req.id, job_id=job.id, kind=a.kind, pages=a.pages, note=a.note))
        return [{"type": "text", "text": "posted"}]
    if tool == "report":
        raise NotImplementedError("report: hook for the worker task (T4)")
    raise ValueError(f"unknown tool: {tool}")


# --- recognizer ---


def ensure_ocr_server(url: str = OCR_URL, timeout_s: int = 300) -> float:
    """Start the PaddleOCR-VL server if it is not up, then warm it up; returns the warm-up seconds."""
    raise NotImplementedError


class PaddleRecognizer:
    """PaddleOCR-VL client for the local llama.cpp server (OpenAI-compatible API)."""

    def __init__(self, url: str = OCR_URL):
        self.url = url

    def __call__(self, image: Image.Image, mode: str) -> str:
        """Text of the first choice for `image` with the prompt OCR_PROMPTS[mode]."""
        raise NotImplementedError


# --- policy ---


class Turn(NamedTuple):
    tool: str
    args: dict
    prompt_tokens: int
    completion_tokens: int


class Policy(Protocol):
    def step(self, messages: list) -> Turn: ...


class TinkerPolicy:
    """MODEL with the tml_v0 renderer; the tool specs are the TOOLBOX JSON Schemas."""

    def __init__(self, model: str = MODEL):
        self.model = model

    def step(self, messages: list) -> Turn:
        """Sample one turn (EFFORT, MAX_TOKENS) and parse its tool call."""
        raise NotImplementedError


# --- worker ---


def run_job(run: Run, job: Job) -> None:
    """Ask the policy for turns and run their tools until a ready/failed report or a limit."""
    raise NotImplementedError


def worker(run: Run) -> None:
    """Run jobs from `run.jobs` until a None stop signal."""
    raise NotImplementedError


# --- router ---


def plan(doc: Document, req: ExtractionRequest) -> Job:
    """One root job for all pages; max_steps = min(40, 8 + page_count // 5)."""
    raise NotImplementedError


def react(doc: Document, req: ExtractionRequest, msg: Message | None, run: Run) -> list[Job]:
    """For a split message: child jobs, and a merge job put in `run.waiting`; else []."""
    raise NotImplementedError


def merge(doc: Document, req: ExtractionRequest, reports: dict[str, Report]) -> ExtractionResult:
    """The result from the ready reports, without a model call."""
    raise NotImplementedError


# --- process ---


def process(req: ExtractionRequest, policy: Policy | None = None,
            recognizer: Callable[[Image.Image, str], str] | None = None, workers: int = 4) -> ExtractionResult:
    """Ingest, run the router and the worker threads until the final job ends or the deadline, then merge."""
    raise NotImplementedError


# --- eval ---


def run_eval(tasks: list[dict], data: Path, results: Path, policy: Policy | None = None,
             recognizer: Callable[[Image.Image, str], str] | None = None) -> list[dict]:
    """process() for each GDP.pdf task; append one baseline-format row (+ steps, cost, jobs) to `results`."""
    raise NotImplementedError
