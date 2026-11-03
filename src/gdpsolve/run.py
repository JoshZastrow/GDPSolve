"""process() v0: ExtractionRequest -> ExtractionResult (docs/process-spec.md).

A router gives jobs to worker threads. A worker uses tools to read the document and sends a report.
The router merges the reports into the result.
Each task changes only its sections. tinker and tinker_cookbook are imported inside functions.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

from gdpsolve.schemas import (
    Crop,
    Document,
    ExtractionRequest,
    ExtractionResult,
    InspectPdf,
    Job,
    Message,
    ReadNativeText,
    Recognize,
    Region,
    Render,
    Report,
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
    """Hash the file, try the passwords, read the metadata, and store the bytes; or a failed/password_required result.

    An encrypted PDF is stored decrypted, so the tools open `doc.pdf` without a password.
    """
    import hashlib
    import io

    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    from gdpsolve.eval import pdf_input
    from gdpsolve.schemas import DocumentMetadata
    from gdpsolve.verifier import store

    def result(status: str, sha: str = "", encrypted: bool = False, error: str | None = None) -> ExtractionResult:
        meta = DocumentMetadata(filename=req.path.name, sha256=sha, page_count=0, encrypted=encrypted,
                                has_text_layer=False)
        return ExtractionResult(request_id=req.id, document=meta, status=status, error=error)

    try:
        data = req.path.read_bytes()
    except OSError as e:
        return result("failed", error=str(e))
    sha = hashlib.sha256(data).hexdigest()
    for password in [None, *(p.get_secret_value() for p in req.passwords)]:
        try:
            pdf = pdfium.PdfDocument(data, password=password)
            break
        except pdfium.PdfiumError as e:
            if e.err_code != pdfium_c.FPDF_ERR_PASSWORD:
                return result("failed", sha, error=str(e))
    else:
        return result("password_required", sha, encrypted=True)
    if password is not None:
        buf = io.BytesIO()
        pdf.save(buf, flags=pdfium_c.FPDF_REMOVE_SECURITY)
        data = buf.getvalue()
    return Document(metadata=pdf_input.metadata(req.path, password), pdf=store.put(data, "application/pdf"))


# --- tools ---


def inspect_pdf(run: Run, args: InspectPdf) -> str:
    """Page size, rotation, text layer, and image cover of one page, as text."""
    raise NotImplementedError


def read_native_text(run: Run, args: ReadNativeText) -> str:
    """Text layer of one page, or of one box."""
    raise NotImplementedError


def render(run: Run, args: Render) -> Region:
    """PNG of one page at `dpi`, in the store and in `run.regions`."""
    raise NotImplementedError


def crop(run: Run, args: Crop) -> Region:
    """PNG of one box at `dpi`, in the store and in `run.regions`."""
    raise NotImplementedError


def recognize(run: Run, args: Recognize) -> str:
    """`run.recognizer` on one Region: text, HTML, or LaTeX."""
    raise NotImplementedError


def run_tool(run: Run, job: Job, tool: str, args: dict) -> list[dict]:
    """Run one of the 8 tools for `job`; the observation as content parts ({"type": "text"|"image", ...})."""
    raise NotImplementedError


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
