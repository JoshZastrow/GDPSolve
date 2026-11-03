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
        report = self.reports.get(self.final_id)
        return report is not None and report.state in ("ready", "failed")

    def put_merge_job_when_children_ended(self) -> None:
        """Move each waiting job whose `after` jobs have all ended to `jobs`, with their reports in its instructions."""
        for job in list(self.waiting):
            reports = [self.reports.get(i) for i in job.after]
            if not all(r is not None and r.state in ("ready", "failed") for r in reports):
                continue
            lines = "\n".join(f"- job {r.job_id} ({r.state}): {r.summary}" + (f"\n  {r.answer}" if r.answer else "")
                              for r in reports)
            with self.lock:
                self.waiting.remove(job)
            self.jobs.put(job.model_copy(update={"instructions": f"{job.instructions}\n\nReports of the child jobs:\n"
                                                                 f"{lines}\n\nMerge them into the final answer."}))


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
    """`run.recognizer` on one Region: text, a Markdown table, or LaTeX."""
    from PIL import Image

    from gdpsolve.verifier import store

    region = run.regions.get(args.region)
    if region is None:
        return f"unknown region: {args.region}; make one with render or crop"
    return _memo(run, "recognize", args,
                 lambda: run.recognizer(Image.open(io.BytesIO(store.get(region.image))), args.mode))


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
    raise ValueError(f"unknown tool: {tool}")


# --- recognizer ---


LLAMA_LOG = Path("var/log/llama-serve.log")
_warmed: set[str] = set()  # server URLs that ensure_ocr_server has warmed up


def _healthy(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False


def ensure_ocr_server(url: str = OCR_URL, timeout_s: int = 300) -> float:
    """Start the PaddleOCR-VL server if it is not up, then warm it up; returns the warm-up seconds.

    A second call for the same URL does nothing and returns 0.0. The server is never stopped.
    """
    import shutil
    import subprocess
    from urllib.parse import urlparse

    from PIL import Image, ImageDraw

    if url in _warmed:
        return 0.0
    if not _healthy(url):
        llama = shutil.which("llama") or str(Path.home() / ".local/bin/llama")
        cmd = [llama, "serve", "-hf", "PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "--temp", "0",
               "--port", str(urlparse(url).port), "-np", "4", "--alias", "paddleocr-vl"]
        LLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LLAMA_LOG.open("ab") as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        end = time.time() + timeout_s
        while not _healthy(url):
            if time.time() >= end:
                raise RuntimeError(f"OCR server at {url} not healthy after {timeout_s} s; see {LLAMA_LOG}")
            time.sleep(2)
    image = Image.new("RGB", (160, 64), "white")
    ImageDraw.Draw(image).text((16, 12), "warm", fill="black", font_size=32)
    start = time.time()
    PaddleRecognizer(url)(image, "text")
    _warmed.add(url)
    return time.time() - start


def otsl_to_markdown(otsl: str) -> str:
    """PaddleOCR-VL table output (OTSL: <fcel> cell, <ecel> empty cell, <nl> end of row) as a Markdown table.

    Merged cells (<lcel>, <ucel>, <xcel>) become empty cells. Text without OTSL tags is returned as it is.
    """
    import re

    if not re.search(r"<(fcel|ecel)>", otsl):
        return otsl
    rows = []
    for line in otsl.split("<nl>"):
        parts = re.split(r"<(fcel|ecel|lcel|ucel|xcel)>", line)  # [before, tag, text, tag, text, ...]
        cells = [text.strip().replace("|", "\\|") if tag == "fcel" else ""
                 for tag, text in zip(parts[1::2], parts[2::2])]
        if cells:
            rows.append(cells)
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


class PaddleRecognizer:
    """PaddleOCR-VL client for the local llama.cpp server (OpenAI-compatible API)."""

    def __init__(self, url: str = OCR_URL):
        self.url = url

    def __call__(self, image: Image.Image, mode: str) -> str:
        """Text of the first choice for `image` with the prompt OCR_PROMPTS[mode]; tables as Markdown."""
        import base64
        import urllib.request

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {
            "model": "paddleocr-vl",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": OCR_PROMPTS[mode]},
            ]}],
            "temperature": 0,
            "max_tokens": 2048,
        }
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"]
        return otsl_to_markdown(text) if mode == "table" else text


# --- policy ---


class Turn(NamedTuple):
    tool: str
    args: dict
    prompt_tokens: int
    completion_tokens: int


class Policy(Protocol):
    def step(self, messages: list) -> Turn: ...


TOOLS = ["inspect_pdf", "read_native_text", "render", "crop", "recognize", "split", "post", "report"]  # v0 TOOLBOX subset


def tool_specs() -> list[dict]:
    """The v0 tool specs: each TOOLBOX model's docstring and JSON Schema."""
    from gdpsolve.schemas import TOOLBOX

    return [{"name": name, "description": TOOLBOX[name].__doc__.strip(), "parameters": TOOLBOX[name].model_json_schema()}
            for name in TOOLS]


class TinkerPolicy:
    """MODEL with the tml_v0 renderer; the tool specs are the TOOLBOX JSON Schemas.

    `step` takes the cookbook Message dicts that run_job makes (see the worker section). It uses the first tool
    call of the completion. A completion without a tool call becomes a failed `report` with its text as the answer.
    The renderer and the sampling client are made at the first step; the client is shared by the worker threads.
    """

    def __init__(self, model: str = MODEL):
        self.model = model
        self.renderer = None
        self.client = None
        self._lock = threading.Lock()

    def _setup(self) -> None:
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        with self._lock:
            if self.renderer is None:
                self.renderer = renderers.get_renderer("tml_v0", get_tokenizer(self.model))
            if self.client is None:
                self.client = tinker.ServiceClient().create_sampling_client(base_model=self.model)

    def step(self, messages: list) -> Turn:
        """Sample one turn (EFFORT, MAX_TOKENS) and parse its tool call."""
        import json

        import tinker
        from tinker_cookbook import renderers

        if self.renderer is None or self.client is None:
            self._setup()
        prefix = self.renderer.create_conversation_prefix_with_tools(tool_specs())
        prompt = self.renderer.build_generation_prompt(prefix + messages, effort=EFFORT)
        params = tinker.SamplingParams(max_tokens=MAX_TOKENS, stop=self.renderer.get_stop_sequences())
        tokens = self.client.sample(prompt, num_samples=1, sampling_params=params).result().sequences[0].tokens
        message, _ = self.renderer.parse_response(tokens)
        if calls := message.get("tool_calls"):
            tool, args = calls[0].function.name, json.loads(calls[0].function.arguments)
        else:
            tool = "report"
            args = {"state": "failed", "summary": "no tool call", "answer": renderers.get_text_content(message)}
        return Turn(tool, args, prompt.length, len(tokens))


# --- worker ---


# Policy messages are tinker_cookbook Message dicts:
#   {"role": "user", "content": [text parts]}                     first message, and the forced-report prompt
#   {"role": "assistant", "content": "", "tool_calls": [{"id", "type": "function", "function": {"name", "arguments"}}]}
#   {"role": "tool", "tool_call_id", "name", "content": [text/image parts]}   tool result, Thread news, budget left
# Image parts are {"type": "image", "image": PIL.Image or data URL}, as run_tool returns them.


def _thread_text(msgs: list[Message]) -> str:
    return "\n".join(f"[{m.seq}] {m.kind} from job {m.job_id}" + (f" pages {m.pages}" if m.pages else "")
                     + (f": {m.note}" if m.note else "") for m in msgs)


def _news(run: Run, job: Job, seen: int) -> tuple[list[Message], int]:
    """Thread messages after index `seen` from other jobs, and the new `seen`."""
    with run.lock:
        new = run.thread[seen:]
    return [m for m in new if m.job_id != job.id], seen + len(new)


def _budget(run: Run, job: Job, steps: int) -> str:
    return (f"Budget left: {job.max_steps - steps} steps, {max(0, int(run.deadline - 30 - time.time()))} s, "
            f"${max(0.0, MAX_COST_PER_DOC - run.cost):.3f}.")


def _limit(run: Run, job: Job, steps: int) -> str:
    if steps >= job.max_steps:
        return f"You used all {job.max_steps} steps."
    if time.time() >= run.deadline - 30:
        return "The deadline is in less than 30 s."
    if run.cost > MAX_COST_PER_DOC:
        return f"The request cost is more than ${MAX_COST_PER_DOC}."
    return ""


def _report(run: Run, job: Job, args: dict) -> str:
    # The report tool, done here and not in run_tool: it is part of the job loop. It writes the job's latest report.
    report = Report.model_validate({**args, "job_id": job.id})
    with run.lock:
        run.reports[job.id] = report
    return "reported"


def _ended(run: Run, job: Job) -> bool:
    report = run.reports.get(job.id)
    return report is not None and report.state in ("ready", "failed")


def run_job(run: Run, job: Job) -> None:
    """Ask the policy for turns and run their tools until a ready/failed report or a limit."""
    import json

    from gdpsolve import trace

    pages = job.pages or list(range(1, job.document.metadata.page_count + 1))
    news, seen = _news(run, job, 0)
    first = (f"Instructions: {job.instructions}\n\nDocument: {job.document.metadata.filename}, "
             f"{job.document.metadata.page_count} pages. Your job ({job.kind}) covers pages {pages}.\n"
             f"{_budget(run, job, 0)}\nUse the tools to read the pages. End with `report` (state ready or failed).")
    if news:
        first += f"\n\nThread so far:\n{_thread_text(news)}"
    messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": first}]}]
    steps = 0
    while True:
        limit = _limit(run, job, steps)
        if limit:
            messages.append({"role": "user", "content": [{"type": "text", "text":
                             f"{limit} Call `report` now with state ready or failed and your best answer."}]})
        try:
            turn = run.policy.step(messages)
        except Exception as e:  # noqa: BLE001 -- a sampling error ends this job, not the worker thread
            _report(run, job, {"state": "failed", "summary": "policy error", "error": f"{type(e).__name__}: {e}"})
            return
        run.charge(turn)
        trace.call("step", job_id=job.id, tool=turn.tool, prompt_tokens=turn.prompt_tokens,
                   completion_tokens=turn.completion_tokens)
        steps += 1
        try:
            if turn.tool == "report":
                parts = [{"type": "text", "text": _report(run, job, turn.args)}]
            else:
                parts = run_tool(run, job, turn.tool, turn.args)
        except Exception as e:  # noqa: BLE001 -- the model sees the error and can try again
            parts = [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}]
        if (turn.tool == "report" and _ended(run, job)) or limit:
            break
        news, seen = _news(run, job, seen)
        if news:
            parts.append({"type": "text", "text": f"New Thread messages:\n{_thread_text(news)}"})
        parts.append({"type": "text", "text": _budget(run, job, steps)})
        call_id = f"call_{steps}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": turn.tool, "arguments": json.dumps(turn.args)}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "name": turn.tool, "content": parts})
    if not _ended(run, job):
        _report(run, job, {"state": "failed", "summary": "no ready report before the limit"})


def worker(run: Run) -> None:
    """Run jobs from `run.jobs` until a None stop signal."""
    while True:
        job = run.jobs.get()
        if job is None:
            return
        run_job(run, job)


# --- router ---


MIN_PAGES_PER_CHILD = 5
MAX_CHILDREN = 16
MIN_SPLIT_TIME_S = 120


def _max_steps(page_count: int) -> int:
    return min(40, 8 + page_count // 5)


def plan(doc: Document, req: ExtractionRequest) -> Job:
    """One root job for all pages; max_steps = min(40, 8 + page_count // 5)."""
    from datetime import datetime, timedelta

    return Job(request_id=req.id, document=doc, instructions=req.instructions,
               max_steps=_max_steps(doc.metadata.page_count),
               deadline_at=datetime.now() + timedelta(seconds=req.deadline_s))


def react(doc: Document, req: ExtractionRequest, msg: Message | None, run: Run) -> list[Job]:
    """For a split message: child jobs, and a merge job put in `run.waiting`; else [].

    Only the final job may split, and only once (the merge job id ends in ".merge"). The children cover the split's
    pages, or all pages, in equal runs of at least MIN_PAGES_PER_CHILD pages; at most MAX_CHILDREN. A split with
    less than MIN_SPLIT_TIME_S left, or too few pages for two children, gives [].
    """
    from datetime import datetime
    from itertools import pairwise

    if (msg is None or msg.kind != "split" or msg.job_id != run.final_id or run.final_id.endswith(".merge")
            or run.deadline - time.time() < MIN_SPLIT_TIME_S):
        return []
    pages = msg.pages or list(range(1, doc.metadata.page_count + 1))
    size = max(MIN_PAGES_PER_CHILD, msg.pages_per_job or MIN_PAGES_PER_CHILD)
    n = min(MAX_CHILDREN, len(pages) // size)
    if n < 2:
        return []
    deadline_at = datetime.fromtimestamp(run.deadline)
    bounds = [len(pages) * i // n for i in range(n + 1)]
    children = [Job(id=f"{msg.job_id}.{i}", request_id=req.id, parent_id=msg.job_id, document=doc,
                    instructions=req.instructions, pages=pages[lo:hi], max_steps=_max_steps(hi - lo),
                    deadline_at=deadline_at)
                for i, (lo, hi) in enumerate(pairwise(bounds))]
    merge_job = Job(id=f"{msg.job_id}.merge", request_id=req.id, parent_id=msg.job_id, kind="merge", document=doc,
                    instructions=req.instructions, after=[c.id for c in children],
                    max_steps=_max_steps(doc.metadata.page_count), deadline_at=deadline_at)
    with run.lock:
        run.waiting.append(merge_job)
        run.final_id = merge_job.id
    return children


def merge(doc: Document, req: ExtractionRequest, reports: dict[str, Report], final_id: str,
          root_id: str) -> ExtractionResult:
    """The result from the ready reports, without a model call.

    The final job's ready report, else the root job's, else the ready answers joined ("partial"), else "failed".
    """
    ready = [r for r in reports.values() if r.state == "ready"]
    by_id = {r.job_id: r for r in ready}
    if final_id in by_id or root_id in by_id:
        best = by_id.get(final_id) or by_id[root_id]
        status, answer, fields = "ok", best.answer, best.fields
    elif ready:
        status = "partial"
        answer = "\n\n".join(r.answer for r in ready if r.answer)
        fields = [f for r in ready for f in r.fields]
    else:
        return ExtractionResult(request_id=req.id, document=doc.metadata, status="failed",
                                error="; ".join(r.error or r.summary for r in reports.values()) or "no report")
    return ExtractionResult(request_id=req.id, document=doc.metadata, status=status, answer=answer, fields=fields,
                            confidence=min((f.confidence for f in fields), default=0.0))


# --- process ---


def process(req: ExtractionRequest, policy: Policy | None = None,
            recognizer: Callable[[Image.Image, str], str] | None = None, workers: int = 4) -> ExtractionResult:
    """Ingest, run the router and the worker threads until the final job ends or the deadline, then merge.

    The OCR server is started only for the default recognizer.
    """
    doc = ingest(req)
    if not isinstance(doc, Document):
        return doc
    if recognizer is None:
        ensure_ocr_server()
        recognizer = PaddleRecognizer()
    run = Run(doc, req, policy or TinkerPolicy(), recognizer)
    root = plan(doc, req)
    run.final_id = root.id
    threads = [threading.Thread(target=worker, args=(run,), daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    run.jobs.put(root)
    # A job posts its split before it reports, so a non-empty `events` means a split is still to come.
    while time.time() < run.deadline and not (run.final_ended() and run.events.empty()):
        try:
            msg = run.events.get(timeout=1)
        except queue.Empty:
            msg = None
        for job in react(doc, req, msg, run):
            run.jobs.put(job)
        run.put_merge_job_when_children_ended()
    for _ in threads:
        run.jobs.put(None)
    end = time.time() + 1
    for t in threads:
        t.join(timeout=max(0.0, end - time.time()))
    return merge(doc, req, run.reports, run.final_id, root.id)


# --- eval ---


class _Counted:
    """Wraps a policy: counts the steps, the jobs (first steps), and the cost of one request."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self.steps = self.jobs = 0
        self.cost = 0.0
        self._lock = threading.Lock()

    def step(self, messages: list) -> Turn:
        turn = self.policy.step(messages)
        with self._lock:
            self.steps += 1
            self.jobs += len(messages) == 1
            self.cost += (turn.prompt_tokens * PRICES["prefill"] + turn.completion_tokens * PRICES["sample"]) / 1e6
        return turn


def run_eval(tasks: list[dict], data: Path, results: Path, policy: Policy | None = None,
             recognizer: Callable[[Image.Image, str], str] | None = None, workers: int = 4) -> list[dict]:
    """process() for each GDP.pdf task; append one baseline-format row (+ steps, cost, jobs) to `results`.

    Rows are ExtractionResult plus model, effort, truncated_at_page (always None), steps, cost, jobs; each is
    written when its task ends. The caller picks the tasks (baseline.pending for resume). Returns the new rows.
    """
    from gdpsolve import trace
    from gdpsolve.schemas import DocumentMetadata

    if recognizer is None:
        ensure_ocr_server()
        recognizer = PaddleRecognizer()
    policy = policy or TinkerPolicy()
    rows = []
    for task in tasks:
        path = data / task["pdf_path"]
        counted = _Counted(policy)
        start = time.monotonic()
        try:
            result = process(ExtractionRequest(id=task["task_id"], path=path, instructions=task["prompt"]),
                             counted, recognizer, workers)
        except Exception as e:  # noqa: BLE001 -- recorded as a failed row; resume retries it
            meta = DocumentMetadata(filename=path.name, sha256="", page_count=0, encrypted=False,
                                    has_text_layer=False)
            result = ExtractionResult(request_id=task["task_id"], document=meta, status="failed",
                                      error=f"{type(e).__name__}: {e}")
        result.latency_s = round(time.monotonic() - start, 2)
        row = result.model_dump(mode="json") | {"model": MODEL, "effort": EFFORT, "truncated_at_page": None,
                                                "steps": counted.steps, "cost": counted.cost, "jobs": counted.jobs}
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        trace.call("process", id=result.request_id, status=result.status, steps=counted.steps, cost=counted.cost,
                   jobs=counted.jobs, latency_s=result.latency_s, error=result.error)
        print(f"{result.request_id} {result.status} {result.latency_s}s ${counted.cost:.3f}", flush=True)
        rows.append(row)
    return rows
