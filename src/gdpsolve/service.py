"""Document service (docs/service-spec.md): Reducto-shaped parse, extract, split, and classify over HTTP.

The router (the HTTP handler) puts each request on the job queue as one Job and waits for it. Worker threads
take jobs from the queue. Jobs share only the object store (var/objects) and their results (var/jobs).
tinker and tinker_cookbook are imported inside functions.
"""

import io
import json
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from gdpsolve import run
from gdpsolve.eval import baseline, pdf_input
from gdpsolve.schemas import ObjectRef
from gdpsolve.verifier import store

JOBS = Path("var/jobs")
DPI = 100  # same as pdf_input.content
WHOLE_PAGE = {"left": 0, "top": 0, "width": 1, "height": 1}


class BadInput(ValueError):
    """The request is wrong; the service answers 422."""


class Tinker:
    """The model through the cookbook renderer, as in eval/baseline.py, and one Tinker sampling client.

    The renderer and the client are made at first use; the client is shared by the worker threads. With a
    checkpoint, the client samples from that Tinker path instead of the base model.
    """

    def __init__(self, model: str, checkpoint: str | None, effort: float, max_tokens: int):
        self.model, self.checkpoint, self.effort, self.max_tokens = model, checkpoint, effort, max_tokens
        self.renderer = None
        self.client = None
        self._lock = threading.Lock()

    def _renderer(self):
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        with self._lock:
            if self.renderer is None:
                name = model_info.get_recommended_renderer_name(self.model)
                self.renderer = renderers.get_renderer(name, get_tokenizer(self.model))
        return self.renderer

    def count_tokens(self, parts: list[dict]) -> int:
        """Tokens of a user message with `parts`, counted as baseline.prepare counts them."""
        return self._renderer().build_generation_prompt([{"role": "user", "content": parts}]).length

    def prompt(self, content: list[dict]):
        """The ModelInput for one user message with `content`."""
        return self._renderer().build_generation_prompt([{"role": "user", "content": content}], effort=self.effort)

    def sample(self, content: list[dict]) -> dict:
        """One completion: {text, finish ("stop" or "length"), prompt_tokens, completion_tokens, prompt_sha256}."""
        import tinker

        renderer, prompt = self._renderer(), self.prompt(content)
        with self._lock:
            if self.client is None:
                service = tinker.ServiceClient()
                self.client = (service.create_sampling_client(model_path=self.checkpoint) if self.checkpoint
                               else service.create_sampling_client(base_model=self.model))
        params = tinker.SamplingParams(max_tokens=self.max_tokens, stop=renderer.get_stop_sequences())
        tokens = self.client.sample(prompt, num_samples=1, sampling_params=params).result().sequences[0].tokens
        text, clean = baseline.answer_text(renderer, tokens)
        return {"text": text, "finish": "stop" if clean else "length", "prompt_tokens": prompt.length,
                "completion_tokens": len(tokens), "prompt_sha256": baseline.store_prompt(renderer, prompt)}


@dataclass
class Config:
    """Server settings. `sampler` has count_tokens(parts) and sample(content), as Tinker; `recognizer` is
    called as recognizer(image, mode), as run.PaddleRecognizer. Tests give fakes."""

    parser: str = "native"  # native | paddle
    model: str = run.MODEL
    checkpoint: str | None = None
    effort: float = 0.3
    max_tokens: int = 8192
    token_budget: int = 120_000
    sampler: Any = None
    recognizer: Any = None

    def __post_init__(self):
        if self.sampler is None:
            self.sampler = Tinker(self.model, self.checkpoint, self.effort, self.max_tokens)
        if self.parser == "paddle" and self.recognizer is None:
            self.recognizer = run.PaddleRecognizer()

    def settings(self) -> dict:
        return {"parser": self.parser, "model": self.model, "checkpoint": self.checkpoint, "effort": self.effort,
                "max_tokens": self.max_tokens, "token_budget": self.token_budget}


# --- jobs ---


def handle(config: Config, kind: str, payload, job_id: str | None = None) -> dict:
    """Do one job and return its response; a parse, extract, split, or classify result is also saved in var/jobs."""
    if kind == "upload":
        return {"file_id": "gdpsolve://" + store.put(payload, "application/octet-stream").sha256, "presigned_url": None}
    if not isinstance(payload, dict):
        raise BadInput("the body must be a JSON object")
    job_id = job_id or uuid.uuid4().hex
    start = time.monotonic()
    out = HANDLERS[kind](config, payload)
    out = {"response_type": kind, "job_id": job_id, "duration": round(time.monotonic() - start, 3)} | out
    JOBS.mkdir(parents=True, exist_ok=True)
    (JOBS / f"{job_id}.json").write_text(json.dumps(out))
    return out


def _input(payload: dict) -> str:
    if not isinstance(payload.get("input"), str):
        raise BadInput("input is required: a gdpsolve:// file id or a jobid:// parse job")
    return payload["input"]


def _page_numbers(page_range, count: int) -> list[int]:
    """1-indexed pages: all for None, start..end for a dict, the listed pages for a list."""
    if page_range is None:
        wanted = range(1, count + 1)
    elif isinstance(page_range, dict):
        wanted = range(page_range.get("start", 1), page_range.get("end", count) + 1)
    elif isinstance(page_range, list):
        wanted = sorted(set(page_range))
    else:
        raise BadInput("page_range must be {start, end}, a list of pages, or null")
    return [n for n in wanted if 1 <= n <= count]


def _chunks(config: Config, payload: dict) -> list[dict]:
    """The chunks of a jobid:// parse job, or of a new parse of a file id."""
    source = _input(payload)
    if source.startswith("jobid://"):
        job_id = source.removeprefix("jobid://")
        if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not (JOBS / f"{job_id}.json").exists():
            raise BadInput(f"unknown job: {source}")
        job = json.loads((JOBS / f"{job_id}.json").read_text())
        if job["response_type"] != "parse":
            raise BadInput(f"{source} is a {job['response_type']} job, not a parse job")
        return job["result"]["chunks"]
    return parse(config, payload)["result"]["chunks"]


def content(chunks: list[dict], text: str, count_tokens, budget: int) -> tuple[list[dict], int | None]:
    """User message content: `text`, then the chunks' pages as baseline.prepare gives them, until `budget` tokens.

    Returns (content, truncated_at_page).
    """
    from PIL import Image

    def pages():
        for chunk in chunks:
            block = chunk["blocks"][0]
            image = None
            if block["type"] == "Figure":
                sha = block["image_url"].removeprefix("gdpsolve://")
                image = Image.open(io.BytesIO(store.get(ObjectRef(sha256=sha, media_type="image/png", size=0))))
            yield block["bbox"]["page"], block["content"], image
    parts, truncated_at = pdf_input.parts(pages(), count_tokens, budget)
    return [{"type": "text", "text": text}, *parts], truncated_at


# --- endpoints ---


def parse(config: Config, payload: dict) -> dict:
    """One chunk with one block for each page. A page with a text layer is a Text block. A page without one is
    a Figure block with the page PNG (native), or a Text block with its OCR text (paddle)."""
    import pypdfium2 as pdfium

    source = _input(payload)
    sha = source.removeprefix("gdpsolve://")
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or not (store.ROOT / sha).exists():
        raise BadInput(f"unknown file id: {source}")
    data = store.get(ObjectRef(sha256=sha, media_type="application/pdf", size=0))
    settings = payload.get("settings") or {}
    with run._PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as e:
            raise BadInput(f"{source} is not a PDF that opens without a password: {e}") from None
        pages = []
        for n in _page_numbers(settings.get("page_range"), len(pdf)):
            page = pdf[n - 1]
            text = pdf_input.page_text(page)
            pages.append((n, text, None if text else page.render(scale=DPI / 72).to_pil().copy()))
            page.close()
        pdf.close()  # inside the lock: pdfium objects closed by the garbage collector would race other threads
    chunks = []
    for n, text, image in pages:
        block = {"type": "Text", "bbox": WHOLE_PAGE | {"page": n}, "content": text, "image_url": None}
        if image is not None and config.parser == "paddle":
            block["content"] = config.recognizer(image, "text")
        elif image is not None:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            block |= {"type": "Figure", "image_url": "gdpsolve://" + store.put(buf.getvalue(), "image/png").sha256}
        chunks.append({"content": block["content"], "blocks": [block]})
    return {"usage": {"num_pages": len(chunks)}, "result": {"type": "full", "chunks": chunks}}


def extract(config: Config, payload: dict) -> dict:
    """One sample over the system prompt and the pages, as baseline.prepare builds it.

    An empty schema gives [{"answer": text}]. A schema adds one line to the system prompt and gives the parsed
    JSON; JSON that does not parse gives [{"answer": text}] and confidence "low".
    """
    instructions = payload.get("instructions") or {}
    schema = instructions.get("schema") or {}
    text = instructions.get("system_prompt", "")
    if schema:
        text += f"\nReturn only JSON matching this schema: {json.dumps(schema)}"
    chunks = _chunks(config, payload)
    message, truncated_at = content(chunks, text, config.sampler.count_tokens, config.token_budget)
    s = config.sampler.sample(message)
    out = {"usage": {"num_pages": len(chunks), "prompt_tokens": s["prompt_tokens"],
                     "completion_tokens": s["completion_tokens"],
                     "cost_usd": baseline.cost(config.model, s["prompt_tokens"], s["completion_tokens"])},
           "result": [{"answer": s["text"]}], "truncated_at_page": truncated_at, "finish": s["finish"],
           "prompt_sha256": s["prompt_sha256"]}
    if schema:
        try:
            out["result"] = [json.loads(s["text"])]
        except json.JSONDecodeError:
            out["confidence"] = "low"
    return out


def split(config: Config, payload: dict) -> dict:
    """One sample for each page: which of the sections the page belongs to, as a JSON list of names."""
    sections = payload.get("split_description")
    if not sections:
        raise BadInput("split_description is required: [{name, description}]")
    listing = "\n".join(f"{s['name']}: {s.get('description', '')}" for s in sections)
    ask = (f"Sections:\n{listing}\n\nWhich of these sections does this page belong to? "
           "Return only a JSON list of section names.")
    pages: dict[str, list[int]] = {s["name"]: [] for s in sections}
    for chunk in _chunks(config, payload):
        message, _ = content([chunk], ask, config.sampler.count_tokens, config.token_budget)
        try:
            names = json.loads(config.sampler.sample(message)["text"])
        except json.JSONDecodeError:
            names = []
        for name in names if isinstance(names, list) else []:
            if name in pages:
                pages[name].append(chunk["blocks"][0]["bbox"]["page"])
    return {"result": {"splits": [{"name": name, "pages": p, "conf": "high"} for name, p in pages.items() if p]}}


def classify(config: Config, payload: dict) -> dict:
    """One sample over the pages in page_range (default: the first 5): the name of the category."""
    schema = payload.get("classification_schema")
    if not schema:
        raise BadInput("classification_schema is required: [{category, criteria}]")
    listing = "\n".join(f"{c['category']}: {'; '.join(c.get('criteria', []))}" for c in schema)
    ask = f"Categories:\n{listing}\n\nWhich category is this document? Return only the category name."
    settings = {"page_range": payload.get("page_range", {"start": 1, "end": 5})}
    chunks = _chunks(config, {"input": _input(payload), "settings": settings})
    message, _ = content(chunks, ask, config.sampler.count_tokens, config.token_budget)
    return {"result": {"category": config.sampler.sample(message)["text"].strip()}}


HANDLERS = {"parse": parse, "extract": extract, "split": split, "classify": classify}


# --- router and workers ---


@dataclass
class Job:
    kind: str
    payload: Any
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: Exception | None = None


class Router(BaseHTTPRequestHandler):
    """POST /upload, /parse, /extract, /split, /classify and GET /health. Each POST is one Job on the queue."""

    server: "Server"

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != "/health":
            return self._send(404, {"detail": f"no route {self.path}"})
        s = self.server
        self._send(200, {"ok": True, "workers": s.workers, "queued": s.jobs.qsize(), "config": s.config.settings()})

    def do_POST(self) -> None:
        kind = self.path.strip("/")
        if kind != "upload" and kind not in HANDLERS:
            return self._send(404, {"detail": f"no route {self.path}"})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = body if kind == "upload" else json.loads(body)
        except json.JSONDecodeError as e:
            return self._send(422, {"detail": f"body is not JSON: {e}"})
        job = Job(kind, payload)
        self.server.jobs.put(job)
        job.done.wait()
        if isinstance(job.error, BadInput):
            self._send(422, {"detail": str(job.error)})
        elif job.error is not None:
            self._send(500, {"detail": f"{type(job.error).__name__}: {job.error}"})
        else:
            self._send(200, job.result)


class Server(ThreadingHTTPServer):
    """The HTTP router and `workers` worker threads on one job queue; `workers` is the concurrency cap."""

    daemon_threads = True

    def __init__(self, config: Config, host: str, port: int, workers: int):
        super().__init__((host, port), Router)
        self.config, self.workers, self.jobs = config, workers, queue.Queue()
        for _ in range(workers):
            threading.Thread(target=self.worker, daemon=True).start()

    def worker(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                job.result = handle(self.config, job.kind, job.payload, job.id)
            except Exception as e:  # noqa: BLE001 -- the router answers 422 or 500
                job.error = e
            job.done.set()

    def stop(self) -> None:
        """Stop the workers after the jobs already queued."""
        for _ in range(self.workers):
            self.jobs.put(None)


def server(config: Config, host: str = "127.0.0.1", port: int = 8787, workers: int = 8) -> Server:
    return Server(config, host, port, workers)


def serve(config: Config, host: str = "127.0.0.1", port: int = 8787, workers: int = 8) -> None:
    """Serve until stopped."""
    server(config, host, port, workers).serve_forever()
