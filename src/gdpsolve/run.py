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
    """Hash the file, try the passwords, read the metadata, and store the bytes; or a failed/password_required result."""
    raise NotImplementedError


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
