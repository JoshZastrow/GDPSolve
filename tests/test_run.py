"""Tests for process() (docs/process-spec.md). Offline: FakePolicy, FakeRecognizer, and PDFs made here.

Each task adds its tests only in its own `# --- T<n> ---` section.
"""

import io
import threading
import time

import pypdfium2 as pdfium
import pytest

from gdpsolve import run
from gdpsolve.eval import pdf_input
from gdpsolve.schemas import ExtractionRequest, Message


def make_pdf(pages: list[str]) -> bytes:
    """A PDF with one page for each string, in its text layer (Helvetica 24)."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        stream = f"BT /F1 24 Tf 72 700 Td ({esc(text)}) Tj ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def make_scan(text: str) -> bytes:
    """A one-page PDF that is only an image of `text` (no text layer)."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (850, 1100), "white")
    ImageDraw.Draw(image).text((100, 100), text, fill="black", font_size=48)
    buf = io.BytesIO()
    image.save(buf, format="PDF", resolution=100)
    return buf.getvalue()


def make_encrypted(pdf: bytes, password: str) -> bytes:
    """`pdf` encrypted with the user password `password`."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf)))
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakePolicy:
    """Gives the scripted turns in order, one for each step. Records the messages of each step."""

    def __init__(self, turns: list[run.Turn]):
        self.turns = list(turns)
        self.seen: list[list] = []
        self._lock = threading.Lock()

    def step(self, messages: list) -> run.Turn:
        with self._lock:
            self.seen.append(list(messages))
            return self.turns.pop(0)


class FakeRecognizer:
    """Returns f"{mode}:text". Records each (image, mode) call."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, image, mode: str) -> str:
        self.calls.append((image, mode))
        return f"{mode}:text"


def make_run(tmp_path, policy=None, recognizer=None, deadline_s=300) -> run.Run:
    """A Run over a request for a missing file; doc is None until T1 gives a real Document."""
    req = ExtractionRequest(path=tmp_path / "a.pdf", instructions="q", deadline_s=deadline_s)
    return run.Run(None, req, policy or FakePolicy([]), recognizer or FakeRecognizer())


# --- T0 ---


def test_make_pdf_has_readable_text():
    pdf = pdfium.PdfDocument(make_pdf(["Hello", "Page (two)"]))
    assert len(pdf) == 2
    assert [pdf_input.page_text(p) for p in pdf] == ["Hello", "Page (two)"]


def test_make_scan_has_no_text_layer():
    pdf = pdfium.PdfDocument(make_scan("Hello"))
    assert len(pdf) == 1 and pdf_input.page_text(pdf[0]) == ""


def test_make_encrypted_needs_the_password():
    data = make_encrypted(make_pdf(["Secret"]), "pw")
    with pytest.raises(pdfium.PdfiumError):
        pdfium.PdfDocument(data)
    assert pdf_input.page_text(pdfium.PdfDocument(data, password="pw")[0]) == "Secret"


def test_fakes_record_calls():
    turn = run.Turn("report", {"state": "ready", "summary": "s"}, 10, 2)
    policy = FakePolicy([turn])
    assert policy.step(["m"]) == turn and policy.seen == [["m"]]
    rec = FakeRecognizer()
    assert rec("img", "table") == "table:text" and rec.calls == [("img", "table")]


def test_run_state(tmp_path):
    r = make_run(tmp_path, deadline_s=60)
    assert r.jobs.empty() and r.events.empty() and r.thread == [] and r.cost == 0.0
    assert r.reports == {} and r.regions == {} and r.memo == {} and r.waiting == [] and r.final_id == ""
    assert time.time() + 59 < r.deadline <= time.time() + 60


def test_run_post_numbers_messages_and_sends_splits_to_events(tmp_path):
    r = make_run(tmp_path)
    note = r.post(Message(thread="t", job_id="j", kind="note", note="n"))
    split = r.post(Message(thread="t", job_id="j", kind="split", pages_per_job=5))
    assert [m.seq for m in r.thread] == [0, 1] and (note.seq, split.seq) == (0, 1)
    assert r.events.get_nowait() is split and r.events.empty()


def test_run_charge_adds_turn_cost(tmp_path):
    r = make_run(tmp_path)
    assert r.charge(run.Turn("post", {}, 1_000_000, 1_000_000)) == pytest.approx(0.58 + 1.44)
    r.charge(run.Turn("post", {}, 1_000_000, 0))
    assert r.cost == pytest.approx(0.58 * 2 + 1.44)


def test_config():
    assert run.MODEL == "thinkingmachines/Inkling-Small" and run.EFFORT == 0.3 and run.MAX_TOKENS == 4096
    assert run.MAX_COST_PER_DOC == 0.15 and run.OCR_URL.startswith("http")
    assert run.OCR_PROMPTS["handwriting"] == run.OCR_PROMPTS["text"] == "OCR:"
    assert run.OCR_PROMPTS["symbol"] == "Spotting:"


STUBS = [
    ("ingest", lambda r: run.ingest(r.req)),
    ("inspect_pdf", lambda r: run.inspect_pdf(r, None)),
    ("read_native_text", lambda r: run.read_native_text(r, None)),
    ("render", lambda r: run.render(r, None)),
    ("crop", lambda r: run.crop(r, None)),
    ("recognize", lambda r: run.recognize(r, None)),
    ("run_tool", lambda r: run.run_tool(r, None, "inspect_pdf", {"page": 1})),
    ("ensure_ocr_server", lambda r: run.ensure_ocr_server()),
    ("PaddleRecognizer", lambda r: run.PaddleRecognizer()(None, "text")),
    ("TinkerPolicy", lambda r: run.TinkerPolicy().step([])),
    ("plan", lambda r: run.plan(None, r.req)),
    ("react", lambda r: run.react(None, r.req, None, r)),
    ("merge", lambda r: run.merge(None, r.req, {})),
    ("process", lambda r: run.process(r.req, FakePolicy([]), FakeRecognizer())),
    ("final_ended", lambda r: r.final_ended()),
    ("put_merge_job_when_children_ended", lambda r: r.put_merge_job_when_children_ended()),
    ("run_eval", lambda r: run.run_eval([], r.req.path.parent, r.req.path.parent / "results.jsonl")),
]


@pytest.mark.parametrize("name,call", STUBS, ids=[s[0] for s in STUBS])
def test_stub_raises_not_implemented(tmp_path, name, call):
    with pytest.raises(NotImplementedError):
        call(make_run(tmp_path))


# --- T1 ---


# --- T2 ---


# --- T3 ---


# --- T4 ---


def make_job(r: run.Run, max_steps: int = 10, pages=None, job_id: str = "j1"):
    """A Job over a 3-page Document (no real PDF; the T4 tests fake run_tool)."""
    from datetime import datetime

    from gdpsolve.schemas import Document, DocumentMetadata, Job, ObjectRef

    doc = Document(metadata=DocumentMetadata(filename="a.pdf", sha256="0" * 64, page_count=3, encrypted=False,
                                             has_text_layer=True),
                   pdf=ObjectRef(sha256="0" * 64, media_type="application/pdf", size=1))
    return Job(id=job_id, request_id=r.req.id, document=doc, instructions="Find the total.", pages=pages,
               max_steps=max_steps, deadline_at=datetime.fromtimestamp(r.deadline))


def text_of(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


READY = run.Turn("report", {"state": "ready", "summary": "done", "answer": "42"}, 100, 10)
INSPECT = run.Turn("inspect_pdf", {"page": 1}, 100, 10)


@pytest.fixture
def tool_calls(monkeypatch):
    """Fake run_tool: records (job id, tool, args) and returns one text part."""
    calls = []

    def fake(r, job, tool, args):
        calls.append((job.id, tool, args))
        return [{"type": "text", "text": f"{tool} result"}]

    monkeypatch.setattr(run, "run_tool", fake)
    return calls


def test_run_job_report_ready_ends_the_job(tmp_path, tool_calls, monkeypatch):
    steps = []
    monkeypatch.setattr("gdpsolve.trace.call", lambda kind, **f: steps.append((kind, f)))
    policy = FakePolicy([INSPECT, READY, INSPECT])
    r = make_run(tmp_path, policy=policy)
    run.run_job(r, make_job(r))
    report = r.reports["j1"]
    assert report.state == "ready" and report.answer == "42" and report.job_id == "j1"
    assert len(policy.seen) == 2 and policy.turns == [INSPECT]
    assert tool_calls == [("j1", "inspect_pdf", {"page": 1})]
    assert "Find the total." in text_of(policy.seen[0][0])
    assert "inspect_pdf result" in text_of(policy.seen[1][-1])
    assert r.cost == pytest.approx(2 * (100 * 0.58 + 10 * 1.44) / 1e6)
    assert [(k, f["job_id"], f["tool"], f["prompt_tokens"], f["completion_tokens"]) for k, f in steps] == [
        ("step", "j1", "inspect_pdf", 100, 10), ("step", "j1", "report", 100, 10)]


def test_run_job_report_progress_does_not_end_the_job(tmp_path, tool_calls):
    progress = run.Turn("report", {"state": "progress", "summary": "half"}, 1, 1)
    policy = FakePolicy([progress, READY])
    r = make_run(tmp_path, policy=policy)
    run.run_job(r, make_job(r))
    assert len(policy.seen) == 2 and r.reports["j1"].state == "ready"


def test_run_job_tool_error_becomes_an_observation(tmp_path, monkeypatch):
    def broken(r, job, tool, args):
        raise ValueError("page 9 is out of range")

    monkeypatch.setattr(run, "run_tool", broken)
    policy = FakePolicy([INSPECT, READY])
    r = make_run(tmp_path, policy=policy)
    run.run_job(r, make_job(r))
    assert "page 9 is out of range" in text_of(policy.seen[1][-1])
    assert r.reports["j1"].state == "ready"


def test_run_job_bad_report_args_become_an_observation(tmp_path, tool_calls):
    bad = run.Turn("report", {"state": "done"}, 1, 1)
    policy = FakePolicy([bad, READY])
    r = make_run(tmp_path, policy=policy)
    run.run_job(r, make_job(r))
    assert "state" in text_of(policy.seen[1][-1]) and r.reports["j1"].state == "ready"


@pytest.mark.parametrize("limit", ["steps", "deadline", "cost"])
def test_run_job_limit_forces_one_report_turn(tmp_path, tool_calls, limit):
    policy = FakePolicy([INSPECT, INSPECT, READY])
    r = make_run(tmp_path, policy=policy, deadline_s=20 if limit == "deadline" else 300)
    if limit == "cost":
        r.cost = run.MAX_COST_PER_DOC + 0.01
    run.run_job(r, make_job(r, max_steps=2))
    forced_at = 2 if limit == "steps" else 0
    assert len(policy.seen) == forced_at + 1
    assert "report" in text_of(policy.seen[forced_at][-1]).lower()
    assert r.reports["j1"].state == "ready"


def test_run_job_writes_failed_report_when_the_forced_turn_does_not_report(tmp_path, tool_calls):
    policy = FakePolicy([INSPECT, INSPECT])
    r = make_run(tmp_path, policy=policy)
    run.run_job(r, make_job(r, max_steps=1))
    assert len(policy.seen) == 2 and r.reports["j1"].state == "failed"


def test_run_job_shows_other_jobs_messages_but_not_its_own(tmp_path, monkeypatch):
    def posting(r, job, tool, args):
        r.post(Message(thread=r.req.id, job_id="other", kind="note", note="table continues on p. 3"))
        r.post(Message(thread=r.req.id, job_id=job.id, kind="note", note="my own note"))
        return [{"type": "text", "text": "posted"}]

    monkeypatch.setattr(run, "run_tool", posting)
    policy = FakePolicy([INSPECT, INSPECT, READY])
    r = make_run(tmp_path, policy=policy)
    r.post(Message(thread=r.req.id, job_id="earlier", kind="unrelated", pages=[2], note="cover page"))
    run.run_job(r, make_job(r))
    assert "cover page" in text_of(policy.seen[0][0])
    first, second = text_of(policy.seen[1][-1]), text_of(policy.seen[2][-1])
    assert "table continues on p. 3" in first and "my own note" not in first and "cover page" not in first
    assert second.count("table continues on p. 3") == 1  # only the message from the second call


def test_worker_runs_jobs_until_none(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(run, "run_job", lambda r, job: ran.append(job.id))
    r = make_run(tmp_path)
    for job_id in ["a", "b"]:
        r.jobs.put(make_job(r, job_id=job_id))
    r.jobs.put(None)
    t = threading.Thread(target=run.worker, args=(r,), daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive() and ran == ["a", "b"]


# --- T5 ---


# --- T6 ---


# --- T7 ---
