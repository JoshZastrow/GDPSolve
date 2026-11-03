"""Tests for process() (docs/process-spec.md). Offline: FakePolicy, FakeRecognizer, and PDFs made here.

Each task adds its tests only in its own `# --- T<n> ---` section.
"""

import hashlib
import io
import threading
import time

import pypdfium2 as pdfium
import pytest
from pydantic import SecretStr

from gdpsolve import run
from gdpsolve.eval import pdf_input
from gdpsolve.schemas import Document, ExtractionRequest, ExtractionResult, Message


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
    ("inspect_pdf", lambda r: run.inspect_pdf(r, None)),
    ("read_native_text", lambda r: run.read_native_text(r, None)),
    ("render", lambda r: run.render(r, None)),
    ("crop", lambda r: run.crop(r, None)),
    ("recognize", lambda r: run.recognize(r, None)),
    ("run_tool", lambda r: run.run_tool(r, None, "inspect_pdf", {"page": 1})),
    ("ensure_ocr_server", lambda r: run.ensure_ocr_server()),
    ("PaddleRecognizer", lambda r: run.PaddleRecognizer()(None, "text")),
    ("TinkerPolicy", lambda r: run.TinkerPolicy().step([])),
    ("run_job", lambda r: run.run_job(r, None)),
    ("worker", lambda r: run.worker(r)),
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


def ingest_file(tmp_path, monkeypatch, data: bytes, passwords=()):
    """ingest() of `data` written to tmp_path/a.pdf, with the store in tmp_path/var/objects."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.pdf").write_bytes(data)
    return run.ingest(ExtractionRequest(path=tmp_path / "a.pdf", instructions="q", passwords=list(passwords)))


def test_ingest_plain_pdf(tmp_path, monkeypatch):
    from gdpsolve.verifier import store

    data = make_pdf(["One", "Two", "Three"])
    doc = ingest_file(tmp_path, monkeypatch, data)
    assert isinstance(doc, Document)
    assert doc.metadata.sha256 == hashlib.sha256(data).hexdigest()
    assert doc.metadata.page_count == 3 and doc.metadata.filename == "a.pdf"
    assert not doc.metadata.encrypted and doc.metadata.has_text_layer
    assert store.get(doc.pdf) == data and doc.pdf.media_type == "application/pdf"


def test_ingest_right_password_opens_encrypted_pdf(tmp_path, monkeypatch):
    from gdpsolve.verifier import store

    data = make_encrypted(make_pdf(["Secret"]), "pw")
    doc = ingest_file(tmp_path, monkeypatch, data, [SecretStr("nope"), SecretStr("pw")])
    assert isinstance(doc, Document)
    assert doc.metadata.encrypted and doc.metadata.page_count == 1
    assert doc.metadata.sha256 == hashlib.sha256(data).hexdigest()
    stored = pdfium.PdfDocument(store.get(doc.pdf))  # decrypted: opens without the password
    assert pdf_input.page_text(stored[0]) == "Secret"


@pytest.mark.parametrize("passwords", [[], [SecretStr("nope")]], ids=["none", "wrong"])
def test_ingest_wrong_password_gives_password_required(tmp_path, monkeypatch, passwords):
    data = make_encrypted(make_pdf(["Secret"]), "pw")
    result = ingest_file(tmp_path, monkeypatch, data, passwords)
    assert isinstance(result, ExtractionResult) and result.status == "password_required"
    assert result.document.encrypted and result.document.sha256 == hashlib.sha256(data).hexdigest()


def test_ingest_corrupt_file_fails(tmp_path, monkeypatch):
    result = ingest_file(tmp_path, monkeypatch, b"not a pdf")
    assert isinstance(result, ExtractionResult) and result.status == "failed" and result.error


def test_ingest_missing_file_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run.ingest(ExtractionRequest(path=tmp_path / "missing.pdf", instructions="q"))
    assert isinstance(result, ExtractionResult) and result.status == "failed" and result.error


# --- T2 ---


# --- T3 ---


# --- T4 ---


# --- T5 ---


# --- T6 ---


# --- T7 ---
