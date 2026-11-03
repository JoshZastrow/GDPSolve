"""Tests for process() (docs/process-spec.md). Offline: FakePolicy, FakeRecognizer, and PDFs made here.

Each task adds its tests only in its own `# --- T<n> ---` section.
"""

import hashlib
import io
import os
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


def doc_run(tmp_path, monkeypatch, pdf: bytes, recognizer=None) -> run.Run:
    """A Run over `pdf`, with the store in tmp_path."""
    from gdpsolve.schemas import Document, DocumentMetadata
    from gdpsolve.verifier import store

    monkeypatch.chdir(tmp_path)
    ref = store.put(pdf, "application/pdf")
    meta = DocumentMetadata(filename="a.pdf", sha256=ref.sha256, page_count=len(pdfium.PdfDocument(pdf)),
                            encrypted=False, has_text_layer=True)
    r = make_run(tmp_path, recognizer=recognizer)
    r.doc = Document(metadata=meta, pdf=ref)
    return r


def test_read_native_text_is_the_page_text(tmp_path, monkeypatch):
    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello", "Total due 42"]))
    assert run.read_native_text(r, run.ReadNativeText(page=2)) == "Total due 42"


def test_read_native_text_in_a_box(tmp_path, monkeypatch):
    from gdpsolve.schemas import BoundingBox

    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello"]))
    # boxes are in points from the top left; the text baseline is at 792 - 700 = 92
    inside = BoundingBox(page=1, x0=60, y0=60, x1=300, y1=110)
    outside = BoundingBox(page=1, x0=60, y0=400, x1=300, y1=500)
    assert run.read_native_text(r, run.ReadNativeText(page=1, bbox=inside)) == "Hello"
    assert run.read_native_text(r, run.ReadNativeText(page=1, bbox=outside)) == ""


def test_inspect_pdf_text_page(tmp_path, monkeypatch):
    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello"]))
    out = run.inspect_pdf(r, run.InspectPdf(page=1))
    assert "612 x 792" in out and "rotation 0" in out
    assert "has_text_layer=True" in out and "image_cover=0.00" in out


def test_inspect_pdf_scan_has_no_text_layer(tmp_path, monkeypatch):
    r = doc_run(tmp_path, monkeypatch, make_scan("Hello"))
    out = run.inspect_pdf(r, run.InspectPdf(page=1))
    assert "has_text_layer=False" in out and "image_cover=1.00" in out


def test_render_gives_a_region_with_a_png_in_the_store(tmp_path, monkeypatch):
    from PIL import Image

    from gdpsolve.verifier import store

    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello"]))
    region = run.render(r, run.Render(page=1, dpi=72))
    assert region.page == 1 and region.dpi == 72 and region.bbox is None and region.ops == []
    assert region.doc_sha256 == r.doc.metadata.sha256 and region.image.media_type == "image/png"
    assert r.regions[region.id] == region
    image = Image.open(io.BytesIO(store.get(region.image)))
    assert image.format == "PNG" and image.size == (612, 792)
    assert region.id != run.render(r, run.Render(page=1, dpi=100)).id


def test_crop_gives_a_padded_region(tmp_path, monkeypatch):
    from PIL import Image

    from gdpsolve.schemas import BoundingBox
    from gdpsolve.verifier import store

    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello"]))
    box = BoundingBox(page=1, x0=100, y0=100, x1=200, y1=150)
    region = run.crop(r, run.Crop(page=1, bbox=box, padding=10, dpi=144))
    assert (region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1) == (90, 90, 210, 160)
    assert Image.open(io.BytesIO(store.get(region.image))).size == (240, 140)
    assert r.regions[region.id] == region


def test_same_arguments_make_one_memo_entry(tmp_path, monkeypatch):
    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello"]))
    assert run.read_native_text(r, run.ReadNativeText(page=1)) == "Hello"
    assert run.read_native_text(r, run.ReadNativeText(page=1)) == "Hello"
    assert len(r.memo) == 1
    a = run.render(r, run.Render(page=1, dpi=72))
    b = run.render(r, run.Render(page=1, dpi=72))
    assert a == b and len(r.memo) == 2 and len(r.regions) == 1


def test_run_tool_gives_content_parts(tmp_path, monkeypatch):
    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello"]))
    assert run.run_tool(r, None, "read_native_text", {"page": 1}) == [{"type": "text", "text": "Hello"}]
    parts = run.run_tool(r, None, "render", {"page": 1, "dpi": 72})
    assert [p["type"] for p in parts] == ["text", "image"]
    assert next(iter(r.regions)) in parts[0]["text"] and parts[1]["image"].size == (612, 792)
    with pytest.raises(ValueError):
        run.run_tool(r, None, "detect_layout", {"page": 1})


def test_run_tool_split_and_post_go_to_the_thread(tmp_path, monkeypatch):
    from gdpsolve.schemas import Document, DocumentMetadata, Job, ObjectRef

    r = make_run(tmp_path)
    meta = DocumentMetadata(filename="a.pdf", sha256="0" * 64, page_count=1, encrypted=False, has_text_layer=True)
    doc = Document(metadata=meta, pdf=ObjectRef(sha256="0" * 64, media_type="application/pdf", size=1))
    job = Job(id="j1", request_id=r.req.id, document=doc, instructions="q", deadline_at="2030-01-01T00:00:00Z")
    assert run.run_tool(r, job, "post", {"kind": "note", "pages": [3], "note": "n"}) == [
        {"type": "text", "text": "posted"}]
    assert run.run_tool(r, job, "split", {"pages_per_job": 5}) == [{"type": "text", "text": "split requested"}]
    note, split = r.thread
    assert (note.thread, note.job_id, note.kind, note.pages, note.note) == (r.req.id, "j1", "note", [3], "n")
    assert (split.kind, split.pages_per_job) == ("split", 5) and r.events.get_nowait() is split


# --- T3 ---


class FakeOcrServer:
    """A llama-server stand-in on a free port: GET /health and POST /v1/chat/completions."""

    def __init__(self, health: int = 200, content: str = "warm", port: int = 0):
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.health, self.content, self.bodies = health, content, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code: int, body: dict):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply(fake.health, {"status": "ok" if fake.health == 200 else "loading"})

            def do_POST(self):
                fake.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.reply(200, {"choices": [{"message": {"role": "assistant", "content": fake.content}}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ocr_server():
    server = FakeOcrServer()
    yield server
    server.close()


def free_url() -> str:
    """A URL where nothing listens."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def image_of(text: str):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (200, 60), "white")
    ImageDraw.Draw(image).text((10, 10), text, fill="black", font_size=32)
    return image


@pytest.mark.parametrize("mode,prompt", list(run.OCR_PROMPTS.items()))
def test_each_mode_gives_its_prompt(ocr_server, mode, prompt):
    run.PaddleRecognizer(ocr_server.url)(image_of("x"), mode)
    assert ocr_server.bodies[0]["messages"][0]["content"][-1] == {"type": "text", "text": prompt}


def test_request_body_has_image_url_prompt_and_temperature_0(ocr_server):
    import base64

    from PIL import Image

    out = run.PaddleRecognizer(ocr_server.url)(image_of("x"), "text")
    assert out == "warm"
    (body,) = ocr_server.bodies
    assert body["model"] == "paddleocr-vl" and body["temperature"] == 0 and body["max_tokens"] == 2048
    (msg,) = body["messages"]
    image_part, text_part = msg["content"]
    assert msg["role"] == "user" and text_part == {"type": "text", "text": "OCR:"}
    url = image_part["image_url"]["url"]
    assert image_part["type"] == "image_url" and url.startswith("data:image/png;base64,")
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (200, 60)


def test_table_output_becomes_markdown(ocr_server):
    ocr_server.content = "<fcel>Item<fcel>Q1<fcel>Q2<nl><fcel>Freight<fcel>1,100<ecel><nl>"
    assert run.PaddleRecognizer(ocr_server.url)(image_of("x"), "table") == (
        "| Item | Q1 | Q2 |\n| --- | --- | --- |\n| Freight | 1,100 |  |")


def test_otsl_to_markdown():
    assert run.otsl_to_markdown("<fcel>A<fcel>B<nl><fcel>1<nl>") == "| A | B |\n| --- | --- |\n| 1 |  |"
    assert run.otsl_to_markdown("<fcel>a|b<lcel><nl>") == "| a\\|b |  |\n| --- | --- |"
    assert run.otsl_to_markdown("plain text") == "plain text"


def test_ensure_ocr_server_returns_at_once_when_health_is_200(ocr_server, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must not start a server"))
    seconds = run.ensure_ocr_server(ocr_server.url, timeout_s=5)
    assert seconds >= 0 and len(ocr_server.bodies) == 1  # one warm-up request
    assert ocr_server.bodies[0]["messages"][0]["content"][-1]["text"] == "OCR:"
    assert run.ensure_ocr_server(ocr_server.url) == 0.0 and len(ocr_server.bodies) == 1  # a second call does nothing


def test_ensure_ocr_server_starts_the_server_when_it_is_down(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.chdir(tmp_path)
    url = free_url()
    port = int(url.rsplit(":", 1)[1])
    started = []

    def fake_popen(cmd, **kwargs):  # the "server" comes up on the port in the command
        started.append((cmd, FakeOcrServer(port=port)))
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    try:
        assert run.ensure_ocr_server(url, timeout_s=10) >= 0
    finally:
        for _, server in started:
            server.close()
    (cmd, server), = started
    assert cmd[1:] == ["serve", "-hf", "PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "--temp", "0",
                       "--port", str(port), "-np", "4", "--alias", "paddleocr-vl"]
    assert (tmp_path / "var/log/llama-serve.log").exists() and len(server.bodies) == 1


def test_ensure_ocr_server_raises_after_the_timeout(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: object())
    with pytest.raises(RuntimeError, match="llama-serve.log"):
        run.ensure_ocr_server(free_url(), timeout_s=0)


def test_recognize_runs_the_recognizer_on_a_region(tmp_path, monkeypatch):
    rec = FakeRecognizer()
    r = doc_run(tmp_path, monkeypatch, make_pdf(["Hello"]), recognizer=rec)
    region = run.render(r, run.Render(page=1, dpi=72))
    assert run.recognize(r, run.Recognize(region=region.id, mode="table")) == "table:text"
    assert run.run_tool(r, None, "recognize", {"region": region.id, "mode": "table"}) == [
        {"type": "text", "text": "table:text"}]
    (image, mode), = rec.calls  # the second call came from the memo
    assert mode == "table" and image.size == (612, 792)
    assert "unknown region" in run.recognize(r, run.Recognize(region="nope", mode="text"))


@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="needs RUN_LIVE=1 and the local OCR server")
def test_live_paddle_reads_a_word_from_a_rendered_page(tmp_path, monkeypatch):
    print(f"ocr_warmup_s={run.ensure_ocr_server():.2f}")
    r = doc_run(tmp_path, monkeypatch, make_pdf(["Invoice"]), recognizer=run.PaddleRecognizer())
    region = run.render(r, run.Render(page=1, dpi=150))
    assert "Invoice" in run.recognize(r, run.Recognize(region=region.id, mode="text"))


# --- T4 ---


# --- T5 ---


# --- T6 ---


# --- T7 ---
