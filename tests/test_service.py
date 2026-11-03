"""Tests for the document service and its SDK (docs/service-spec.md). Offline: FakeSampler, FakeRecognizer."""

import json
import threading
import time

import pytest
from test_run import FakeRecognizer, make_pdf, make_scan
from typer.testing import CliRunner

from gdpsolve import client, service
from gdpsolve.verifier import store


class FakeSampler:
    """Every part costs 10 tokens. Gives the scripted answers in order (the last one repeats) after `delay` s.

    Records the content of each call.
    """

    def __init__(self, answers=("the answer",), delay: float = 0.0):
        self.answers = list(answers)
        self.delay = delay
        self.seen: list[list] = []
        self._lock = threading.Lock()

    def count_tokens(self, parts: list[dict]) -> int:
        return 10 * len(parts)

    def sample(self, content: list[dict]) -> dict:
        time.sleep(self.delay)
        with self._lock:
            self.seen.append(content)
            text = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return {"text": text, "finish": "stop", "prompt_tokens": 100, "completion_tokens": 5, "prompt_sha256": "0" * 64}


def config(**kw) -> service.Config:
    return service.Config(**{"sampler": FakeSampler(), "model": "thinkingmachines/Inkling-Small"} | kw)


@pytest.fixture
def var(tmp_path, monkeypatch):
    """Run with the object store and job results in tmp_path/var."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "var"


def upload(data: bytes) -> str:
    return service.handle(config(), "upload", data)["file_id"]


# --- upload ---


def test_upload_gives_content_addressed_id(var):
    a = service.handle(config(), "upload", b"%PDF-1")
    assert a["file_id"].startswith("gdpsolve://") and len(a["file_id"]) == len("gdpsolve://") + 64
    assert a["presigned_url"] is None
    assert service.handle(config(), "upload", b"%PDF-1") == a
    assert service.handle(config(), "upload", b"%PDF-2") != a


# --- parse ---


def test_parse_native_text_and_figure_pages(var):
    file_id = upload(make_pdf(["Total 42", "Second"]))
    out = service.handle(config(), "parse", {"input": file_id})
    assert out["response_type"] == "parse" and out["usage"] == {"num_pages": 2}
    assert len(out["job_id"]) == 32 and out["duration"] >= 0
    chunks = out["result"]["chunks"]
    assert out["result"]["type"] == "full" and [c["content"] for c in chunks] == ["Total 42", "Second"]
    block = chunks[0]["blocks"][0]
    assert block["type"] == "Text" and block["content"] == "Total 42" and block["image_url"] is None
    assert block["bbox"] == {"left": 0, "top": 0, "width": 1, "height": 1, "page": 1}

    scan = service.handle(config(), "parse", {"input": upload(make_scan("Hello"))})
    block = scan["result"]["chunks"][0]["blocks"][0]
    assert block["type"] == "Figure" and block["content"] == ""
    sha = block["image_url"].removeprefix("gdpsolve://")
    assert (var / "objects" / sha).read_bytes()[:4] == b"\x89PNG"


def test_parse_page_range(var):
    file_id = upload(make_pdf(["one", "two", "three", "four"]))
    pages = lambda out: [c["blocks"][0]["bbox"]["page"] for c in out["result"]["chunks"]]  # noqa: E731
    assert pages(service.handle(config(), "parse", {"input": file_id, "settings": {"page_range": {"start": 2, "end": 3}}})) == [2, 3]
    assert pages(service.handle(config(), "parse", {"input": file_id, "settings": {"page_range": [4, 1]}})) == [1, 4]
    assert pages(service.handle(config(), "parse", {"input": file_id, "settings": {"page_range": None}})) == [1, 2, 3, 4]


def test_parse_paddle_ocrs_pages_without_text(var):
    rec = FakeRecognizer()
    out = service.handle(config(parser="paddle", recognizer=rec), "parse",
                         {"input": upload(make_pdf(["native"]))})
    assert out["result"]["chunks"][0]["blocks"][0]["content"] == "native" and rec.calls == []
    out = service.handle(config(parser="paddle", recognizer=rec), "parse", {"input": upload(make_scan("Hello"))})
    block = out["result"]["chunks"][0]["blocks"][0]
    assert block["type"] == "Text" and block["content"] == "text:text"
    assert [mode for _, mode in rec.calls] == ["text"] and rec.calls[0][0].size == (850, 1100)


def test_parse_bad_input_is_422(var):
    with pytest.raises(service.BadInput):
        service.handle(config(), "parse", {})
    with pytest.raises(service.BadInput):
        service.handle(config(), "parse", {"input": "gdpsolve://" + "0" * 64})


# --- extract ---


def test_extract_free_form(var):
    sampler = FakeSampler(["42"])
    out = service.handle(config(sampler=sampler), "extract",
                         {"input": upload(make_pdf(["Total 42"])), "instructions": {"schema": {}, "system_prompt": "q"}})
    assert out["response_type"] == "extract" and out["result"] == [{"answer": "42"}]
    assert sampler.seen == [[{"type": "text", "text": "q"}, {"type": "text", "text": "--- page 1 ---\nTotal 42"}]]
    assert out["usage"]["num_pages"] == 1 and out["usage"]["prompt_tokens"] == 100
    assert out["usage"]["completion_tokens"] == 5 and out["usage"]["cost_usd"] == pytest.approx((100 * 0.58 + 5 * 1.44) / 1e6)
    assert out["truncated_at_page"] is None and out["finish"] == "stop" and out["prompt_sha256"] == "0" * 64


def test_extract_images_and_token_budget(var):
    sampler = FakeSampler()
    pdf = make_scan("Hello")
    out = service.handle(config(sampler=sampler, token_budget=20), "extract",
                         {"input": upload(pdf), "instructions": {"schema": {}, "system_prompt": "q"}})
    _, header, image = sampler.seen[0]
    assert header == {"type": "text", "text": "--- page 1 ---"} and image["type"] == "image"
    assert image["image"].size == (850, 1100) and out["truncated_at_page"] is None
    out = service.handle(config(sampler=sampler, token_budget=10), "extract",
                         {"input": upload(make_pdf(["a", "b"])), "instructions": {"schema": {}, "system_prompt": "q"}})
    assert out["truncated_at_page"] == 2 and len(sampler.seen[1]) == 2


def test_extract_schema(var):
    sampler = FakeSampler(['{"total": 42}', "not json"])
    payload = {"input": upload(make_pdf(["Total 42"])),
               "instructions": {"schema": {"type": "object"}, "system_prompt": "q"}}
    out = service.handle(config(sampler=sampler), "extract", payload)
    assert out["result"] == [{"total": 42}] and "confidence" not in out
    assert sampler.seen[0][0]["text"] == 'q\nReturn only JSON matching this schema: {"type": "object"}'
    out = service.handle(config(sampler=sampler), "extract", payload)
    assert out["result"] == [{"answer": "not json"}] and out["confidence"] == "low"


def test_extract_reuses_a_parse_job(var):
    parsed = service.handle(config(), "parse", {"input": upload(make_pdf(["one", "two"])),
                                                "settings": {"page_range": [2]}})
    assert json.loads((var / "jobs" / f"{parsed['job_id']}.json").read_text()) == parsed
    sampler = FakeSampler()
    service.handle(config(sampler=sampler), "extract",
                   {"input": f"jobid://{parsed['job_id']}", "instructions": {"schema": {}, "system_prompt": "q"}})
    assert sampler.seen[0][1:] == [{"type": "text", "text": "--- page 2 ---\ntwo"}]


def test_extract_prompt_matches_m1_baseline(var, tmp_path):
    """PARITY: the service's prompt for a document is token for token the M1 baseline prompt."""
    pytest.importorskip("tinker_cookbook")
    from gdpsolve.eval import baseline

    pdf = make_pdf(["Total 42", "Second page"])
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "a.pdf").write_bytes(pdf)
    scan = tmp_path / "pdfs" / "b.pdf"
    scan.write_bytes(make_scan("Hello"))
    tasks = [{"task_id": "a", "prompt": "What is the total?", "pdf_path": "pdfs/a.pdf"},
             {"task_id": "b", "prompt": "Read it.", "pdf_path": "pdfs/b.pdf"}]
    model = "thinkingmachines/Inkling-Small"
    try:
        _, prepared = baseline.prepare(model, tasks, tmp_path, 120_000, effort=0.3)
    except OSError as e:  # tokenizer not in the local cache
        pytest.skip(str(e))
    tinker = service.Tinker(model, None, effort=0.3, max_tokens=8192)
    cfg = config(sampler=tinker)
    for task, expected, truncated_at, _ in prepared:
        parsed = service.handle(cfg, "parse", {"input": upload((tmp_path / task["pdf_path"]).read_bytes())})
        content, truncated = service.content(parsed["result"]["chunks"], task["prompt"], tinker.count_tokens, 120_000)
        prompt = tinker.prompt(content)
        assert truncated == truncated_at
        assert prompt.length == expected.length and prompt.chunks == expected.chunks


# --- split, classify ---


def test_split(var):
    sampler = FakeSampler(['["Summary"]', '["Summary", "Tables"]', "no"])
    sections = [{"name": "Summary", "description": "overview"}, {"name": "Tables", "description": "numbers"}]
    out = service.handle(config(sampler=sampler), "split",
                         {"input": upload(make_pdf(["a", "b", "c"])), "split_description": sections})
    assert out["response_type"] == "split"
    assert out["result"] == {"splits": [{"name": "Summary", "pages": [1, 2], "conf": "high"},
                                        {"name": "Tables", "pages": [2], "conf": "high"}]}
    assert len(sampler.seen) == 3 and "Summary: overview" in sampler.seen[0][0]["text"]


def test_classify(var):
    sampler = FakeSampler([" invoice\n"])
    schema = [{"category": "invoice", "criteria": ["has a total"]}, {"category": "letter", "criteria": ["greeting"]}]
    out = service.handle(config(sampler=sampler), "classify",
                         {"input": upload(make_pdf([str(n) for n in range(1, 8)])), "classification_schema": schema})
    assert out["response_type"] == "classify" and out["result"] == {"category": "invoice"}
    pages = [p["text"] for p in sampler.seen[0][1:]]
    assert pages[0] == "--- page 1 ---\n1" and pages[-1] == "--- page 5 ---\n5"  # first 5 pages by default
    assert "invoice: has a total" in sampler.seen[0][0]["text"]


# --- server and SDK ---


@pytest.fixture
def server(var):
    """A server on a free port in a background thread; yields (url, server)."""
    holder = {}

    def start(cfg: service.Config, workers: int = 8):
        srv = service.server(cfg, "127.0.0.1", 0, workers)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        holder["srv"] = srv
        return f"http://127.0.0.1:{srv.server_port}", srv

    yield start
    if "srv" in holder:
        holder["srv"].shutdown()
        holder["srv"].stop()


def test_sdk_round_trips_every_endpoint(server):
    url, _ = server(config(sampler=FakeSampler(['["S"]'])))
    sdk = client.GDPSolve(url)
    assert sdk.health()["ok"] is True and sdk.health()["workers"] == 8
    assert sdk.health()["config"]["model"] == "thinkingmachines/Inkling-Small"
    file_id = sdk.upload(make_pdf(["Total 42"]))["file_id"]
    parsed = sdk.parse.run(input=file_id)
    assert parsed["result"]["chunks"][0]["content"] == "Total 42"
    assert sdk.extract.run(input=f"jobid://{parsed['job_id']}",
                           instructions={"schema": {}, "system_prompt": "q"})["result"] == [{"answer": '["S"]'}]
    assert sdk.split.run(input=file_id, split_description=[{"name": "S", "description": "d"}])["result"] == {
        "splits": [{"name": "S", "pages": [1], "conf": "high"}]}
    assert sdk.classify.run(input=file_id, classification_schema=[{"category": "S", "criteria": []}])[
        "result"] == {"category": '["S"]'}
    with pytest.raises(client.GDPSolveError) as e:
        sdk.parse.run(input="gdpsolve://missing")
    assert e.value.status == 422 and "missing" in e.value.detail


def test_sdk_upload_path_and_job_failure(server, tmp_path):
    class Broken(FakeSampler):
        def sample(self, content):
            raise RuntimeError("tinker down")

    url, _ = server(config(sampler=Broken()))
    sdk = client.GDPSolve(url)
    (tmp_path / "a.pdf").write_bytes(make_pdf(["x"]))
    file_id = sdk.upload(tmp_path / "a.pdf")["file_id"]
    with pytest.raises(client.GDPSolveError) as e:
        sdk.extract.run(input=file_id, instructions={"schema": {}, "system_prompt": "q"})
    assert e.value.status == 500 and "tinker down" in e.value.detail


def test_workers_run_in_parallel(server):
    url, _ = server(config(sampler=FakeSampler(delay=0.5)), workers=8)
    sdk = client.GDPSolve(url)
    file_id = sdk.upload(make_pdf(["x"]))["file_id"]
    start = time.monotonic()
    threads = [threading.Thread(target=sdk.extract.run, args=(file_id,),
                                kwargs={"instructions": {"schema": {}, "system_prompt": "q"}}) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.monotonic() - start < 8 * 0.5 / 2  # serial would take 4 s


# --- eval hook ---


def test_eval_gdp_pdf_service_writes_baseline_rows(server, tmp_path, monkeypatch):
    from gdpsolve.cli import app
    from gdpsolve.eval import baseline, gdp_pdf
    from gdpsolve.schemas import ExtractionResult

    (tmp_path / "pdfs").mkdir()
    for name in "ab":
        (tmp_path / "pdfs" / f"{name}.pdf").write_bytes(make_pdf([f"doc {name}"]))
    tasks = [{"task_id": name, "prompt": "q", "pdf_path": f"pdfs/{name}.pdf"} for name in "ab"]
    monkeypatch.setattr(gdp_pdf, "load_tasks", lambda data: tasks)
    url, _ = server(config(sampler=FakeSampler(["42"])))
    out = tmp_path / "runs" / "service"
    args = ["eval", "gdp-pdf", "--system", "service", "--url", url, "--out", str(out), "--data", str(tmp_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output

    rows = baseline.done(out / "results.jsonl")
    assert set(rows) == {"a", "b"}
    for row in rows.values():
        ExtractionResult.model_validate({k: row[k] for k in ExtractionResult.model_fields if k in row})
        assert row["status"] == "ok" and row["answer"] == "42" and row["truncated_at_page"] is None
        assert row["model"] == "thinkingmachines/Inkling-Small" and row["effort"] == 0.3
        assert row["prompt_tokens"] == 100 and row["completion_tokens"] == 5 and row["finish"] == "stop"
        assert row["prompt_sha256"] == "0" * 64 and row["latency_s"] >= 0
        assert row["document"]["filename"].endswith(".pdf")
    stats = json.loads((out / "run.json").read_text())
    assert stats["system"] == "service" and stats["n"] == 2 and stats["wall_s"] > 0
    assert stats["throughput_docs_per_min"] > 0 and stats["latency_p50"] is not None
    assert stats["latency_p95"] is not None and stats["cost_usd"] == pytest.approx(2 * (100 * 0.58 + 5 * 1.44) / 1e6)
    assert stats["config"]["parser"] == "native"

    assert CliRunner().invoke(app, args).exit_code == 0  # resume: nothing left to do
    assert len((out / "results.jsonl").read_text().splitlines()) == 2
