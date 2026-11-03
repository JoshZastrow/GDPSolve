import asyncio
import json
import math
import re

import pypdfium2 as pdfium
import pytest

from gdpsolve.eval import gdp_pdf, pdf_input, report


def _pdf(path, text: str) -> None:
    """One page with `text` in its text layer, then one blank page (no text: rendered as an image)."""
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
             b"/Resources << /Font << /F1 5 0 R >> >> >>"),
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    raw = b"%PDF-1.4\n" + b"".join(b"%d 0 obj\n" % i + o + b"\nendobj\n" for i, o in enumerate(objs, 1))
    pdf = pdfium.PdfDocument(raw + b"trailer << /Root 1 0 R >>\n%%EOF")
    pdf.new_page(612, 792)
    pdf.save(path)


def _row() -> dict:
    row = {"task_id": "t1", "prompt": "q", "pdf_path": "pdfs/a.pdf"}
    for n in range(1, 31):
        row |= {f"rubric - {n}. criterion": None, f"rubric - {n}. criterion_type": None,
                f"rubric - {n}. criterion_severity": None}
    row |= {"rubric - 3. criterion": "States B. ", "rubric - 3. criterion_type": "Dodged Bullet",
            "rubric - 1. criterion": "States A.", "rubric - 1. criterion_type": "Primary Intent",
            "rubric - 1. criterion_severity": "Certain dealbreaker", "rubric - 2. criterion": ""}
    return row


def test_pdf_content_text_and_scanned_pages(tmp_path):
    path = tmp_path / "a.pdf"
    _pdf(path, "Total 42")
    parts, truncated_at = pdf_input.content(path, count_tokens=lambda parts: 10)
    assert parts[0] == {"type": "text", "text": "--- page 1 ---\nTotal 42"}
    assert parts[1] == {"type": "text", "text": "--- page 2 ---"}
    assert parts[2]["type"] == "image" and parts[2]["image"].size == (850, 1100)  # 100 DPI letter
    assert truncated_at is None
    parts, truncated_at = pdf_input.content(path, count_tokens=lambda parts: 10, budget=15)
    assert len(parts) == 1 and truncated_at == 2

    meta = pdf_input.metadata(path)
    assert meta.page_count == 2 and meta.has_text_layer and len(meta.sha256) == 64


def test_criteria_from_parquet_row():
    found = gdp_pdf.criteria(_row())
    assert [(c["n"], c["text"]) for c in found] == [(1, "States A."), (3, "States B. ")]  # text unchanged
    assert found[0]["type"] == "Primary Intent" and found[0]["severity"] == "Certain dealbreaker"
    assert found[1]["severity"] is None


def test_official_scorer_adapter_with_mock_judge(monkeypatch):
    scorer = pytest.importorskip("gdp_pdf.scorer")
    from inspect_ai.model import ModelOutput, get_model

    from gdpsolve.eval import official

    verdicts = ['{"score": "1", "rationale": "states A"}', '{"score": "0", "rationale": "misses B"}']
    judge = get_model("mockllm/model", memoize=False, custom_outputs=[
        ModelOutput.from_content(model="mockllm/model", content=v) for v in verdicts])
    monkeypatch.setattr(scorer, "get_model", lambda *a, **k: judge)

    assert official.rubrics(_row()) == [{"id": "1", "criterion": "States A."}, {"id": "3", "criterion": "States B. "}]
    values, rows, _ = asyncio.run(official.grade_all("mockllm/model", [_row()], {"t1": "A."}, concurrency=1))
    assert values == {"t1": {"all_pass": 0.0, "mean_criteria": 0.5}}
    assert sorted(r["result"] for r in rows) == ["fail", "pass"]
    assert {r["criterion_id"] for r in rows} == {"t1:1", "t1:3"}
    assert set(rows[0]) == {"task_id", "criterion_id", "criterion", "result", "rationale"}

    nan = float("nan")
    agg = official.aggregate([{"all_pass": 1.0, "mean_criteria": 1.0}, {"all_pass": nan, "mean_criteria": 0.5},
                              {"all_pass": 0.0, "mean_criteria": nan}])
    assert agg["all_pass"] == 0.5 and agg["mean_criteria"] == 0.75 and agg["n_all_pass"] == 2


def test_report_data_shape(tmp_path):
    grades = [{"task_id": "t1", "criterion_id": "t1:1", "criterion": "States A.", "result": "pass", "rationale": ""},
              {"task_id": "t1", "criterion_id": "t1:3", "criterion": "States B. ", "result": "fail", "rationale": ""},
              {"task_id": "t1", "criterion_id": "t1:2", "criterion": "x", "result": "unscored", "rationale": None}]
    results = {"t1": {"request_id": "t1", "answer": "A. </script>", "prompt_sha256": "0" * 64}}
    decisions = report.decisions([_row()], grades, results)
    assert [d["label"] for d in decisions] == [1, 0]  # unscored verdicts are excluded
    ref = "Qwen/Qwen3.8-27B"
    decisions[0]["p"][ref], decisions[1]["p"][ref] = 0.9, None
    assert report.brier(decisions, ref) == (pytest.approx(0.99), 1)

    meta = {"checker": "rubric", "data": "gdp-pdf", "labels": "l", "dataset": "d", "generated_at": "g",
            "sample": False, "baseline": {"model": "m", "judge": "j", "all_pass": 0.1, "mean_criteria": 0.5,
                                          "n_tasks": 1}}
    data = {"meta": meta, "models": [{"ref": ref, "label": "Untrained verifier", "role": "verifier"}],
            "decisions": decisions}
    with pytest.raises(ValueError, match="provenance"):
        report.render(data)  # real data needs the request ids of the answers used
    html = report.render(data, request_ids={"t1"})
    block = re.search(r'id="report-data">\n(.*?)\n</script>', html, re.DOTALL).group(1)
    assert "</script>" not in block
    parsed = json.loads(block)
    assert parsed == data
    for d in parsed["decisions"]:
        assert set(d) == {"task_id", "criterion_id", "criterion", "type", "severity", "response", "label", "p",
                          "request_id", "prompt_sha256"}
        assert d["label"] in (0, 1) and d["type"] in ("Primary Intent", "Dodged Bullet")
        assert all(p is None or 0 <= p <= 1 for p in d["p"].values()) and not math.isnan(d["label"])


def test_answer_text_strips_thinking_from_cut_off_completions():
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from gdpsolve.eval import baseline

    tokenizer = get_tokenizer("thinkingmachines/Inkling-Small")
    renderer = renderers.get_renderer("tml_v0", tokenizer)
    assert tokenizer.decode(baseline._CLOSE) == "<|end_message|><|content_model_end_sampling|>"
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "let me think"}, {"type": "text", "text": "The answer is 42."}]}]
    model_input, weights = renderer.build_supervised_example(messages)
    tokens = [t for c in model_input.chunks for t in c.tokens]
    response = [t for t, w in zip(tokens, weights.tolist(), strict=True) if w > 0]
    assert baseline.answer_text(renderer, response) == ("The answer is 42.", True)
    assert baseline.answer_text(renderer, response[:6]) == ("", False)  # cut off while thinking
    text, clean = baseline.answer_text(renderer, response[:-3])  # cut off mid-answer: partial text kept
    assert not clean and text.startswith("The answer") and "<|" not in text


def test_pending_rerun_truncated(tmp_path):
    from gdpsolve.eval import baseline

    results = tmp_path / "results.jsonl"
    rows = [{"request_id": "a", "status": "ok", "finish": "stop"}, {"request_id": "b", "status": "ok", "finish": "length"},
            {"request_id": "c", "status": "failed"}]
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tasks = [{"task_id": t} for t in "abcd"]
    assert [t["task_id"] for t in baseline.pending(tasks, results)] == ["c", "d"]
    assert [t["task_id"] for t in baseline.pending(tasks, results, rerun_truncated=True)] == ["b", "c", "d"]
