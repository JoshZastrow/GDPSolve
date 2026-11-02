from datetime import UTC, datetime

from gdpsolve import ExtractionRequest, ExtractionResult
from gdpsolve.schemas import Document, DocumentMetadata, Job, ObjectRef, Report


def _doc() -> Document:
    meta = DocumentMetadata(filename="a.pdf", sha256="0" * 64, page_count=1, encrypted=False, has_text_layer=True)
    return Document(metadata=meta, pdf=ObjectRef(sha256="0" * 64, media_type="application/pdf", size=4))


def test_request_hides_passwords(tmp_path):
    req = ExtractionRequest(path=tmp_path / "a.pdf", instructions="extract totals", passwords=["pw"])
    assert req.passwords[0].get_secret_value() == "pw"
    assert "pw" not in req.model_dump_json()


def test_contract_roundtrip():
    doc = _doc()
    job = Job(request_id="r1", document=doc, instructions="q", deadline_at=datetime.now(UTC))
    report = Report(job_id=job.id, state="ready", summary="done", answer="42")
    res = ExtractionResult(request_id="r1", document=doc.metadata, status="ok", answer=report.answer)
    for obj in (job, report, res):
        assert type(obj).model_validate_json(obj.model_dump_json()) == obj


def test_map_reduce_jobs_link_by_request():
    doc = _doc()
    deadline = datetime.now(UTC)
    shards = [
        Job(request_id="r1", document=doc, instructions="q", pages=list(range(i, i + 25)), deadline_at=deadline)
        for i in range(1, 201, 25)
    ]
    merge = Job(request_id="r1", kind="merge", document=doc, instructions="q", after=[j.id for j in shards], deadline_at=deadline)
    assert len(shards) == 8 and set(merge.after) == {j.id for j in shards}
    assert sum(len(j.pages) for j in shards) == 200


def test_toolbox_specs_are_valid_json_schemas():
    from gdpsolve.schemas import TOOLBOX

    assert len(TOOLBOX) == 12 and {"split", "post", "report"} <= set(TOOLBOX)
    for name, args in TOOLBOX.items():
        schema = args.model_json_schema()
        assert schema["type"] == "object", name


def test_thread_message_roundtrip():
    from gdpsolve.schemas import Message

    msg = Message(thread="r1", job_id="j1", kind="follow", pages=[38], note="table continues from p. 37")
    assert Message.model_validate_json(msg.model_dump_json()) == msg


def test_verify_result_follows_official_aggregation():
    from gdpsolve.schemas import CriterionScore, VerifyResult

    scored = VerifyResult(
        scores=[CriterionScore(criterion_id="c1", p_pass=0.9), CriterionScore(criterion_id="c2", p_pass=0.5)],
        checkpoint="tinker://run/sampler_weights/000016",
    )
    assert scored.mean_criteria == 0.7 and abs(scored.all_pass - 0.45) < 1e-9
    partial = VerifyResult(
        scores=[CriterionScore(criterion_id="c1", p_pass=0.9), CriterionScore(criterion_id="c2")],
        checkpoint="tinker://run/sampler_weights/000016",
    )
    assert partial.mean_criteria == 0.9 and partial.all_pass is None  # unscored: skipped, not failed
    assert VerifyResult.model_validate_json(partial.model_dump_json()) == partial
    assert scored.model_dump()["mean_criteria"] == 0.7  # aggregates appear in JSON output


def test_checker_configs():
    from gdpsolve.schemas import CHECKERS, CheckerConfig

    assert set(CHECKERS) == {"rubric", "support"}
    assert CHECKERS["rubric"].model_ref == "Qwen/Qwen3.8-27B"  # zero-shot: the base model
    trained = CHECKERS["rubric"].model_copy(update={"checkpoint": "tinker://run/sampler_weights/000128"})
    assert trained.model_ref == "tinker://run/sampler_weights/000128"
    assert CheckerConfig.model_validate_json(trained.model_dump_json()) == trained


def test_verifier_contracts_roundtrip():
    from gdpsolve.schemas import (
        Claim,
        Criterion,
        ExtractedField,
        FieldCheck,
        Region,
        SourceSpan,
        SupportRequest,
        SupportResult,
        VerifyRequest,
    )

    crit = Criterion(id="c1", text="The response should state YTD revenue: $7.4M.")
    months = [SourceSpan(id=f"s{i}", page=3, text=f"{v:,}", method="text_layer") for i, v in enumerate(
        [1200, 1250, 1180, 1240, 1260, 1270], start=1)]
    ytd = ExtractedField(name="ytd_revenue", value=7400, confidence=0.9, sources=months, kind="computed",
                         formula="s1 + s2 + s3 + s4 + s5 + s6")
    chart = Region(id="r1", doc_sha256="0" * 64, page=5, dpi=150,
                   image=ObjectRef(sha256="1" * 64, media_type="image/png", size=10))
    trend = SourceSpan(id="s7", page=5, text="freight cost bar rises Q1->Q2", method="vlm", region="r1")
    claim = Claim(id="k1", text="Freight costs rose in Q2.", source_ids=["s7"])
    summary = ExtractedField(name="margin_driver", value="Q2 margin fell because freight costs rose.",
                             confidence=0.7, sources=[trend], kind="summary", claims=[claim])
    objs = [
        VerifyRequest(response="YTD revenue was $7.4M.", criteria=[crit]),
        ytd,
        summary,
        SupportRequest(claims=[claim], sources=[trend], regions=[chart]),
        SupportResult(scores={"k1": 0.97}, checkpoint="tinker://run/sampler_weights/000032"),
        FieldCheck(field="ytd_revenue", ok=True),
    ]
    for obj in objs:
        assert type(obj).model_validate_json(obj.model_dump_json()) == obj
