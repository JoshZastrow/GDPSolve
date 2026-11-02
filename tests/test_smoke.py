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
