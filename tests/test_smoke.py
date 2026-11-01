from gdpsolve import ExtractionRequest, ExtractionResult
from gdpsolve.schemas import DocumentMetadata


def test_contract_roundtrip(tmp_path):
    req = ExtractionRequest(path=tmp_path, instructions="extract totals", passwords=["pw"])
    assert req.passwords[0].get_secret_value() == "pw"
    assert "pw" not in req.model_dump_json()

    meta = DocumentMetadata(filename="a.pdf", sha256="0" * 64, page_count=1, encrypted=False, has_text_layer=True)
    res = ExtractionResult(job_id="j1", document=meta, status="ok", confidence=0.9)
    assert ExtractionResult.model_validate_json(res.model_dump_json()) == res
