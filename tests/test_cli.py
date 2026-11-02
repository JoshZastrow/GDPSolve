from typer.testing import CliRunner

from gdpsolve import cli

runner = CliRunner()


def test_download_gdp_pdf(monkeypatch, tmp_path):
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "a.pdf").write_bytes(b"%PDF")
    calls = {}

    def fake_download(repo, revision, out):
        calls.update(repo=repo, revision=revision, out=out)
        return tmp_path

    monkeypatch.setattr(cli.gdp_pdf, "download", fake_download)
    result = runner.invoke(cli.app, ["download", "gdp-pdf", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls["revision"] == cli.gdp_pdf.DEFAULT_REVISION
    assert "(1 PDFs)" in result.output

