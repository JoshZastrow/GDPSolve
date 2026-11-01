"""`gdpsolve` CLI. Command groups follow the system's stages (docs/TRD.md section 11)."""

from pathlib import Path
from typing import Annotated

import typer

from gdpsolve.eval import gdp_pdf

app = typer.Typer(no_args_is_help=True, help="GDPSolve: document extraction system and RL training pipeline.")
eval_app = typer.Typer(no_args_is_help=True, help="GDP.pdf benchmark: download and scoring.")
app.add_typer(eval_app, name="eval")


@eval_app.command("download")
def eval_download(
    repo: Annotated[str, typer.Option(help="Hugging Face dataset repo.")] = gdp_pdf.DEFAULT_REPO,
    revision: Annotated[str, typer.Option(help="Pinned dataset revision.")] = gdp_pdf.DEFAULT_REVISION,
    out: Annotated[Path, typer.Option(help="Output directory.")] = gdp_pdf.DEFAULT_DIR,
) -> None:
    """Download the GDP.pdf benchmark (evaluation only; never train on it)."""
    path = gdp_pdf.download(repo, revision, out)
    pdfs = list(path.rglob("*.pdf"))
    typer.echo(f"downloaded {repo}@{revision[:8]} -> {path} ({len(pdfs)} PDFs)")
