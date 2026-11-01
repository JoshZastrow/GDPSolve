"""Download the GDP.pdf benchmark (100 tasks: data.parquet + PDFs) into data/gdp_pdf/.

The official `surgeai/GDP.pdf` HF dataset currently returns 404, so we default to a
community mirror whose card and file layout match the official harness
(https://github.com/surge-ai/gdp-pdf). Pin the revision so results are reproducible.

EVALUATION ONLY: the dataset card forbids training on this data or its derivatives.
"""

from pathlib import Path

import typer
from huggingface_hub import snapshot_download

DEFAULT_REPO = "Oitaaa/GDP.pdf"
DEFAULT_REVISION = "7f7657830967e418091b010bc4ba0550a87d5e09"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gdp_pdf"


def main(repo: str = DEFAULT_REPO, revision: str = DEFAULT_REVISION, out: Path = DATA_DIR) -> None:
    path = snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=out)
    pdfs = list(Path(path).rglob("*.pdf"))
    print(f"downloaded {repo}@{revision[:8]} -> {path} ({len(pdfs)} PDFs)")


if __name__ == "__main__":
    typer.run(main)
