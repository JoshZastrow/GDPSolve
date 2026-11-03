"""GDP.pdf benchmark (100 tasks: data.parquet + PDFs).

The official `surgeai/GDP.pdf` HF dataset currently returns 404, so we default to a
community mirror whose card and file layout match the official harness
(https://github.com/surge-ai/gdp-pdf). Pin the revision so results are reproducible.

EVALUATION ONLY: the dataset card forbids training on this data or its derivatives.
"""

from pathlib import Path

DEFAULT_REPO = "Oitaaa/GDP.pdf"
DEFAULT_REVISION = "7f7657830967e418091b010bc4ba0550a87d5e09"
DEFAULT_DIR = Path("data/gdp_pdf")


def download(repo: str = DEFAULT_REPO, revision: str = DEFAULT_REVISION, out: Path = DEFAULT_DIR) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=out))


def load_tasks(data: Path = DEFAULT_DIR) -> list[dict]:
    """Rows of data.parquet as dicts; absent rubric columns are None."""
    import pyarrow.parquet as pq

    return pq.read_table(data / "data.parquet").to_pylist()


def criteria(row: dict) -> list[dict]:
    """Non-empty `rubric - N. criterion` columns ordered by N, as {n, text, type, severity}.

    Same selection as the official loader (gdp_pdf.dataset._rubrics_from_row); text is unchanged.
    """
    found = []
    for n in range(1, 100):
        text = row.get(f"rubric - {n}. criterion")
        if text not in (None, ""):
            found.append({"n": n, "text": str(text), "type": row.get(f"rubric - {n}. criterion_type"),
                          "severity": row.get(f"rubric - {n}. criterion_severity")})
    return found
