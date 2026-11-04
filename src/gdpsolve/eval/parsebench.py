"""ParseBench (llamaindex/ParseBench): document parsing, about 2,078 pages in five dimensions.

Scoring is ParseBench's own deterministic harness (`parse-bench`, pinned in the `parsebench` extra), invoked
the way its leaderboard re-scores published outputs (ParseBench scripts/leaderboard/rescore.py), so scores
compare directly with parsebench.ai.

Outputs to score are ParseBench `InferenceResult` JSON files, one per page:
`<results>/<chart|layout|table|text>/<test_id>.result.json`, with `product_type: "parse"`.
A page without a result file counts as 0, as on the leaderboard.

EVALUATION ONLY by project policy (docs/TRD.md section 2), although the dataset itself is Apache-2.0.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = "llamaindex/ParseBench"
REVISION = "2805a1d940f95a203e0ae4b88be9934f7765b3fc"  # leaderboard/results.yaml data_revision
TEST_REVISION = "68bbab242f749df2e2ef753daabcbbbe291d943e"  # `test-data` branch: 3 documents per category
DEFAULT_DIR = Path("data/parsebench")
TEST_DIR = Path("data/parsebench-test")

# Leaderboard column -> (evaluation group, headline metric), as in ParseBench scripts/leaderboard/common.py.
# The score is avg_<metric> from the group's _evaluation_report.json; Overall is the unweighted mean.
DIMENSIONS: dict[str, tuple[str, str]] = {
    "tables": ("table", "grits_trm_composite"),
    "charts": ("chart", "rule_pass_rate"),
    "content_faithfulness": ("text_content", "content_faithfulness"),
    "semantic_formatting": ("text_formatting", "semantic_formatting"),
    "visual_grounding": ("layout", "layout_element_rule_pass_rate"),
}
# Evaluation group -> results subdirectory (both text groups score the same text/ outputs).
RESULTS_DIR = {"table": "table", "chart": "chart", "text_content": "text", "text_formatting": "text", "layout": "layout"}


def download(test: bool = False, out: Path | None = None, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    out = out or (TEST_DIR if test else DEFAULT_DIR)
    revision = revision or (TEST_REVISION if test else REVISION)
    return Path(snapshot_download(repo_id=REPO, repo_type="dataset", revision=revision, local_dir=out))


def load_cases(data: Path, group: str) -> dict:
    """ParseBench test cases of one evaluation group, by test_id."""
    from parse_bench.test_cases.loader import load_test_cases

    return {tc.test_id: tc for tc in load_test_cases(root_dir=data, require_test_json=False) if tc.group == group}


def evaluate_group(results: Path, data: Path, group: str, report_dir: Path) -> dict:
    """`parse-bench evaluation run` for one group; returns its _evaluation_report.json."""
    cmd = [sys.executable, "-m", "parse_bench.cli", "evaluation", "run", str(results), "-t", str(data),
           "-g", group, "-r", str(report_dir), "--force", "True", "--export_csv", "False",
           "--export_rule_csv", "False", "--export_markdown", "False", "--export_html", "False"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    report = report_dir / "_evaluation_report.json"
    if proc.returncode != 0 or not report.exists():
        raise RuntimeError(f"parse-bench evaluation of {group!r} failed (exit {proc.returncode}):\n"
                           f"{(proc.stdout + proc.stderr)[-2000:]}")
    return json.loads(report.read_text())


def headline_score(report: dict, group: str) -> float | None:
    """The group's leaderboard score in [0, 1]: avg_<metric> from its report.

    The report has no avg_<metric> when no page produced the metric. If that is because every page failed
    in a way the leaderboard charges to the parser (e.g. no layout boxes at all), the score is 0, not missing;
    the per-page rule is the verifier's (gdpsolve.verifier.parsebench.headline).
    """
    value = (report.get("aggregate_metrics") or {}).get(f"avg_{dict(DIMENSIONS.values())[group]}")
    if value is not None:
        return value
    from parse_bench.schemas.evaluation import EvaluationResult

    from gdpsolve.verifier.parsebench import headline

    pages = [headline(EvaluationResult.model_validate(r), group) for r in report.get("per_example_results", [])]
    pages = [p for p in pages if p is not None]
    return sum(pages) / len(pages) if pages else None


def evaluate(results: Path, data: Path, report_dir: Path, dimensions: list[str] | None = None) -> dict:
    """Headline score per dimension (0-100, like the leaderboard), plus overall when all five ran."""
    scores, counts = {}, {}
    for dim in dimensions or list(DIMENSIONS):
        group, _ = DIMENSIONS[dim]
        report = evaluate_group(results, data, group, report_dir / group)
        value = headline_score(report, group)
        scores[dim] = None if value is None else round(100 * value, 2)
        counts[dim] = {k: report.get(k) for k in ("total_examples", "successful", "failed", "skipped")}
    done = [scores.get(d) for d in DIMENSIONS]
    overall = round(sum(done) / len(done), 2) if all(v is not None for v in done) else None
    return {"overall": overall, "scores": scores, "counts": counts}
