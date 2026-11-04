"""ParseBench verifier vs. ParseBench's own harness, on the test subset.

Needs the `parsebench` extra and `gdpsolve download parsebench --test`. Uses published leaderboard outputs from
var/parsebench/output/llamaparse_cost_effective when present (see docs/parsebench.md), else synthetic outputs.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("parse_bench")

from gdpsolve.eval import parsebench
from gdpsolve.verifier import parsebench as verifier

DATA = parsebench.TEST_DIR
PUBLISHED = Path("var/parsebench/output/llamaparse_cost_effective")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run `gdpsolve download parsebench --test`")


def _result_path(results: Path, group: str, test_id: str) -> Path:
    return results / parsebench.RESULTS_DIR[group] / (test_id.split("/", 1)[1] + ".result.json")


def _write_outputs(group: str, cases: dict, results: Path) -> None:
    """One result file per page: published outputs if available, else table ground truth / blank pages."""
    for test_id, case in cases.items():
        path = _result_path(results, group, test_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if PUBLISHED.exists():
            shutil.copy(_result_path(PUBLISHED, group, test_id), path)
        else:
            markdown = (case.expected_markdown or "") if group == "table" else ""
            path.write_text(verifier.parse_result(case, markdown).model_dump_json())


@pytest.mark.parametrize("dimension", list(parsebench.DIMENSIONS))
def test_verifier_matches_harness(dimension, tmp_path):
    group, metric = parsebench.DIMENSIONS[dimension]
    if group == "layout" and PUBLISHED.exists() and importlib.util.find_spec("llama_cloud") is None:
        pytest.skip("the LlamaParse layout adapter needs llama-cloud: uv run --with llama-cloud pytest ...")
    cases = parsebench.load_cases(DATA, group)
    results = tmp_path / "results"
    _write_outputs(group, cases, results)

    report = parsebench.evaluate_group(results, DATA, group, tmp_path / "report")
    harness = {r["test_id"]: next((m["value"] for m in r["metrics"] if m["metric_name"] == metric), None)
               for r in report["per_example_results"]}
    ours = {test_id: verifier.score(json.loads(_result_path(results, group, test_id).read_text()), case)
            for test_id, case in cases.items()}

    for test_id, value in harness.items():
        if value is not None:
            assert ours[test_id] == pytest.approx(value, abs=1e-9), test_id
    avg = report["aggregate_metrics"].get(f"avg_{metric}")
    scored = [v for v in ours.values() if v is not None]
    if avg is not None:
        assert sum(scored) / len(scored) == pytest.approx(avg, abs=1e-9)
    group_score = parsebench.headline_score(report, group)
    assert (group_score is None) == (not scored)
    if scored:
        assert sum(scored) / len(scored) == pytest.approx(group_score, abs=1e-9)


def test_blank_output_scores_zero():
    for group in ("table", "chart", "text_content", "text_formatting"):
        for case in parsebench.load_cases(DATA, group).values():
            assert verifier.score(verifier.parse_result(case, ""), case) == 0.0, case.test_id


def test_ground_truth_table_scores_high():
    for case in parsebench.load_cases(DATA, "table").values():
        assert verifier.score(verifier.parse_result(case, case.expected_markdown), case) > 0.95, case.test_id
