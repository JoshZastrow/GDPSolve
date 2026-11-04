"""ParseBench page verifier: the leaderboard's own per-page score, usable as a reward.

One page (a ParseBench test case) plus one parse output -> the headline metric of the page's dimension
(gdpsolve.eval.parsebench.DIMENSIONS), in [0, 1]. Routing and evaluators are ParseBench's
`EvaluationRunner` (pinned `parse-bench`), so averaging `score()` over a group reproduces the
group's `avg_<metric>` from `gdpsolve eval parsebench` (tests/test_parsebench.py checks this).

Deterministic and rule-based: no model calls. Needs the `parsebench` extra.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cache
from typing import Any

from gdpsolve.eval.parsebench import DIMENSIONS

METRIC = dict(DIMENSIONS.values())  # evaluation group -> headline metric


@cache
def _runner():
    from parse_bench.evaluation.runner import EvaluationRunner

    return EvaluationRunner(output_dir=".")  # the defaults `parse-bench evaluation run` uses


def evaluate(result: Any, case: Any) -> Any | None:
    """ParseBench EvaluationResult for one page, routed as EvaluationRunner.run_evaluation does.

    `result` is an InferenceResult (or its dict); `case` a ParseBench TestCase.
    None when the runner would skip the page (no evaluator applies).
    """
    from parse_bench.schemas.pipeline_io import InferenceResult
    from parse_bench.schemas.product import ProductType
    from parse_bench.test_cases.schema import LayoutDetectionTestCase, ParseTestCase

    runner = _runner()
    result = result if isinstance(result, InferenceResult) else InferenceResult.model_validate(result)
    if isinstance(case, ParseTestCase) and (case.qa_configs or case.qa_config is not None):
        raise ValueError(f"{case.test_id}: QA test cases are not part of the five leaderboard dimensions")
    if isinstance(case, LayoutDetectionTestCase) and result.product_type == ProductType.PARSE:
        return runner._evaluate_single(result, case, runner._evaluators["layout_detection"], True)
    parse_capable = result.product_type == ProductType.PARSE or (
        result.product_type == ProductType.LAYOUT_DETECTION and runner._is_llamaparse_output(result))
    if parse_capable and runner._has_mixed_rules(case):
        return runner._evaluate_single(result, case, None, "multi_task")
    evaluator = runner._evaluators.get(result.product_type.value)
    if evaluator is None or not evaluator.can_evaluate(result, case):
        return None
    return runner._evaluate_single(result, case, evaluator, False)


def headline(evaluation: Any | None, group: str) -> float | None:
    """The page's leaderboard score: the group's headline metric.

    Failures the leaderboard counts against the parser score 0. Harness errors, skipped pages, and pages
    without the metric return None: the leaderboard leaves those out of the average.
    """
    from parse_bench.evaluation.runner import _is_infra_failure, _is_skipped_result

    if evaluation is None:
        return None
    if not evaluation.success:
        return None if _is_skipped_result(evaluation) or _is_infra_failure(evaluation) else 0.0
    return next((m.value for m in evaluation.metrics if m.metric_name == METRIC[group]), None)


def score(result: Any, case: Any) -> float | None:
    """Leaderboard score of one page in [0, 1], or None when unscored."""
    return headline(evaluate(result, case), case.group)


def parse_result(case: Any, markdown: str, layout_pages: list[dict] | None = None,
                 pipeline_name: str = "gdpsolve") -> Any:
    """Wrap a parser's page output as the InferenceResult ParseBench scores.

    `markdown` is the page's Markdown (tables as HTML, charts as tables); `layout_pages` are
    ParseLayoutPageIR dicts with item bboxes, needed only for visual grounding.
    """
    from parse_bench.schemas.parse_output import ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult

    now = datetime.now(UTC)
    output = ParseOutput(example_id=case.test_id, pipeline_name=pipeline_name, markdown=markdown,
                         pages=[{"page_index": 0, "markdown": markdown}], layout_pages=layout_pages or [])
    request = InferenceRequest(example_id=case.test_id, source_file_path=str(case.file_path), product_type="parse")
    return InferenceResult(request=request, pipeline_name=pipeline_name, product_type="parse", raw_output={},
                           output=output, started_at=now, completed_at=now, latency_in_ms=0)
