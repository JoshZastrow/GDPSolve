"""Official GDP.pdf grades (docs/m1-baseline.md, step 2).

Calls the upstream scorer unchanged (gdp_pdf.scorer.rubric_scorer, surge-ai/gdp-pdf@7a72a51) with a
minimal state: it reads only state.metadata["rubrics"] and state.output.completion.
"""

import asyncio
import math
from types import SimpleNamespace

from gdpsolve.eval.gdp_pdf import criteria

JUDGE = "google/gemini-3.5-flash"
# Assumed judge $ per 1M tokens (input, output); Gemini 3.5 Flash pricing is not published here.
JUDGE_PRICE = (0.50, 3.00)


def rubrics(task: dict) -> list[dict]:
    return [{"id": str(c["n"]), "criterion": c["text"]} for c in criteria(task)]


async def grade(score, task: dict, answer: str) -> tuple[dict, list[dict]]:
    """Score one answer. Returns (value {all_pass, mean_criteria}, grades.jsonl rows)."""
    state = SimpleNamespace(metadata={"rubrics": rubrics(task)}, output=SimpleNamespace(completion=answer))
    result = await score(state, None)
    texts = {r["id"]: r["criterion"] for r in state.metadata["rubrics"]}
    rows = [{"task_id": task["task_id"], "criterion_id": f"{task['task_id']}:{g['rubric_id']}",
             "criterion": texts[g["rubric_id"]], "result": g["result"], "rationale": g["rationale"]}
            for g in result.metadata["rubric_grades"]]
    return result.value, rows


async def grade_all(judge: str, tasks: list[dict], answers: dict[str, str], concurrency: int):
    """Grade every task that has an answer. Returns ({task_id: value}, rows, judge usage)."""
    from gdp_pdf.scorer import rubric_scorer
    from inspect_ai.model._model import init_model_usage, model_usage

    init_model_usage()
    score = rubric_scorer(judge_model=judge)
    gate = asyncio.Semaphore(concurrency)

    async def one(task):
        async with gate:
            return task["task_id"], await grade(score, task, answers[task["task_id"]])

    graded = await asyncio.gather(*(one(t) for t in tasks if t["task_id"] in answers))
    values = {tid: value for tid, (value, _) in graded}
    rows = [row for _, (_, task_rows) in graded for row in task_rows]
    return values, rows, model_usage()


def aggregate(values: list[dict]) -> dict:
    """Official aggregation: mean over tasks; NaN (unscored) values are excluded."""
    out = {}
    for key in ("all_pass", "mean_criteria"):
        xs = [v[key] for v in values if not math.isnan(v[key])]
        out[key] = sum(xs) / len(xs) if xs else None
        out[f"n_{key}"] = len(xs)
    return out


def estimate(tasks: list[dict], answers: dict[str, str], out_tokens: int = 500) -> tuple[int, int, float]:
    """Rough judge (input tokens, output tokens, $): about 4 characters per token."""
    from gdp_pdf.scorer import _judge_prompt

    calls = [_judge_prompt(answers[t["task_id"]], r["criterion"]) for t in tasks if t["task_id"] in answers
             for r in rubrics(t)]
    tokens_in = sum(len(p) for p in calls) // 4
    tokens_out = out_tokens * len(calls)
    return tokens_in, tokens_out, (tokens_in * JUDGE_PRICE[0] + tokens_out * JUDGE_PRICE[1]) / 1e6
