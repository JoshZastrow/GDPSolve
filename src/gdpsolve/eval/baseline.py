"""M1 baseline answers on GDP.pdf (docs/m1-baseline.md, step 1).

One Tinker sample per task: the GDP.pdf prompt plus the PDF content, through the cookbook renderer.
results.jsonl rows are `ExtractionResult` plus run fields: model, effort, truncated_at_page, prompt_tokens,
completion_tokens, finish ("stop" or "length"), prompt_sha256 (decoded prompt in the object store).
"""

import asyncio
import json
import time
from pathlib import Path

from gdpsolve import trace
from gdpsolve.eval import pdf_input
from gdpsolve.schemas import ExtractionResult

# Tinker $ per 1M tokens: (prefill, sample)
PRICES = {"thinkingmachines/Inkling-Small": (0.58, 1.44)}
DEFAULT_EFFORT = 0.9  # tinker_cookbook.renderers.tml_v0.DEFAULT_EFFORT
# tml_v0 <|end_message|><|content_model_end_sampling|>: closes a cut-off completion so it can be parsed
_CLOSE = [200010, 200006]


def done(results: Path) -> dict[str, dict]:
    """Rows already in results.jsonl by request_id; a later row replaces an earlier one."""
    if not results.exists():
        return {}
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    return {r["request_id"]: r for r in rows}


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prefill, sample = PRICES[model]
    return (prompt_tokens * prefill + completion_tokens * sample) / 1e6


def prepare(model: str, tasks: list[dict], data: Path, token_budget: int, effort: float = DEFAULT_EFFORT):
    """Render each task's prompt. Returns (renderer, [(task, ModelInput, truncated_at_page, metadata)])."""
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(model), get_tokenizer(model))

    def count_tokens(parts: list[dict]) -> int:
        return renderer.build_generation_prompt([{"role": "user", "content": parts}]).length

    prepared = []
    for task in tasks:
        path = data / task["pdf_path"]
        parts, truncated_at = pdf_input.content(path, count_tokens, token_budget)
        message = {"role": "user", "content": [{"type": "text", "text": task["prompt"]}, *parts]}
        prompt = renderer.build_generation_prompt([message], effort=effort)
        prepared.append((task, prompt, truncated_at, pdf_input.metadata(path)))
    return renderer, prepared


def answer_text(renderer, tokens: list[int]) -> tuple[str, bool]:
    """The completion's text part, never its thinking or control tokens; and whether it ended cleanly.

    A cut-off completion is closed with _CLOSE and parsed again, so a partial text part is kept and a
    completion cut off while thinking gives "".
    """
    from tinker_cookbook import renderers

    message, termination = renderer.parse_response(tokens)
    if termination.is_clean:
        return renderers.get_text_content(message), True
    message, termination = renderer.parse_response(tokens + _CLOSE)
    return (renderers.get_text_content(message) if termination.is_clean else ""), False


def store_prompt(renderer, prompt) -> str:
    """Save the decoded prompt (images as <image>) to the object store; returns its sha256."""
    from gdpsolve.verifier import store

    return store.put(trace.prompt_text(renderer.tokenizer, prompt).encode(), "text/plain").sha256


def backfill(model: str, tasks: list[dict], data: Path, token_budget: int, results: Path) -> int:
    """Set prompt_sha256 on existing rows by rebuilding their prompts with the same settings (no Tinker calls)."""
    rows = done(results)
    todo = [t for t in tasks if t["task_id"] in rows and "prompt_sha256" not in rows[t["task_id"]]]
    renderer, prepared = prepare(model, todo, data, token_budget)
    for task, prompt, _, _ in prepared:
        row = rows[task["task_id"]]
        if prompt.length != row["prompt_tokens"]:
            raise ValueError(f"{task['task_id']}: rebuilt prompt has {prompt.length} tokens, row has {row['prompt_tokens']}")
        row["prompt_sha256"] = store_prompt(renderer, prompt)
    results.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    return len(prepared)


async def answer_all(model: str, renderer, prepared: list, results: Path, max_tokens: int,
                     concurrency: int, effort: float) -> None:
    """Sample one answer per prepared task and append each row to results.jsonl as it finishes."""
    import tinker

    client = await tinker.ServiceClient().create_sampling_client_async(base_model=model)
    params = tinker.SamplingParams(max_tokens=max_tokens, stop=renderer.get_stop_sequences())
    gate = asyncio.Semaphore(concurrency)

    async def one(task, prompt, truncated_at, meta) -> None:
        async with gate:
            start = time.monotonic()
            run = {"model": model, "effort": effort, "truncated_at_page": truncated_at, "prompt_tokens": prompt.length,
                   "prompt_sha256": store_prompt(renderer, prompt)}
            try:
                resp = await client.sample_async(prompt, num_samples=1, sampling_params=params)
                tokens = resp.sequences[0].tokens
                answer, clean = answer_text(renderer, tokens)
                result = ExtractionResult(request_id=task["task_id"], document=meta, status="ok", answer=answer)
                run |= {"completion_tokens": len(tokens), "finish": "stop" if clean else "length"}
            except Exception as e:  # noqa: BLE001 -- recorded as a failed row; resume retries it
                result = ExtractionResult(request_id=task["task_id"], document=meta, status="failed",
                                          error=f"{type(e).__name__}: {e}")
            result.latency_s = round(time.monotonic() - start, 2)
            with results.open("a") as f:
                f.write(json.dumps(result.model_dump(mode="json") | run) + "\n")
            trace.call("baseline", id=result.request_id, prompt_sha256=run["prompt_sha256"],
                       prompt_tokens=prompt.length, completion_tokens=run.get("completion_tokens"),
                       finish=run.get("finish"), latency_s=result.latency_s, error=result.error)
            print(f"{result.request_id} {result.status} {result.latency_s}s", flush=True)

    await asyncio.gather(*(one(*p) for p in prepared))


def pending(tasks: list[dict], results: Path, rerun_truncated: bool = False) -> list[dict]:
    """Tasks without an ok row in results.jsonl; with rerun_truncated, also those whose answer hit max tokens."""
    ok = {rid for rid, r in done(results).items()
          if r["status"] == "ok" and not (rerun_truncated and r.get("finish") == "length")}
    return [t for t in tasks if t["task_id"] not in ok]
