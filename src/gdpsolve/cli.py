"""`gdpsolve` CLI: `gdpsolve <action> <resource>` (docs/TRD.md section 11)."""

import contextlib
from pathlib import Path
from typing import Annotated

import typer

from gdpsolve.eval import gdp_pdf

app = typer.Typer(no_args_is_help=True, help="GDPSolve: document extraction system and RL training pipeline.")
download_app = typer.Typer(no_args_is_help=True, help="Fetch datasets.")
app.add_typer(download_app, name="download")
synth_app = typer.Typer(no_args_is_help=True, help="Generate synthetic data (never from GDP.pdf).")
app.add_typer(synth_app, name="synth")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate a system or verifier; prints one number.")
app.add_typer(eval_app, name="eval")
score_app = typer.Typer(no_args_is_help=True, help="Score responses.")
app.add_typer(score_app, name="score")

Checker = Annotated[str, typer.Option(help="rubric | support (schemas.CHECKERS).")]
Checkpoint = Annotated[str | None, typer.Option(help="tinker:// sampler path; default: untrained base model.")]
Budget = Annotated[float, typer.Option(help="Abort if the estimated Tinker cost (USD) is higher.")]


def _checker(name: str, checkpoint: str | None):
    from gdpsolve.schemas import CHECKERS

    if name not in CHECKERS:
        raise typer.BadParameter(f"--checker must be one of {sorted(CHECKERS)}")
    return CHECKERS[name].model_copy(update={"checkpoint": checkpoint}) if checkpoint else CHECKERS[name]


@contextlib.contextmanager
def _traced(run_dir: Path, **config):
    """Trace a run into run_dir (gdpsolve.trace): config, metrics, Tinker call timing, calls.jsonl."""
    from gdpsolve import trace

    logger = trace.start(run_dir, config)
    try:
        yield logger
    finally:
        trace.stop(logger)
        typer.echo(f"trace: {run_dir}", err=True)


def _guard(estimate: float, budget: float) -> None:
    typer.echo(f"estimated Tinker cost: ${estimate:.2f} (budget ${budget:.2f})", err=True)
    if estimate > budget:
        raise typer.Exit(1)


@download_app.command("gdp-pdf")
def download_gdp_pdf(
    repo: Annotated[str, typer.Option(help="Hugging Face dataset repo.")] = gdp_pdf.DEFAULT_REPO,
    revision: Annotated[str, typer.Option(help="Pinned dataset revision.")] = gdp_pdf.DEFAULT_REVISION,
    out: Annotated[Path, typer.Option(help="Output directory.")] = gdp_pdf.DEFAULT_DIR,
) -> None:
    """Download the GDP.pdf benchmark (evaluation only; never train on it)."""
    path = gdp_pdf.download(repo, revision, out)
    pdfs = list(path.rglob("*.pdf"))
    typer.echo(f"downloaded {repo}@{revision[:8]} -> {path} ({len(pdfs)} PDFs)")


@synth_app.command("tasks")
def synth_tasks(
    n: Annotated[int, typer.Option(help="Tasks to request from the teacher model.")] = 40,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/synth"),
    seed: int = 0,
    budget: Budget = 1.0,
) -> None:
    """Synthetic check set: tasks plus rubric and support decisions with labels by construction."""
    import asyncio

    from gdpsolve.verifier import synth

    _guard(n * (400 * 0.58 + 1500 * 1.44) / 1e6, budget)  # Inkling-Small prefill/sample $/1M tokens
    counts = synth.build(asyncio.run(synth.generate_tasks(n, seed)), out, seed)
    typer.echo(f"{counts} -> {out}")


@score_app.command("verifier")
def score_verifier(checker: Checker = "rubric", checkpoint: Checkpoint = None) -> None:
    """Score one request: VerifyRequest (rubric) or SupportRequest (support) JSON on stdin, result on stdout."""
    import asyncio
    import sys

    from gdpsolve.schemas import SupportRequest, VerifyRequest
    from gdpsolve.verifier import support, verify

    cfg = _checker(checker, checkpoint)
    raw = sys.stdin.read()
    if checker == "rubric":
        result = asyncio.run(verify(VerifyRequest.model_validate_json(raw), cfg))
    else:
        result = asyncio.run(support(SupportRequest.model_validate_json(raw), cfg))
    typer.echo(result.model_dump_json())


@eval_app.command("verifier")
def eval_verifier(
    checker: Checker = "rubric",
    checkpoint: Checkpoint = None,
    data: Annotated[str, typer.Option(help="synthetic | gdp-pdf")] = "synthetic",
    synth_dir: Annotated[Path, typer.Option(help="Output of `synth tasks`.")] = Path("data/synth"),
    run: Annotated[Path, typer.Option(help="gdp-pdf: run with results.jsonl, grades.jsonl, summary.json.")] = Path(
        "var/runs/m1"),
    budget: Budget = 1.0,
) -> None:
    """Brier reward of a checker against labels: synthetic (by construction) or gdp-pdf (official verdicts)."""
    import asyncio
    import json

    cfg = _checker(checker, checkpoint)
    if data == "gdp-pdf":
        with _traced(run / f"verifier-{checker}", checker=checker, cfg=cfg.model_dump(), data=data, run=str(run)):
            return _eval_verifier_gdp_pdf(checker, cfg, run, budget)
    if data != "synthetic":
        raise typer.BadParameter("--data must be synthetic or gdp-pdf")
    from gdpsolve.verifier import synth

    rows = [json.loads(line) for line in (synth_dir / f"{checker}.jsonl").read_text().splitlines()]
    _guard(len(rows) * (400 if checker == "support" else 300) * 0.70 / 1e6, budget)  # prefill, ≤ $0.70/1M
    trace_dir = Path("var/runs") / f"synthetic-{checker}"
    with _traced(trace_dir, checker=checker, cfg=cfg.model_dump(), data=data, synth_dir=str(synth_dir)) as logger:
        if checker == "rubric":
            ps = asyncio.run(synth.score_rubric(rows, cfg))
            report = synth.rubric_report(rows, ps)
        else:
            ps = asyncio.run(synth.score_support(rows, cfg))
            report = {"brier_reward": synth.brier_reward(list(zip(ps, [r["label"] for r in rows], strict=True))),
                      "n": len(rows), "unscored": sum(p is None for p in ps)}
        logger.log_metrics(report)
    typer.echo(json.dumps(report), err=True)
    typer.echo(f"{report['brier_reward']:.4f}")


def _eval_verifier_gdp_pdf(checker: str, cfg, run: Path, budget: float) -> None:
    """M1: verify() against official GDP.pdf verdicts; prints the Brier reward and writes report.html."""
    import asyncio
    import json
    from datetime import UTC, datetime

    from gdpsolve.eval import baseline, report

    results = {rid: r for rid, r in baseline.done(run / "results.jsonl").items() if r["status"] == "ok"}
    grades = [json.loads(line) for line in (run / "grades.jsonl").read_text().splitlines()]
    decisions = report.decisions(gdp_pdf.load_tasks(), grades, results)
    _guard(report.estimate(decisions), budget)
    asyncio.run(report.score(decisions, cfg))

    reward, n = report.brier(decisions, cfg.model_ref)
    summary = json.loads((run / "summary.json").read_text())
    (run / "report.html").write_text(report.render({
        "meta": {
            "checker": checker, "data": "gdp-pdf", "labels": "Official Gemini 3.5 Flash verdicts (eval_only)",
            "dataset": f"{gdp_pdf.DEFAULT_REPO}@{gdp_pdf.DEFAULT_REVISION[:7]}",
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "sample": False,
            "baseline": {k: summary[k] for k in ("model", "judge", "all_pass", "mean_criteria", "n_tasks",
                                                 "effort_counts", "hit_max_tokens")},
        },
        "models": [{"ref": cfg.model_ref, "label": "Untrained verifier", "role": "verifier"}],
        "decisions": decisions,
    }, request_ids=set(results)))
    typer.echo(json.dumps({"brier_reward": reward, "n": n, "unscored": len(decisions) - n,
                           "report": str(run / "report.html")}), err=True)
    typer.echo(f"{reward:.4f}")


@eval_app.command("gdp-pdf")
def eval_gdp_pdf(
    system: Annotated[str, typer.Option(help="baseline (one sample per task) | process (process(): router + "
                                             "workers) | service (extract through the service SDK).")] = "baseline",
    model: Annotated[str, typer.Option(help="baseline: Tinker base model.")] = "thinkingmachines/Inkling-Small",
    out: Annotated[Path | None, typer.Option(help="Run directory for results.jsonl; default var/runs/m1 (baseline), "
                                                  "var/runs/m2 (process), or var/runs/service.")] = None,
    data: Annotated[Path, typer.Option(help="GDP.pdf directory.")] = gdp_pdf.DEFAULT_DIR,
    limit: Annotated[int | None, typer.Option(help="Only the first N tasks.")] = None,
    max_tokens: Annotated[int, typer.Option(help="Completion token cap.")] = 4096,
    token_budget: Annotated[int, typer.Option(help="PDF content token budget.")] = 120_000,
    concurrency: Annotated[int, typer.Option(help="Parallel Tinker requests.")] = 8,
    effort: Annotated[float, typer.Option(help="Reasoning effort for the tml_v0 renderer, in [0, 1).")] = 0.9,
    rerun_truncated: Annotated[bool, typer.Option(help="Also re-answer rows whose answer hit max tokens.")] = False,
    workers: Annotated[int, typer.Option(help="process: worker threads for each document.")] = 4,
    url: Annotated[str, typer.Option(help="service: URL of `gdpsolve serve`.")] = "http://127.0.0.1:8787",
    budget: Budget = 5.0,
) -> None:
    """Answers on GDP.pdf (evaluation only); resumes from results.jsonl.

    Example: gdpsolve eval gdp-pdf --system process --limit 20 --out var/runs/m2
    """
    import asyncio
    import json

    from gdpsolve.eval import baseline

    if system not in ("baseline", "process", "service"):
        raise typer.BadParameter("--system must be baseline, process, or service")
    tasks = gdp_pdf.load_tasks(data)[:limit]
    if system == "service":
        return _eval_gdp_pdf_service(tasks, out or Path("var/runs/service"), data, url, concurrency)
    if system == "process":
        return _eval_gdp_pdf_process(tasks, out or Path("var/runs/m2"), data, workers, budget)
    out = out or Path("var/runs/m1")
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    todo = baseline.pending(tasks, results, rerun_truncated)
    renderer, prepared = baseline.prepare(model, todo, data, token_budget, effort)
    prompt_tokens = sum(p[1].length for p in prepared)
    typer.echo(f"{len(prepared)} tasks to answer ({len(tasks) - len(todo)} done): {prompt_tokens:,} prompt tokens "
               f"+ up to {max_tokens:,} sample tokens each", err=True)
    _guard(baseline.cost(model, prompt_tokens, max_tokens * len(prepared)), budget)
    dataset = f"{gdp_pdf.DEFAULT_REPO}@{gdp_pdf.DEFAULT_REVISION}"
    with _traced(out / "baseline", model=model, token_budget=token_budget, max_tokens=max_tokens, effort=effort,
                 rerun_truncated=rerun_truncated, dataset=dataset) as logger:
        before = len(results.read_text().splitlines()) if results.exists() else 0
        asyncio.run(baseline.answer_all(model, renderer, prepared, results, max_tokens, concurrency, effort))

        new = [json.loads(line) for line in results.read_text().splitlines()[before:]]
        ids = {t["task_id"] for t in tasks}
        rows = [r for r in baseline.done(results).values() if r["request_id"] in ids]
        counts = {"ok": sum(r["status"] == "ok" for r in rows), "failed": sum(r["status"] == "failed" for r in rows),
                  "truncated_documents": sum(r["truncated_at_page"] is not None for r in rows),
                  "hit_max_tokens": sum(r.get("finish") == "length" for r in rows),
                  "tinker_cost_this_run": sum(baseline.cost(model, r["prompt_tokens"], r.get("completion_tokens", 0))
                                              for r in new)}
        logger.log_metrics(counts)
    typer.echo(f"{results}: {json.dumps(counts)}")


def _eval_gdp_pdf_process(tasks: list[dict], out: Path, data: Path, workers: int, budget: float) -> None:
    """M2: process() for each task without an ok row; logs status counts, cost, and latency percentiles."""
    import json

    from gdpsolve import run
    from gdpsolve.eval import baseline

    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    todo = baseline.pending(tasks, results)
    typer.echo(f"{len(todo)} tasks to answer ({len(tasks) - len(todo)} done)", err=True)
    _guard(len(todo) * run.MAX_COST_PER_DOC, budget)
    dataset = f"{gdp_pdf.DEFAULT_REPO}@{gdp_pdf.DEFAULT_REVISION}"
    with _traced(out / "process", system="process", model=run.MODEL, effort=run.EFFORT, workers=workers,
                 max_cost_per_doc=run.MAX_COST_PER_DOC, dataset=dataset) as logger:
        new = run.run_eval(todo, data, results, workers=workers)
        ids = {t["task_id"] for t in tasks}
        rows = [r for r in baseline.done(results).values() if r["request_id"] in ids]
        latencies = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)

        def pct(q: float) -> float | None:
            return latencies[min(len(latencies) - 1, int(q * len(latencies)))] if latencies else None

        counts = {**{s: sum(r["status"] == s for r in rows) for s in ("ok", "partial", "failed", "password_required")},
                  "tinker_cost_this_run": sum(r["cost"] for r in new), "latency_p50": pct(0.5),
                  "latency_p95": pct(0.95)}
        logger.log_metrics(counts)
    typer.echo(f"{results}: {json.dumps(counts)}")


def _eval_gdp_pdf_service(tasks: list[dict], out: Path, data: Path, url: str, concurrency: int) -> None:
    """Each task without an ok row through the service SDK (upload, then extract), `concurrency` at a time.

    Rows have the baseline format; latency_s is the client wall time for the document. Writes run.json.
    """
    import json
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from gdpsolve.client import GDPSolve
    from gdpsolve.eval import baseline, pdf_input
    from gdpsolve.schemas import ExtractionResult

    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    todo = baseline.pending(tasks, results)
    sdk = GDPSolve(url)
    config = sdk.health()["config"]
    typer.echo(f"{len(todo)} tasks to answer ({len(tasks) - len(todo)} done) via {url}: {json.dumps(config)}",
               err=True)
    meta = {t["task_id"]: pdf_input.metadata(data / t["pdf_path"]) for t in todo}  # pdfium is not thread-safe
    lock = threading.Lock()
    new: list[tuple[dict, float]] = []

    def one(task: dict) -> None:
        start = time.monotonic()
        run = {"model": config["model"], "effort": config["effort"], "truncated_at_page": None}
        cost = 0.0
        try:
            file_id = sdk.upload(data / task["pdf_path"])["file_id"]
            r = sdk.extract.run(input=file_id, instructions={"schema": {}, "system_prompt": task["prompt"]})
            result = ExtractionResult(request_id=task["task_id"], document=meta[task["task_id"]], status="ok",
                                      answer=r["result"][0]["answer"])
            run |= {"truncated_at_page": r["truncated_at_page"], "prompt_tokens": r["usage"]["prompt_tokens"],
                    "completion_tokens": r["usage"]["completion_tokens"], "finish": r["finish"],
                    "prompt_sha256": r["prompt_sha256"]}
            cost = r["usage"]["cost_usd"]
        except Exception as e:  # noqa: BLE001 -- recorded as a failed row; resume retries it
            result = ExtractionResult(request_id=task["task_id"], document=meta[task["task_id"]], status="failed",
                                      error=f"{type(e).__name__}: {e}")
        result.latency_s = round(time.monotonic() - start, 2)
        row = result.model_dump(mode="json") | run
        with lock:
            with results.open("a") as f:
                f.write(json.dumps(row) + "\n")
            new.append((row, cost))
        typer.echo(f"{result.request_id} {result.status} {result.latency_s}s", err=True)

    start = time.monotonic()
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, todo))
    wall = time.monotonic() - start
    latencies = sorted(row["latency_s"] for row, _ in new)

    def pct(q: float) -> float | None:
        return latencies[min(len(latencies) - 1, int(q * len(latencies)))] if latencies else None

    stats = {"system": "service", "n": len(new), "wall_s": round(wall, 3),
             "throughput_docs_per_min": round(len(new) / wall * 60, 2), "latency_p50": pct(0.5),
             "latency_p95": pct(0.95), "cost_usd": sum(c for _, c in new),
             "ok": sum(row["status"] == "ok" for row, _ in new), "url": url, "config": config}
    (out / "run.json").write_text(json.dumps(stats, indent=2) + "\n")
    typer.echo(f"{results}: {json.dumps(stats)}")


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8787,
    workers: Annotated[int, typer.Option(help="Worker threads: jobs that run at the same time.")] = 8,
    parser: Annotated[str, typer.Option(help="native (text layer; pages without one as images) | paddle "
                                             "(pages without a text layer through PaddleOCR-VL).")] = "native",
    model: Annotated[str, typer.Option(help="Tinker base model.")] = "thinkingmachines/Inkling-Small",
    checkpoint: Checkpoint = None,
    effort: Annotated[float, typer.Option(help="Reasoning effort for the tml_v0 renderer, in [0, 1).")] = 0.3,
    max_tokens: Annotated[int, typer.Option(help="Completion token cap.")] = 8192,
    token_budget: Annotated[int, typer.Option(help="PDF content token budget.")] = 120_000,
) -> None:
    """Run the document service (docs/service-spec.md) until stopped."""
    from gdpsolve import run, service

    if parser not in ("native", "paddle"):
        raise typer.BadParameter("--parser must be native or paddle")
    if parser == "paddle":
        run.ensure_ocr_server()
    cfg = service.Config(parser=parser, model=model, checkpoint=checkpoint, effort=effort, max_tokens=max_tokens,
                         token_budget=token_budget)
    typer.echo(f"serving on http://{host}:{port} with {workers} workers: {cfg.settings()}", err=True)
    service.serve(cfg, host, port, workers)


@app.command("extract")
def extract(
    path: Annotated[Path, typer.Argument(help="A PDF, or a directory: one request for each *.pdf in it.")],
    instructions: Annotated[str, typer.Option("--instructions", "-i", help="What to extract.")],
    password: Annotated[list[str] | None, typer.Option(help="Password to try; repeat for more.")] = None,
    workers: Annotated[int, typer.Option(help="Worker threads for each document.")] = 4,
) -> None:
    """Run process() on each PDF; one ExtractionResult JSON per line on stdout."""
    from pydantic import SecretStr

    from gdpsolve import run
    from gdpsolve.schemas import ExtractionRequest

    pdfs = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
    for pdf in pdfs:
        req = ExtractionRequest(path=pdf, instructions=instructions, passwords=[SecretStr(p) for p in password or []])
        typer.echo(run.process(req, workers=workers).model_dump_json())


@score_app.command("gdp-pdf")
def score_gdp_pdf(
    results: Annotated[Path, typer.Argument(help="results.jsonl from `eval gdp-pdf`.")],
    judge: Annotated[str, typer.Option(help="Official judge model.")] = "google/gemini-3.5-flash",
    data: Annotated[Path, typer.Option(help="GDP.pdf directory.")] = gdp_pdf.DEFAULT_DIR,
    concurrency: Annotated[int, typer.Option(help="Tasks graded in parallel.")] = 8,
    budget: Annotated[float, typer.Option(help="Abort if the estimated judge cost (USD) is higher.")] = 5.0,
) -> None:
    """Official GDP.pdf grades (evaluation only): grades.jsonl and summary.json next to results.jsonl."""
    import asyncio
    import json
    from collections import Counter

    from gdpsolve import trace
    from gdpsolve.eval import baseline, official

    rows = list(baseline.done(results).values())
    answers = {r["request_id"]: r["answer"] for r in rows if r["status"] == "ok"}
    tasks = [t for t in gdp_pdf.load_tasks(data) if t["task_id"] in answers]
    tokens_in, tokens_out, estimate = official.estimate(tasks, answers)
    typer.echo(f"{len(tasks)} tasks, {sum(len(official.rubrics(t)) for t in tasks)} judge calls: about "
               f"{tokens_in:,} input + {tokens_out:,} output tokens; estimated judge cost ${estimate:.2f} "
               f"at an assumed ${official.JUDGE_PRICE[0]}/${official.JUDGE_PRICE[1]} per 1M (budget ${budget:.2f})",
               err=True)
    if estimate > budget:
        raise typer.Exit(1)
    with _traced(results.parent / "official", judge=judge, scorer_commit="7a72a51") as logger:
        values, grades, usage = asyncio.run(official.grade_all(judge, tasks, answers, concurrency))
        for g in grades:
            trace.call("judge", task_id=g["task_id"], criterion_id=g["criterion_id"], result=g["result"],
                       rationale=g["rationale"])
        used = usage.get(judge)
        summary = {
            "model": rows[0].get("model") if rows else None,
            "judge": judge,
            **official.aggregate(list(values.values())),
            "n_tasks": len(values),
            "unscored": sum(g["result"] == "unscored" for g in grades),
            "failed": sum(r["status"] == "failed" for r in rows),
            "truncated": sum(r["truncated_at_page"] is not None for r in rows),
            "effort_counts": dict(Counter(str(r.get("effort")) for r in rows)),
            "hit_max_tokens": sum(r.get("finish") == "length" for r in rows),
            "judge_tokens": used and {"input": used.input_tokens, "output": used.output_tokens,
                                      "reasoning": used.reasoning_tokens},
            "judge_cost": used and used.total_cost,
        }
        logger.log_metrics({k: v for k, v in summary.items() if isinstance(v, int | float)})

    (results.parent / "grades.jsonl").write_text("".join(json.dumps(g) + "\n" for g in grades))
    (results.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(json.dumps(summary, indent=2))

