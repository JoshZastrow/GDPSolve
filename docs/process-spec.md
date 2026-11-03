# `process()` v0: implementation spec (M2)

Design source: [`architecture.md`](architecture.md). Baseline to beat: [`m1-baseline.md`](m1-baseline.md).
This spec is written to about 80% of ASD-STE100. Code names are not changed.

## 1. Purpose

`process()` changes one `ExtractionRequest` into one `ExtractionResult`.
A router gives jobs to workers. A worker uses tools to read the document.
The worker sends a report. The router merges the reports into the result.

The M1 baseline reads the PDF in one prompt. On GDP.pdf it gets these results:
- strict pass rate (`all_pass`): 0.10
- mean rubric score (`mean_criteria`): 0.556

`process()` v0 must do better on the same tasks, and it must not use more than 300 s per document.

## 2. Scope

**In v0:**
- `ingest`: hash the file, try the passwords, read the metadata, and put the bytes in the store.
- The router: `plan`, `react` (for `split` only), and `merge`.
- A pool of worker threads, each with the `def worker(): while True:` loop.
- 8 tools: `inspect_pdf`, `read_native_text`, `render`, `crop`, `recognize`, `split`, `post`, `report`.
- PaddleOCR-VL in a local GGUF server (`llama serve`), with a warm start.
- The Thread, kept in memory.
- Three limits: a deadline, a step limit, and a cost limit.
- Tracing, with `gdpsolve.trace`.
- The CLI: `gdpsolve eval gdp-pdf --system process`, and `gdpsolve extract`.

**Not in v0:**
- The SQLite queue, leases, and separate worker processes.
- The tools `detect_layout`, `tile`, `correct_image`, and `validate`.
- New jobs from a `handback` message.
- A tool memo that stays after the process stops.
- RL training.

## 3. Rules for the work

### 3.1 Test-driven development

Do these steps for each task:
1. Write the tests first.
2. Run the tests. Make sure that they fail.
3. Commit the failing tests.
4. Write the code. Make the tests pass.
5. Run `uv run ruff check .` and `uv run --extra train pytest`. Both must pass.

Tests must run offline. Use `FakePolicy`, `FakeRecognizer`, and the PDFs that the tests make.
A live test must skip unless `RUN_LIVE=1` is set.

### 3.2 Files

Make only these changes:

| File | Change |
|---|---|
| `src/gdpsolve/run.py` | New. All code for `process()`. |
| `tests/test_run.py` | New. All tests for `process()`. |
| `src/gdpsolve/cli.py` | Edit. Add `--system` to `eval gdp-pdf`. Add the `extract` command. |
| `pyproject.toml` | Edit. Add `pypdf` to the dev group, for the encrypted test PDFs. |

`run.py` has these sections, in this order:
config, ingest, tools, recognizer, policy, worker, router, process, eval.
Each section starts with a comment line: `# --- <name> ---`.
Each task changes only its sections.

### 3.3 Use the code that exists

Do not write new code for these functions:

| Need | Use |
|---|---|
| PDF metadata | `gdpsolve.eval.pdf_input.metadata` |
| Text of one page | `gdpsolve.eval.pdf_input._text`. Rename it to `page_text`. |
| Store bytes by hash | `gdpsolve.verifier.store.put`, `get` |
| Tracing | `gdpsolve.trace.start`, `call`, `stop`, `prompt_text` |
| Data types | `schemas.py`: `ExtractionRequest`, `ExtractionResult`, `Document`, `Job`, `Report`, `ReportArgs`, `Message`, `Region`, `SourceSpan`, `ExtractedField` |
| Tool arguments | `schemas.TOOLBOX` models: `InspectPdf`, `ReadNativeText`, `Render`, `Crop`, `Recognize`, `Split`, `Post`, `ReportArgs` |
| GDP.pdf tasks | `gdpsolve.eval.gdp_pdf.load_tasks` |
| Official scores | `gdpsolve score gdp-pdf`. Do not change it. |
| Tool calls in Inkling | `tml_v0` renderer: `create_conversation_prefix_with_tools`, `build_generation_prompt(..., effort=)`, and `parse_response`, which gives `tool_calls` |

## 4. Design

### 4.1 `process()`

```python
def process(req, policy=None, recognizer=None, workers=4) -> ExtractionResult:
    doc = ingest(req)                     # can return a result now: failed or password_required
    run = Run(doc, req, policy or TinkerPolicy(), recognizer or PaddleRecognizer())
    threads = [Thread(target=worker, args=(run,), daemon=True) for _ in range(workers)]
    run.jobs.put(plan(doc, req))
    while not run.final_ended() and time() < run.deadline:
        msg = run.events.get(timeout=1)   # split messages from the split tool, or None
        for job in react(doc, req, msg, run):
            run.jobs.put(job)
        run.put_merge_job_when_children_ended()
    for _ in threads:
        run.jobs.put(None)                # stop signal
    return merge(doc, req, run.reports)
```

`Run` holds the state of one request:
- `jobs`: a `queue.Queue[Job | None]`
- `events`: a `queue.Queue[Message]`, which gets `split` messages
- `thread`: a `list[Message]`, the Thread
- `reports`: a `dict[str, Report]` (job id to latest report)
- `regions`: a `dict[str, Region]`
- `memo`: a `dict[str, str]`, the tool cache
- `cost`: a float, in USD
- `lock`: a `threading.Lock`. Use it for every write to `thread`, `reports`, `regions`, `memo`, and `cost`.
- `deadline`: `start + req.deadline_s`

### 4.2 `worker()` and `run_job()`

```python
def worker(run):
    while True:
        job = run.jobs.get()
        if job is None:
            return
        run_job(run, job)
```

`run_job(run, job)` does these steps:
1. Make the first messages: the instructions, the pages, the budget, and the Thread so far.
2. Ask the policy for one turn: `run.policy.step(messages) -> Turn`.
3. Run the tool of the turn. If the tool raises an error, put the error text in the observation.
4. Add the observation. It contains the tool result, the new Thread messages from other jobs, and the budget left.
5. Stop when the tool is `report` with state `ready` or `failed`.
6. Force one last `report` turn when one of these limits is reached:
   - the time is `deadline - 30 s`
   - the step count is `job.max_steps`
   - the cost of the request is more than `MAX_COST_PER_DOC` (default $0.15)
7. If the last turn does not give a report, write a `failed` report.

### 4.3 Policy

```python
class Turn(NamedTuple):
    tool: str
    args: dict
    prompt_tokens: int
    completion_tokens: int

class Policy(Protocol):
    def step(self, messages: list) -> Turn: ...
```

- `TinkerPolicy` uses `thinkingmachines/Inkling-Small` with the `tml_v0` renderer.
  - The tool specs are the JSON Schemas of the 8 `TOOLBOX` models.
  - Use `effort=0.3`. Use `max_tokens=4096` for each step.
  - Add the cost of each step to `run.cost`. Prices: prefill $0.58, sample $1.44 for each 1M tokens.
  - Write one trace record for each step: `trace.call("step", job_id=..., tool=..., prompt_tokens=..., completion_tokens=..., prompt=trace.prompt_text(...))`.
- `FakePolicy(turns)` gives the turns in a list, one turn for each step. It is only for tests.

### 4.4 Tools

| Tool | Does | Returns to the model |
|---|---|---|
| `inspect_pdf` | Reads the page size, rotation, text layer, and image cover | Text |
| `read_native_text` | Reads the text layer of one page, or of one box | Text |
| `render` | Makes a PNG of one page at `dpi`. Puts it in the store. | A `Region` id and the image |
| `crop` | Makes a PNG of one box at `dpi`. Puts it in the store. | A `Region` id and the image |
| `recognize` | Runs PaddleOCR-VL on one `Region` | Text, HTML, or LaTeX |
| `split` | Adds a `split` message to the Thread and to `run.events` | Text: "split requested" |
| `post` | Adds a `note`, `unrelated`, `follow`, or `handback` message to the Thread | Text: "posted" |
| `report` | Writes the report of the job | Text: "reported" |

Rules:
- The first five tools are pure. Keep their results in `run.memo` under `sha256(doc, tool, args)`.
- A `Region` id is a hash of the document, page, box, dpi, and operations. The same view gets the same id.
- Only `render` and `crop` send an image to the model.

### 4.5 Recognizer (local GGUF server)

PaddleOCR-VL runs in a local llama.cpp server, not in the Python process.
The server uses the official GGUF build `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` and Metal on Apple Silicon.
It gives an OpenAI-compatible API.

Server command (`llama` comes from `curl -LsSf https://llama.app/install.sh | sh`):

```bash
llama serve -hf PaddlePaddle/PaddleOCR-VL-1.6-GGUF --temp 0 --port 9931 -np 4 --alias paddleocr-vl
```

- `-hf` downloads the model and the multimodal projector at the first start. Later starts use the cache.
- `-np 4` gives 4 slots, so 4 worker threads can call it at the same time.
- The log goes to `var/log/llama-serve.log`.

```python
OCR_URL = os.environ.get("GDPSOLVE_OCR_URL", "http://127.0.0.1:9931")

def ensure_ocr_server(url=OCR_URL, timeout_s=300) -> float: ...   # warm start; returns warm-up seconds

class PaddleRecognizer:
    def __call__(self, image: PIL.Image.Image, mode: str) -> str: ...
```

**Warm start: `ensure_ocr_server()`.** Do these steps one time, before the first document:
1. Send `GET {url}/health`. If the server answers 200, go to step 4.
2. If not, start the server command above as a subprocess. Write its output to the log file.
3. Send `GET {url}/health` each 2 s. If there is no 200 after `timeout_s`, raise an error that gives the log path.
4. Send one warm-up request: a small image with the word "warm" and the prompt `OCR:`.
   This loads the weights and compiles the Metal kernels before the first real job.
5. Return the warm-up time. Record it as the metric `ocr_warmup_s`.

Call `ensure_ocr_server()` at the start of `gdpsolve eval gdp-pdf --system process` and `gdpsolve extract`.
`process()` also calls it. A second call does nothing.
Do not stop the server at the end of the run. The next run uses the warm server.

**Recognize call.** `PaddleRecognizer` sends `POST {url}/v1/chat/completions` with the standard library (`urllib`):
- `model`: `paddleocr-vl`
- `messages`: one user message. It has the image as a `data:image/png;base64,...` URL and then the prompt text.
- `temperature`: 0. `max_tokens`: 2048.

It returns the text of the first choice. The server has slots, so there is no lock in Python.
Unit tests use `FakeRecognizer` and a fake HTTP server. The live test needs `RUN_LIVE=1`.

Give the model the prompt for the mode:

| `mode` | Prompt |
|---|---|
| `text`, `handwriting` | `OCR:` |
| `table` | `Table Recognition:` |
| `formula` | `Formula Recognition:` |
| `chart` | `Chart Recognition:` |
| `symbol` | `Spotting:` |

### 4.6 Router

- `plan(doc, req) -> Job`: make one root job for all pages.
  Set `max_steps = min(40, 8 + page_count // 5)`.
- `react(doc, req, msg, run) -> list[Job]`: do this only for a `split` message.
  - Make child jobs of at least 5 pages each. Make no more than 16 child jobs.
  - Make one `merge` job with `after` set to the child jobs. Put it in the queue only when all child jobs have ended.
  - If less than 120 s is left, return `[]`.
- `merge(doc, req, reports) -> ExtractionResult`. Do not call a model.
  1. If the merge job has a `ready` report, use its answer and fields.
  2. If not, and the root job has a `ready` report, use the root report.
  3. If not, join the answers of the `ready` reports. Set `status="partial"`.
  4. If no report is `ready`, set `status="failed"`.
  - Set `confidence` to the minimum field confidence. If there are no fields, set it to 0.0.

### 4.7 Eval and CLI

- `gdpsolve eval gdp-pdf --system process`: make one `ExtractionRequest` for each task. Call `process()`.
  Write one row to `results.jsonl` for each task. Use the same row format as the baseline.
  Add these fields to each row: `steps`, `cost`, `jobs`.
- `gdpsolve extract PATH -i "..."`: if PATH is a directory, make one request for each PDF. Write JSONL to stdout.
- `gdpsolve score gdp-pdf` reads the new `results.jsonl`. Do not change it.

## 5. Tasks

Each task is a GitHub issue in project "GDPSolve". The label gives the agent.

| # | Task | Agent | Starts after | Write these tests first |
|---|---|---|---|---|
| T0 | Make the skeleton of `run.py`: sections, `Run`, `Turn`, `Policy`, and stubs. Make the test helpers: `make_pdf`, `make_scan`, `FakePolicy`, `FakeRecognizer`. | agent-1 | — | `make_pdf` gives a PDF with readable text. Each stub raises `NotImplementedError`. |
| T1 | `ingest` | agent-1 | T0 | A plain PDF gives a `Document`. The right password opens an encrypted PDF. A wrong password gives `password_required`. The bytes are in the store. |
| T2 | Tools: `inspect_pdf`, `read_native_text`, `render`, `crop`, and the memo | agent-2 | T0 | The native text is the same as the page text. `render` gives a `Region` with a PNG in the store. The same arguments make one memo entry. |
| T3 | `ensure_ocr_server()` (warm start), `PaddleRecognizer` (GGUF server client), and the `recognize` tool | agent-2 | T0 | Each mode gives the correct prompt. The request body has the image URL, the prompt, and `temperature` 0. `ensure_ocr_server` returns at once when `/health` is 200, starts the server when it is not, and raises after the timeout (fake server and fake subprocess). A live test (`RUN_LIVE=1`) reads one word from a rendered page. |
| T4 | `run_job` and `worker()` | agent-3 | T0 | A `report` turn ends the job. A tool error becomes an observation. Each limit forces a report: steps, deadline, and cost. Messages from other jobs come in the next observation. |
| T5 | Router and `process()` | agent-1 | T1, T4 | One job runs from start to end. A `split` makes child jobs and then a merge job. A short deadline gives `partial`. A wrong password gives `password_required`. |
| T6 | `TinkerPolicy` | agent-3 | T0 | The tool specs come from `TOOLBOX`. A fixed completion becomes the correct `Turn` (real tokenizer; skip offline). |
| T7 | CLI: `eval gdp-pdf --system process`, and `extract` | agent-1 | T5 | `CliRunner` with `FakePolicy` writes `results.jsonl` rows in the baseline format. |
| T8 | Milestone check | lead | T2–T7 | Not applicable. See section 6. |

The tasks run in waves:
1. T0
2. T1, T2, T3, T4, T6, at the same time
3. T5
4. T7
5. T8

Each agent works in its own git worktree. The lead merges the work in wave order.

## 6. Milestone check (T8)

Do these steps:
1. Run a live smoke test on 3 synthetic PDFs. Do not spend more than $0.50.
2. Run `gdpsolve eval gdp-pdf --system process --limit 20 --out var/runs/m2`. Use the first 20 tasks in dataset order.
   Do not select the tasks by their results.
3. Run `gdpsolve score gdp-pdf var/runs/m2/results.jsonl`.
4. Compare `all_pass` and `mean_criteria` with M1 on the same 20 tasks.
5. Measure the p95 latency. It must be less than 300 s.
6. Write the results in this section.

Spend limits:
- Tinker: no more than $0.15 for each document, and no more than $3 for the run.
- Gemini: about $0.55 for the official judge on 20 tasks.

GDP.pdf is for evaluation only. Do not change prompts, tools, or limits because of these results and then run again.

## 7. Results

To be completed by T8.
