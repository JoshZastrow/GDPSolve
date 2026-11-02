# Architecture

How a document goes from a caller to an `ExtractionResult`. Diagram: [`diagrams/architecture.html`](diagrams/architecture.html). Data contract: [`src/gdpsolve/schemas.py`](../src/gdpsolve/schemas.py). Requirements: [TRD](TRD.md).

## Principles

- **One function, one job.** Each module does one thing and takes and returns schema objects.
- **One seam.** Every caller (eval, CLI, SDK) calls `process(request) -> ExtractionResult`. Nothing else is public.
- **Two kinds of state:**
  - small records in SQLite (`var/gdpsolve.db`)
  - blobs in a content-addressed object store (`var/objects/`)
- **One writer per table.** The router writes jobs. Workers write reports.
- **Derived state.** A job's state is its latest report. There is no status column to keep in sync.
- **Text between processes.** Records are JSON, transcripts are NDJSON, and CLI output is JSONL on stdout.

## Entry points

| Caller | Command / call | Does |
|---|---|---|
| Eval | `gdpsolve eval run` → `gdpsolve.eval.run(dataset_dir, out)` | One `ExtractionRequest(id=task_id, path=pdf, instructions=prompt)` per GDP.pdf row, then `process()`, then one result per line in `results.jsonl` |
| Eval | `gdpsolve eval score results.jsonl` | The judge grades each `answer` against its rubric, one call per criterion (the official GDP.pdf scorer), and writes `scores.jsonl`. Separate from `run` so scoring can be repeated without re-extracting. |
| CLI | `gdpsolve extract PATH -i "…"` | Expands a directory into one request per PDF, calls `process()` on each, and prints JSONL |
| SDK | `from gdpsolve import process` | Direct call |

## `process()`

```python
def process(req: ExtractionRequest) -> ExtractionResult:
    doc = ingest(req)                              # hash, decrypt, metadata; PDF bytes -> object store
    jobs = router.plan(doc, req)                   # 1..N Jobs (page shards, step budget)
    queue.put(jobs)
    reports = queue.wait(jobs, req.deadline_s)     # latest Report per job: ready / failed / deadline
    return router.merge(doc, req, reports)         # answer + fields + grounding + confidence
```

| Function | Signature | v0 |
|---|---|---|
| `ingest` | `(ExtractionRequest) -> Document` | Hashes the file, tries each password, reads metadata, and puts the decrypted PDF in the store. A missing password ends early with `status="password_required"`. |
| `router.plan` | `(Document, ExtractionRequest) -> list[Job]` | Heuristic: one job per document, or page shards for long documents. Later the trained router policy π_R (TRD §8). |
| `queue.put` / `queue.wait` | `(list[Job])`; `(list[Job], timeout) -> list[Report]` | SQLite insert; poll the latest report per job |
| `router.merge` | `(Document, ExtractionRequest, list[Report]) -> ExtractionResult` | One job: copy its report. Several: concatenate fields, then a final synthesis turn for the answer. |

The router is two functions, `plan` and `merge`, wrapped around the queue. It holds no state of its own.

## Worker

A worker is a separate process (`gdpsolve serve worker`). It claims a job, runs the agent loop, and publishes `Report`s. `gdpsolve eval run --workers N` starts N of them.

```python
def run_job(job: Job) -> None:
    messages = prompt(job)                                  # instructions, page count, tool specs
    transcript = []
    for _ in range(job.max_steps):
        turn = model.sample(messages, TOOLS)                # Inkling on Tinker
        transcript.append(turn)
        result = TOOLS[turn.tool](job, **turn.args)         # tool output goes back to the model
        messages.append(result)
        if turn.tool == "report" and turn.args["state"] != "progress":
            return
    report(job, state="failed", summary="step budget exhausted")
```

**Tools.** Each one takes a page and returns a value or an `ObjectRef`:

| Tool | Args | Returns | Stores |
|---|---|---|---|
| `render_page` | `page, dpi=150, crop=None` | The image, shown to the model | `image/png` |
| `page_text` | `page` | Text-layer words with bounding boxes | nothing (cheap to recompute) |
| `ocr` | `page` | Layout blocks, tables as HTML, reading order (PaddleOCR-VL-1.6) | `application/json` |
| `report` | `state, summary, answer?, fields?` | Acknowledgement | Writes a `Report` row. The transcript goes to the store as `application/x-ndjson`, and the report carries every artifact ref. |

`report` is the only way a worker talks to the router, and it carries both the **result** (answer and fields) and the **context** (artifact refs and transcript). A `progress` report is optional: it lets `queue.wait` show liveness and lets the router see partial work if the deadline hits.

## Object store

```python
def put(data: bytes, media_type: str) -> ObjectRef      # writes var/objects/ab/abcdef…
def get(ref: ObjectRef) -> bytes
```

The store is content-addressed (sha256), so identical renders and OCR outputs are stored once. Two functions are the whole interface, so moving to S3 later means replacing these two.

## Queue

SQLite in WAL mode, with two tables:

```sql
jobs    (id TEXT PRIMARY KEY, request_id TEXT, body JSON, attempts INT, lease_until TIMESTAMP)  -- router writes
reports (job_id TEXT, seq INT, body JSON, created_at TIMESTAMP, PRIMARY KEY (job_id, seq))   -- workers write
```

- **Claim:** one `UPDATE … SET lease_until = now + 120s, attempts = attempts + 1 WHERE id = (oldest job whose lease expired and with no ready/failed report) RETURNING body`.
- **Retry:** a crashed worker's lease expires and another worker claims the job. After 3 attempts, `queue.wait` treats the job as failed.
- **No status column:** a job is done when its latest report is `ready` or `failed`.

## Process model

```
gdpsolve eval run --workers 4
 ├─ main process: ingest → plan → put → wait → merge  (per task)
 └─ 4 × gdpsolve serve worker                         (claim → loop → report)
        shared: var/gdpsolve.db, var/objects/
        remote: Tinker (Inkling sampling)
```

There is no daemon or server. Everything runs on one machine, and only model sampling leaves it.

## Planned modules

```
src/gdpsolve/
  pipeline.py   process()
  ingest.py     ingest()
  router.py     plan(), merge()
  queue.py      put(), wait(), claim(), report()
  store.py      put(), get()
  worker.py     run_job(), TOOLS, `serve worker` loop
  eval/run.py   run(), score()
```

## What we took from dsh-superagent

`dsh-superagent-prototype/` (DeepSeek) is a proactive coding agent: a main session routes work, and one worker session runs per ticket. We kept its execution core and dropped its chat and authorization layer.

| dsh-superagent | GDPSolve | |
|---|---|---|
| `Engine` is the only writer of Tickets and Assignments | The router is the only writer of `jobs` | kept |
| Ticket plus one Executor session and directory | `Job` plus one worker loop, with its files in the object store | kept |
| Workers talk back only through `orchestrator_report` / `ack` / `read`, with a closed set of Report states | Workers talk back only through `report`, with states `progress` / `ready` / `failed` | kept, fewer states |
| Ticket stage derived from stored Reports | Job state is its latest Report | kept |
| Concurrency slots, transient-error retry with backoff | `--workers N`; lease expiry plus `attempts ≤ 3`; Tinker 429/5xx retried inside `model.sample` | kept |
| Source / Trigger / Thread / Eval / Mandate / Subscription / Channel | — | dropped: there is no chat intake or authorization, and the caller receives the result directly |
| Attention, handoff to a person, panel | — | dropped: no human in the loop |
| Background shell jobs, 30-minute slot rules | — | dropped: tools are synchronous and short |

## Training hook

Every `Report` carries its job's `transcript` (model turns and tool calls as NDJSON). Those transcripts are the rollouts SDPO++ trains on (TRD §8). Transcripts from GDP.pdf runs are evaluation-only and are never used for training (TRD §8.5).
