# Architecture

How a document goes from a caller to an `ExtractionResult`. Diagram: [`diagrams/architecture.html`](diagrams/architecture.html). Data contract: [`src/gdpsolve/schemas.py`](../src/gdpsolve/schemas.py). Requirements: [TRD](TRD.md).

## Principles

- **One function, one job.** Each module does one thing and takes and returns schema objects.
- **One seam.** Every caller (eval, CLI, SDK) calls `process(request) -> ExtractionResult`. Nothing else is public.
- **Two kinds of state:**
  - small records in SQLite (`var/gdpsolve.db`)
  - blobs in a content-addressed object store (`var/objects/`)
- **One writer per table.**
  - The router writes jobs.
  - Workers write reports and Thread messages, always through tools.
  - A worker never creates a job. It asks the router to.
- **Derived state.** A job's state is its latest report, and a page's disposition is its latest Thread message. Nothing keeps a status column in sync.
- **Push, not pull.** New Thread messages arrive in a worker's next observation. Workers never poll.
- **Text between processes.** Records are JSON, transcripts are NDJSON, and CLI output is JSONL on stdout.

## Entry points

| Caller | Command / call | Does |
|---|---|---|
| Eval | `gdpsolve eval gdp-pdf` → `gdpsolve.eval.run(dataset_dir, out)` | One `ExtractionRequest(id=task_id, path=pdf, instructions=prompt)` per GDP.pdf row, then `process()`, then one result per line in `results.jsonl` |
| Eval | `gdpsolve score gdp-pdf results.jsonl` | The judge grades each `answer` against its rubric, one call per criterion (the official GDP.pdf scorer), and writes `scores.jsonl`. Separate from `run` so scoring can be repeated without re-extracting. |
| CLI | `gdpsolve extract PATH -i "…"` | Expands a directory into one request per PDF, calls `process()` on each, and prints JSONL |
| SDK | `from gdpsolve import process` | Direct call |

## `process()`

```python
def process(req: ExtractionRequest) -> ExtractionResult:
    doc = ingest(req)                                   # hash, decrypt, metadata; PDF bytes -> object store
    queue.put([router.plan(doc, req)])                  # one root job
    while queue.open_jobs(req.id) and not past(req.deadline_s):
        for msg in queue.new_messages(req.id, kinds={"split", "handback"}):
            queue.put(router.react(doc, req, msg))      # child jobs + merge job, or a reassignment
        wait_briefly()
    return router.merge(doc, req, queue.reports(req.id))
```

| Function | Signature | v0 |
|---|---|---|
| `ingest` | `(ExtractionRequest) -> Document` | Hashes the file, tries each password, reads metadata, and puts the decrypted PDF in the store. A missing password ends early with `status="password_required"`. |
| `router.plan` | `(Document, ExtractionRequest) -> Job` | One root job over the whole document, with a step budget sized to the page count. |
| `router.react` | `(Document, ExtractionRequest, Message) -> list[Job]` | `split`: child `extract` jobs over the requested shards, plus one `merge` job with `after` set to the children. `handback`: one new job for the returned pages. It may cap shard count or reject the message (returning `[]`) to protect the deadline. |
| `router.merge` | `(Document, ExtractionRequest, list[Report]) -> ExtractionResult` | Copies the final job's `ready` report (the merge job if there is one, otherwise the root job). If that job failed or missed the deadline, it falls back to concatenating the other reports with `status="partial"`. No model call. |
| `queue.*` | `put`, `open_jobs`, `new_messages`, `reports` | SQLite reads and inserts |

The router is three functions around the queue. It holds no state of its own, and it is the only code that writes jobs. This follows dsh-superagent's main session, which alone creates Tickets, including when an Executor hands work back.

## Worker

A worker is a separate process (`gdpsolve serve worker`). It claims a job, runs the agent loop, and publishes through tools. `gdpsolve eval gdp-pdf --workers N` starts N of them.

**The router decides *what* to work on and the worker decides *how*.** There is no fixed pipeline. The worker gets the job and a toolbox, and the policy π_W picks the next tool call from what it has seen so far. That sequence of calls (the trajectory) is what training improves (see [Learned workflow](#learned-workflow)).

```python
def run_job(job: Job) -> None:
    messages = prompt(job, queue.thread(job.request_id))      # instructions, pages, budget, tool specs, Thread so far
    seen = queue.last_seq(job.request_id)
    transcript = []
    for step in range(job.max_steps):
        turn = model.sample(messages, TOOLBOX)                # π_W: Inkling (LoRA) on Tinker
        call = tools.run(job, turn.tool, turn.args)           # memoized ToolCall; split/post append to the Thread
        new = queue.thread(job.request_id, after=seen, exclude_job=job.id)
        seen = max([seen] + [m.seq for m in new])
        messages.append(observe(call, new, budget_left=job.max_steps - step - 1))
        transcript.append((turn, call, new))
        if turn.tool == "report" and turn.args["state"] != "progress":
            return
    report(job, state="failed", summary="step budget exhausted")
```

### Toolbox

The toolbox is the worker's action space, `TOOLBOX` in `schemas.py`. Each tool has a typed args model, and its JSON Schema is the spec the model sees.

| Tool | Args | Returns |
|---|---|---|
| `inspect_pdf` | `page` | Page facts: size, rotation, text layer present, image coverage, native outline entries |
| `render` | `page, dpi` | `Region` (full page), shown to the model |
| `detect_layout` | `page` | Typed blocks with bboxes: text, heading, table, figure, footnote, header, footer |
| `crop` | `page, bbox, padding, dpi` | `Region`, shown to the model |
| `tile` | `page, tile_size, overlap` | `Region`s for a large page |
| `correct_image` | `region, operation` | New `Region`: rotate, deskew, perspective, dewarp, denoise or contrast |
| `recognize` | `region, mode` | Text, handwriting, table HTML, formula, chart data, or symbols (PaddleOCR-VL-1.6) |
| `read_native_text` | `page, bbox?` | Text-layer words with bboxes, which is free and exact when the layer exists |
| `validate` | `output, constraints` | Errors from schema, type, required-field, sum and date/unit checks |
| `split` | `pages_per_job, pages?` | Appends a `split` message. The router decides whether to create the child jobs and the merge job. |
| `post` | `kind, pages, note` | Appends a `handback`, `unrelated`, `follow` or `note` message to the Thread |
| `report` | `state, summary, answer?, fields?, pages_read?` | Publishes to the router. `ready` or `failed` ends the job. |

**Design rules:**
- **Regions are handles.** Image tools return a `Region` id, and later tools take that id. A call chain such as `crop → correct_image → recognize` composes like a pipe, and the image bytes stay in the object store. The model sees an image only when it renders or crops one.
- **Document tools are pure functions of (document, args).** Each call is a `ToolCall` memoized under `sha256(doc, tool, args)`. Repeated calls, whether across shards, requests or thousands of RL rollouts on the same synthetic document, are lookups. That keeps rollouts cheap and transcripts reproducible. `split`, `post` and `report` are the exceptions: they write and are never memoized.
- **Every observation states the remaining budget,** and every `ToolCall` records `cost_ms`. The policy can then learn the trade-off between accuracy and cost, for example `read_native_text` first and `recognize` only on scanned regions.
- **The set stays small and stable.** Adding or changing a tool changes the action space and invalidates the learned policy, so tools change only on purpose.

### How the nine steps map to tools

The steps describe what document understanding involves. They are not an order the worker must follow.

| Step | Tools | Notes |
|---|---|---|
| 1. Inspect and render | `inspect_pdf`, `read_native_text`, `render` | |
| 2. Correct the image | `correct_image` | |
| 3. Analyze layout | `detect_layout` | |
| 4. Select views | `crop`, `tile` | |
| 5. Recognize content | `recognize` | |
| 6. Reconstruct structure | the model's own reasoning; `post(follow)` across shards | Reading order, captions, table cells, cross-page joins |
| 7. Extract fields | the model; `report.fields` | Each field cites `SourceSpan`s |
| 8. Reconcile and validate | `validate` | Plus the merge job across shards |
| 9. Retry or review | re-calling any tool with new args; `post(handback)` | Human review becomes a low-confidence `failed` / `partial` result, since there is no human in the loop |

## Thread

Each request has one **Thread**, an append-only, ordered list of `Message`s written by its workers. It is how parallel workers share what they learn without sharing state. The design follows dsh-superagent's Thread, Subscription and Disposition.

| Message `kind` | Written by | Meaning | Who acts on it | dsh-superagent analog |
|---|---|---|---|---|
| `split` | `split` tool | Fan these pages out | Router: creates child and merge jobs | `handback` → a fresh decision by the main session |
| `handback` | `post` | I can't finish these pages; reassign them | Router: creates a job for them | `handback(ticket, trigger, reason)` |
| `unrelated` | `post` | These pages are irrelevant to the question | Other workers: may skip them | `markUnrelated(ticket, thread, reason)` |
| `follow` | `post` | Related content is elsewhere, e.g. "table continues from p. 37" | The worker that owns those pages | `followForContext(ticket, thread)` |
| `note` | `post` | A finding others should know | Other workers, and the merge job | `note(ticket, text)`, which grants nothing |

**Rules:**
- **Every job of a request is subscribed to its Thread.** In dsh-superagent a Ticket subscribes to Threads. Here, membership is just `thread = request_id`.
- **Delivery is pushed.** After every tool call, the worker's observation includes the messages appended since its last step, excluding its own. This is the same as dsh-superagent delivering Inputs into a session at its next step, and it needs no `read` tool.
- **Disposition is derived.** A page's status is its latest message: `unrelated`, `handed back`, `followed`, or untouched. dsh-superagent stores a `disposition` row per Trigger. We compute it from the Thread instead, so there is one fewer table and one fewer writer.
- **Messages advise, they do not command.** Skipping an `unrelated` page is a policy choice, so training can teach a worker when to trust another worker's judgement.
- **The Thread doubles as the table of contents.** A `note` that names a section and its pages (for example "§4 Indemnification, pp. 88–94") is an outline written by the agents. Together with the native PDF outline from `inspect_pdf`, it gives later workers and the merge job a map of the document without a separate index.

## Large documents: split, map, merge

A 200-page PDF fans out on demand. The router does not decide up front: the root worker looks at the document and asks to split.

```
ingest  ──▶ Document (hash, metadata, PDF in store)
plan    ──▶ root job: whole document, budget sized to page count
root    ──▶ inspect_pdf / read_native_text on a few pages, sees 200 pages
            split(pages_per_job=10)                         → Thread: split
            report(ready, summary="split into 20")
react   ──▶ router: 20 extract jobs (parent = root) + 1 merge job (after = the 20)
map     ──▶ N workers in parallel, each on 10 pages, each choosing its own tools
            post(unrelated, pp. 41–50)                       → others may skip
            post(follow, p. 38, "table continues from p. 37")→ pushed to the p. 38 owner
            post(note, pp. 88–94, "§4 Indemnification")      → outline for the merge job
            report(fields + sources, pages_read)
reduce  ──▶ merge worker: reads the reports and the Thread, resolves conflicts,
            may crop / recognize specific regions to check a disputed value,
            reports the final answer + fields
merge   ──▶ router.merge copies the merge job's report → ExtractionResult
```

Metadata is tracked in four keyed layers. None of them is a shared mutable document, so parallel workers never write to the same row:

| Layer | Key | Written by | Holds |
|---|---|---|---|
| Document | `doc_sha256` | `ingest` | `DocumentMetadata`, the decrypted PDF `ObjectRef` |
| Page / region | `ToolCall.key = sha256(doc, tool, args)` | document tools, insert-or-ignore | Every tool result: renders, crops, layout, recognized text, as `Region`s and output refs |
| Thread | `(request_id, seq)` | workers, through `split` / `post` | Coordination: splits, handbacks, skips, cross-page links, findings |
| Request | `request_id` → `job_id` → `seq` | router (jobs), workers (reports) | Shards, findings with page-grounded sources, coverage |

- **Page facts belong to the document, not the request.** Page 137's layout is the same whichever shard, request or rerun asks for it, so it is keyed by content plus tool arguments and written once.
- **Findings belong to the request.** Each `ExtractedField` carries `SourceSpan(page, bbox, text, method)`, so a finding keeps its page provenance through the merge. `pages_read` records what each job actually examined.
- **The request document is a fold, not a shared object.** Workers append reports and messages. The merge job reads them and writes the one result.

**Why the merge is a worker job, not router code:** combining findings from several shards (summing a table split across pages, reconciling two contradicting values) needs judgement and sometimes another look at a page. As a job, it gets the same tools, the same deadline handling, and its own transcript, which makes it a trainable step.

**Deadline:** `router.react` reserves about 60 s for the merge job and shares out what remains among the children. It rejects a split it cannot finish in time. If children run late, the merge job starts with the reports that have ended, and comparing `pages_read` with the planned shards shows what was missed.

**Chunks below page level:** OCR layout blocks are stored as one JSON object per page. A block is addressed as `(doc_sha256, page, block_index)`, and its bounding box flows into `SourceSpan.bbox`.

## Learned workflow

Two policies are trained from experience, on the same rollouts:

| Policy | Chooses | Trajectory | Reward |
|---|---|---|---|
| Router π_R | The root budget; whether to accept, cap or reject each `split` and `handback`; the budget per child job | Its decisions on one request | Episode reward minus latency and tool cost |
| Worker π_W | The next tool call and its args, including when to `split`, what to `post`, which pages to skip, and when to report | The transcript: model turns, `ToolCall`s, and the Thread messages delivered at each step | Rubric score of the final answer, plus a grounding bonus, minus a cost penalty |

**Training loop** (TRD §8):
1. Collect rollouts on synthetic documents.
2. Score each answer against its rubric.
3. Update both LoRAs with SDPO++. The hint for the self-teacher is the judge's per-criterion feedback, for example "missed the subtotal on p. 12" or "the date format is wrong".
4. Repeat. Over time the policies learn when splitting pays off, which pages are safe to skip, and which views and recognizers work for which page types, such as `tile` and then `recognize(symbol)` on floor plans.

**Denser signal for skips.** A skip only gets the final rubric reward, which is a weak signal. On synthetic data we also know which pages the correct fields cite. An `unrelated` message on a cited page counts as a mistake, and one on an uncited page counts as correct. That gives per-message labels for free.

**Bootstrapping.** SDPO only sharpens behavior the model already samples (TRD §8.3), so the starting policy must already use the tools sensibly:
1. **Prompt prior.** The system prompt includes the tool specs and the nine-step table as guidance, not as a mandate.
2. **Warm start.** Short scripted trajectories run on synthetic documents with known answers are used as SFT data for a few hundred steps. They include split-and-merge on long documents. The static workflow serves as a teacher once, never as runtime code.
3. **Then RL.** The policy is free to drop, reorder or repeat steps whenever that scores better.

**Rollout data** is the transcript the worker already writes. Each step records `(observation, tool, args, ToolCall.result ref, Thread messages delivered, cost_ms)`, and the episode records the rubric reward. No separate logging path is needed.

## Object store

```python
def put(data: bytes, media_type: str) -> ObjectRef      # writes var/objects/ab/abcdef…
def get(ref: ObjectRef) -> bytes
```

The store is content-addressed (sha256), so identical renders and OCR outputs are stored once. Two functions are the whole interface, so moving to S3 later means replacing these two.

## Queue

SQLite in WAL mode, with four tables. Each table has one owner, following the prototype's store, which keeps JSON rows with a single writer per table:

```sql
jobs       (id TEXT PRIMARY KEY, request_id TEXT, body JSON, attempts INT, lease_until TIMESTAMP) -- router writes
reports    (job_id TEXT, seq INT, body JSON, created_at TIMESTAMP, PRIMARY KEY (job_id, seq))  -- workers write
messages   (thread TEXT, seq INT, body JSON, created_at TIMESTAMP, PRIMARY KEY (thread, seq))  -- workers write (split / post)
tool_calls (key TEXT PRIMARY KEY, doc_sha256 TEXT, tool TEXT, body JSON)                       -- tools write, insert-or-ignore
```

- **Claim:** one `UPDATE … SET lease_until = now + 120s, attempts = attempts + 1 WHERE id = (oldest job whose lease expired, with no ready/failed report, and every job in its after list ended) RETURNING body`.
- **Append a message:** inside `BEGIN IMMEDIATE`, use `seq = max(seq) + 1` for the Thread, which gives a total order per request.
- **Retry:** a crashed worker's lease expires and another worker claims the job. After 3 attempts, the job counts as failed.
- **No status columns:** a job is done when its latest report is `ready` or `failed`, and a page's disposition is its latest message.

## Process model

```
gdpsolve eval gdp-pdf --workers 4
 ├─ main process: ingest → plan → (react to split / handback)* → merge   (per task)
 └─ 4 × gdpsolve serve worker                                            (claim → loop → tools)
        shared: var/gdpsolve.db, var/objects/
        remote: Tinker (Inkling sampling)
```

There is no daemon or server. Everything runs on one machine, and only model sampling leaves it.

## Planned modules

v0 puts `process()`, the router, the worker loop and the tools in one file, `src/gdpsolve/run.py`, with an in-process thread pool instead of the SQLite queue ([`process-spec.md`](process-spec.md)). The layout below is the longer-term target.

The document service (`src/gdpsolve/service.py`, `gdpsolve serve`) and its SDK (`src/gdpsolve/client.py`) give Reducto-shaped parse, extract, split, and classify over HTTP; the GDP.pdf eval calls it only through the SDK ([`service-spec.md`](service-spec.md)).

```
src/gdpsolve/
  pipeline.py   process()
  ingest.py     ingest()
  router.py     plan(), react(), merge()
  queue.py      put(), claim(), report(), post(), thread(), open_jobs(), new_messages(), reports()
  store.py      put(), get()
  worker.py     run_job(), `serve worker` loop
  tools.py      run(): dispatch the TOOLBOX, memoize document tools
  eval/run.py   run(), score()
```

## What we took from dsh-superagent

`dsh-superagent-prototype/` (DeepSeek) is a proactive coding agent: a main session routes work, and one worker session runs per ticket. We kept its execution core and coordination model, and dropped its chat intake and authorization layer.

| dsh-superagent | GDPSolve | |
|---|---|---|
| `Engine` is the only writer of Tickets; "only Eval creates one" | The router is the only writer of `jobs` | kept |
| Workers may not start Sessions or schedule work; `handback` queues work for the main session | `split` and `post(handback)` ask the router, and `router.react` decides | kept |
| Ticket plus one Executor session and directory | `Job` plus one worker loop, with its files in the object store | kept |
| Thread: ordered, append-only Triggers | Thread: ordered, append-only `Message`s, one per request | kept |
| Subscription fan-out, Inputs delivered at the next step | Every job of a request subscribed; new messages pushed into the next observation | kept |
| Disposition (a stored record of where each Trigger went) | Page disposition derived from the latest message | kept, derived |
| `markUnrelated`, `followForContext`, `note` | `post(unrelated)`, `post(follow)`, `post(note)` | kept, one tool |
| Workers talk back through `orchestrator_report`, with a closed set of Report states | `report`, with states `progress` / `ready` / `failed` | kept, fewer states |
| Ticket stage derived from stored Reports | Job state is its latest Report | kept |
| Concurrency slots, transient-error retry with backoff | `--workers N`; lease expiry plus `attempts ≤ 3`; Tinker 429/5xx retried inside `model.sample` | kept |
| Worker subagents inside one Assignment | — | not used: child work goes through the queue, so it gets leases, retries and parallel processes |
| Source / Eval / Mandate / Policy / Channel | — | dropped: there is no chat intake or authorization, and the caller receives the result directly |
| Attention, handoff to a person, panel | — | dropped: no human in the loop |
| Background shell jobs, 30-minute slot rules | — | dropped: tools are synchronous and short |

## Training hook

Every `Report` carries its job's `transcript` (model turns, `ToolCall`s and delivered Thread messages, as NDJSON). Those transcripts are the rollouts SDPO++ trains on (TRD §8 and [Learned workflow](#learned-workflow)). Transcripts and Threads from GDP.pdf runs are evaluation-only and are never used for training (TRD §8.5).
