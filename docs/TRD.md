# GDPSolve: Technical Requirements Document

| | |
|---|---|
| Status | Draft v0.1 |
| Date | 2026-11-01 |
| Owner | Josh Zastrow |
| Target | GDP.pdf strict pass rate > 10% within the 5-minute-per-document SLA |

## 1. Summary

GDPSolve is a document-extraction system that runs locally. A **router** plans the work for each document, a **persistent SQLite queue** holds the jobs, and **parallel workers** do the extraction. Each worker is a tool-using LLM agent (Thinking Machines **Inkling**, served through **Tinker**) with local document tools, including **PaddleOCR-VL-1.6** for OCR and layout. The router and the worker model are **co-trained** with asynchronous RL (SDPO++) on *synthetic* document derivatives inside an OpenEnv sandbox. Progress is measured against **GDP.pdf**.

## 2. Benchmark: GDP.pdf

Source harness: <https://github.com/surge-ai/gdp-pdf> (Inspect AI). Local copy: `data/gdp_pdf/`, fetched with `gdpsolve eval download`.

**Profile** (measured on the downloaded data):

| Property | Value |
|---|---|
| Tasks | 100, single `test` split |
| Domains | STEM 15, Healthcare 11, Real Estate / Engineering / Mfg / Construction 10 each, Insurance / HR / Finance 9 each, Legal 7 |
| Rubric criteria per task | mean 12.8, min 3, max 30 (1,275 total: 1,050 "Primary Intent", 225 "Dodged Bullet") |
| Pages per PDF | median 26.5, mean 45.9, max 200; 13 PDFs over 100 pages |
| PDFs without a text layer (scanned) | 9 of 100 (based on the first 3 pages) |
| Encrypted PDFs | 0 |
| Prompt length | median 387 characters |

**Grading.** An LLM judge (the leaderboard uses Gemini 3.5 Flash) scores each rubric criterion 0 or 1. It sees *only the response text and that one criterion*. It never sees the prompt or the PDF. The harness reports two metrics:
- `all_pass`, our **strict pass rate**: the share of tasks where every criterion passes. This is the headline number.
- `mean_criteria`, our **mean rubric score**: the average fraction of criteria satisfied per task.

**Implications for the design:**
1. GDP.pdf grades *answers to questions about documents*. It does not grade structured field dumps. The system must return a free-text `answer` that satisfies the rubric, alongside the structured `fields`. Rubrics reward precision and penalize "dodged bullets", meaning plausible but wrong claims.
2. The judge sees only the response, so the answer has to state values, units, and caveats explicitly. Citations do not help the judge directly, but they make answers more accurate (grounded-then-answer).
3. **Evaluation only.** The dataset card forbids training on GDP.pdf or anything derived from it. Synthetic training documents must therefore be generated *independently* of the GDP.pdf test PDFs, not derived from them. A contamination check is required (§8.5).
4. The official `surgeai/GDP.pdf` HF repo currently returns 404. We use the mirror `Oitaaa/GDP.pdf`, pinned at revision `7f76578`. Its card and schema match the harness. Before reporting any result we must re-verify it against the official source if that becomes available again.

## 3. Requirements

### 3.1 Functional

| ID | Requirement | Design response |
|---|---|---|
| FR-1 | The client SDK accepts a file **or a directory** plus extraction instructions | `GDPSolveClient.extract(path, instructions, output_schema=None, passwords=[])`. A directory expands to one job per PDF. |
| FR-2 | Returns the extraction payload | `ExtractionResult` (`src/gdpsolve/schemas.py`) containing `answer` and `fields` |
| FR-3 | Runs in batch | A batch is a set of queue jobs sharing a `batch_id`. Results are streamed back as they finish, with `wait()` / `as_completed()`. |
| FR-4 | Reports confidence per extraction | `ExtractedField.confidence` plus a document-level `confidence` (§6.4) |
| FR-5 | Handles password-protected documents | Ingest decrypts with the passwords the caller supplies. Without them the status is `password_required`. Passwords are `SecretStr`, never logged or persisted. |
| FR-6 | Returns source grounding | `SourceSpan`: page, bounding box, verbatim text, and method (`text_layer` / `ocr` / `vlm`) for each field |
| FR-7 | Returns document metadata | `DocumentMetadata`: sha256, page count, encryption, text-layer presence, and PDF info dictionary fields |

### 3.2 Non-functional

| ID | Requirement | Design response |
|---|---|---|
| NFR-1 | Document → extraction in **< 5 min** (p95) | Per-job deadline of 300 s, enforced by the router's budget (§7) |
| NFR-2 | **> 10% strict pass rate** on GDP.pdf | Primary success metric. Track `mean_criteria` as the denser signal. |
| NFR-3 | Handles **misaligned tables** and **floor-plan symbols** | Dedicated tools (§6.3): table reconstruction from OCR layout plus VLM crop verification, and a symbol/legend tool for drawings |
| NFR-4 | Runs locally | Router, queue, tools, and OCR run on the local machine. The worker LLM is served remotely by Tinker (see §10, R1). |
| NFR-5 | Reproducible evaluation | Pinned dataset revision, pinned judge model, `uv.lock`, run manifests stored in SQLite |

## 4. Architecture

See [architecture.md](architecture.md) and its [diagram](diagrams/architecture.html). In short:
- Every caller (eval, CLI, SDK) calls one function, `process(ExtractionRequest) -> ExtractionResult`.
- `process()` ingests the PDF, asks the router for jobs, puts them on a SQLite queue, waits for reports, and merges them.
- Worker processes claim jobs and run the Inkling agent loop on Tinker. Their tools store page images and OCR in a content-addressed object store, and they publish results through `Report`s.

## 5. Data contract

`src/gdpsolve/schemas.py` is the source of truth:
- **Public:** `ExtractionRequest` → `ExtractionResult`, with `ExtractedField`, `SourceSpan` and `DocumentMetadata`.
- **Internal:** `Document`, `Job`, `Report` and `ObjectRef`.

`status` ∈ `ok | partial | failed | password_required`. A `partial` status means the deadline was hit and the best answer so far was returned.

## 6. Components

### 6.1 Persistent queue (`gdpsolve.jobqueue`)
- SQLite in WAL mode. Tables: `batches`, `jobs` (`id`, `batch_id`, `doc_sha256`, `plan_json`, `state`, `lease_until`, `attempts`, `deadline_at`), `results`, `traces` (tool calls, token logprobs, and the `policy_version` that produced them, for async RL staleness control), `runs`.
- Claim a job with `UPDATE … SET state='leased', lease_until=now+lease_s WHERE id=(SELECT … LIMIT 1) RETURNING *` inside `BEGIN IMMEDIATE`.
- At-least-once delivery. Job id = hash(doc_sha256, instructions, plan_version), so retries are idempotent. Failures back off; jobs that fail too often move to a dead-letter state.
- Traces live in the same DB, so every production or eval run can become training data. That is only allowed for synthetic documents; GDP.pdf traces are tagged `eval_only` and excluded.

### 6.2 Router (`gdpsolve.router`)
- Input: document metadata, a page-level triage signal (text density, image ratio, table/drawing likelihood), and the instructions.
- Output, a **plan**:
  - which pages to read
  - whether to OCR or use the text layer per page
  - the number of parallel sub-jobs (page shards versus a single pass)
  - the tool-call budget
  - the model tier
- v0 is a heuristic planner. v1 is a small LLM policy π_R trained jointly with the worker (§8). The reward for π_R includes a latency/cost penalty, which keeps plans inside the SLA.

### 6.3 Worker and tools (`gdpsolve.worker`, `gdpsolve.tools`)
The worker runs an agent loop on Inkling and stops when it emits `final_answer` or hits its budget. Tools:

| Tool | Purpose |
|---|---|
| `pdf.text(page)` | Text layer with word-level bounding boxes |
| `pdf.render(page, dpi, crop?)` | Page or crop image for VLM inspection |
| `pdf.search(query)` | Lexical search across pages, returning hits with page and bounding box |
| `ocr.layout(page)` | PaddleOCR-VL-1.6: blocks, reading order, tables (HTML), formulas, charts |
| `table.reconstruct(page, region)` | Rebuilds misaligned or multi-page tables. Merges OCR cells with text-layer words by bounding-box geometry, repairs column drift, and stitches continuations across pages. |
| `symbol.legend(page)` / `symbol.find(page, symbol)` | Floor plans and drawings: parses the legend, matches symbols (VLM crop comparison and template match), counts, and reads dimension strings |
| `calc(expr)` | Deterministic arithmetic for reconciliations and unit conversion |
| `cite(field, page, bbox, text)` | Records grounding. A cited span must exist in the text layer or OCR output for that page, or the citation is rejected. |

### 6.4 Confidence
Field confidence is a calibrated combination of three signals:
- **evidence agreement**: text layer, OCR, and VLM agree on the value
- **self-consistency**: agreement across *k* sampled answers when budget allows
- **token log-probability** of the value span (available from Tinker sampling)

Calibration uses isotonic regression on synthetic held-out data, judged by ECE. Document confidence is the minimum over required fields, or the judge-free self-consistency score of the answer.

### 6.5 Models (`gdpsolve.models`)
- **Worker LLM:** `thinkingmachines/Inkling-Small` (276B total, 12B active, multimodal MoE) through the Tinker sampling client. `Inkling` (975B total, 41B active) is the upper-tier option the router can escalate to. Fine-tuning uses Tinker LoRA.
- **OCR / layout:** `PaddlePaddle/PaddleOCR-VL-1.6`, run locally (CPU or MPS on a Mac is slow; see §7). The serving path (transformers, vLLM, or the GGUF variant) is decided at M2.
- **Router policy:** a small open model trained as a LoRA on Tinker. The base model is open (§10, Q3).

## 7. Latency budget (p95, 46-page document)

| Stage | Budget |
|---|---|
| Ingest + triage | 5 s |
| Router plan | 5 s |
| OCR (only pages selected by the router) | ≤ 90 s, parallel |
| Worker agent loop (≤ 40 tool calls, sharded) | ≤ 150 s |
| Aggregate + confidence | 10 s |
| Slack | 40 s |

Key levers:
- The router selects pages, so 200-page documents are never OCR'd in full.
- OCR results are cached by `(sha256, page, dpi)`.
- PDFs with a text layer skip OCR on text-only pages.
- At 30 seconds before the deadline, the worker is forced to `final_answer`, so a result is always returned.

## 8. Training

### 8.1 Data: synthetic document derivatives
- Generate documents across the same ten professional domains *without* using GDP.pdf content. Sources:
  - permissively licensed public documents (SEC filings, public building plans, clinical guidelines, government forms)
  - procedurally generated PDFs with known ground truth: misaligned and multi-page tables, scanned and noisy renders, rotated pages, floor plans with symbol legends, and encrypted variants
- Each item contains a document, a task prompt, a reference answer, and a rubric written in GDP.pdf style ("Primary Intent" plus "Dodged Bullet" criteria) by a teacher model, with spot checks.
- The environment is sandboxed: an OpenEnv server exposes `reset(task)` / `step(tool_call)` / `reward()` over these documents. Tools in the environment are the same code as the production tools.

### 8.2 Co-training router and worker
- **Episode.** The router samples a plan. Workers roll out under that plan. The aggregator produces an answer, and a judge scores it against the rubric.
- **Worker reward:** rubric score (fraction of criteria passed), plus a bonus for all criteria passing, plus a grounding validity bonus, minus a penalty for invalid citations.
- **Router reward:** the same task reward minus λ·latency and μ·tool-cost, with a hard penalty when the plan violates the SLA.
- **Schedule:** alternating updates (freeze one, update the other) to start, moving to joint updates once both are stable.

### 8.3 Algorithm: SDPO++ (async)
Reference: [Scaling SDPO](https://www.trajectory.ai/field-notes/scaling-sdpo) (Trajectory AI field notes).

**SDPO (Self-Distillation Policy Optimization).**
- The **student** is the policy π_θ(·|s).
- The **teacher** is *the same weights* conditioned on a hint *c* that was not available at rollout time: π_θ(·|s, c).
- No separate teacher, reward model, or labels are needed.

The loss is a per-token reverse (mode-seeking) KL from student to stop-gradient teacher:

```
L_SDPO(θ) = E_{τ∼π_θ} [ Σ_t KL( π_θ(·|s_t) ‖ stopgrad(π_θ(·|s_t, c)) ) ]
```

In practice it is estimated on sampled tokens using the teacher's log-probability. This gives dense credit at every token, so it is well-defined with **group size 1**. GRPO is not: with one rollout per task, its within-group ranking is undefined. That matters here because GDPSolve rollouts are long (≤ 40 tool calls) and expensive.

**Off-policy problem.** Asynchronous rollouts are stale by *K* trainer steps. The naive importance ratio r_t = π_θ(a_t|s_t) / π_{θ−K}(a_t|s_t) is heavy-tailed. Because SDPO weights each token separately, one rare token can dominate an update, and the reference post saw training collapse within 50 steps at K = 3.

**SDPO++ = SDPO + stabilizers.** These are the reference configuration and our starting hyperparameters:

| Setting | Value |
|---|---|
| PPO ratio clipping | ε = 0.2 |
| Per-token advantage clipping | 3× running mean. Preferred over a KL-to-reference penalty, which would cost a second forward pass and anchor the policy. |
| Max staleness | K = 3. Accuracy held through K = 3 and degraded beyond it; about 2× wall-clock speedup. |
| Group size | G = 1, keeping groups where every rollout failed |

**Hints for GDPSolve (*c*).** These are ordered by expected strength. Each needs an ablation.
1. The judge's per-criterion verdicts and rationales for the rollout's own answer: which criteria failed and why.
2. The reference answer for the synthetic task.
3. A successful sibling rollout, when one exists.

Hints must not let the teacher simply copy the answer. The post flags hint-copying as the hardest open issue. Mitigation: give criteria-level feedback rather than verbatim answers, and enforce grounding rules in the reward.

**Known limit.** SDPO sharpens behaviors the model already samples. It does not reliably teach behaviors the model never samples. Tasks where the base pass rate is near zero (likely for floor-plan symbols) may need a behavior-cloning or DAgger warm start first (§12, M3/M4).

**Pipeline mapping:**
- Workers stream trajectories into the SQLite `traces` table, tagged with `policy_version` (§6.1).
- The trainer pulls batches that satisfy staleness ≤ K.
- For each sampled token, it computes teacher log-probabilities under the hint, the clipped importance ratio, and the clipped advantage, then updates the LoRA.
- Router and worker keep separate LoRA adapters (π_R, π_W) trained with the same loss. The router's trajectory is the plan, and its hint is the per-stage latency and outcome of the episode.

### 8.4 Infrastructure
**Tinker is the single training and sampling backend.**
- **Tinker:** LoRA training and sampling for both policies: Inkling for the worker and a small open model for the router.
- **tinker-cookbook:** its `rl` module supplies the environment interface, rollout and training loop primitives, and renderers. SDPO++ (§8.3) is built on these.
- **verifiers (Prime Intellect), used as needed:** for rubric and reward definitions and text-only environments.
  - tinker-cookbook's `recipes/verifiers_rl` runs verifiers environments on Tinker, but only text-based ones.
  - The worker sees page images, so its main environment is implemented directly against `tinker_cookbook.rl`, reusing verifiers rubrics where they fit.
  - The `prime` CLI is only for installing Environments Hub packages (`uv tool install prime`). It is not a project dependency.
- **prime-rl:** not used.
- **OpenEnv:** optional sandbox server. verifiers v1 already ships an `openenv` taskset (`verifiers[openenv]`), so OpenEnv can sit behind the verifiers interface rather than being integrated separately.

**Multimodal environment options**, surveyed 2026-11-01 against installed tinker-cookbook 0.5.9 and verifiers 0.4.0:

| Option | Images in observations? | Fit for GDPSolve |
|---|---|---|
| `tinker_cookbook.rl.Env` (custom) | Yes. Observations are `tinker.ModelInput`, and the Inkling renderer (`tml_v0`) handles image parts. Model info marks both Inkling models `is_vl=True`. | **Primary.** Full control over multi-turn tools, page images, and the SDPO teacher pass. |
| verifiers v1 environment | Yes, at the message level. v1 types include `image_url` content parts, and its training client checks the renderer before passing images through. | **Secondary.** Use it for rubrics, reward functions, and taskset packaging, so environments could later be published to Prime's Environments Hub. |
| `recipes/verifiers_rl` (verifiers on Tinker) | No. The recipe states it supports text-based environments only. | Not usable for the worker until it supports images. |
| Harbor (`verifiers[harbor]`, `recipes/harbor_rl`) | Not natively. Harbor tasks are containers with tests: the agent gets a bash tool, and the reward comes from test results. | Possible as a sandbox format, where the PDF lives in the container and the agent renders pages through tools. Overkill until we need containerized tools. |
| `recipes/rubric` | Text only, as written. | Reference for the LLM-judge rubric reward (sum of per-criterion grades), which matches GDP.pdf grading. |
| `recipes/sdft` | Text only, as written. | Reference for the Tinker self-distillation mechanics: teacher logprobs under context, per-token KL. SDPO++ extends it with feedback hints and the off-policy stabilizers. |
| `recipes/vlm_classifier` | Yes, but supervised and Qwen-only. | Reference for passing images through Tinker renderers. |

### 8.5 Contamination guard
- The canary GUID from the GDP.pdf card is checked against all training corpora.
- MinHash near-duplicate checks compare synthetic documents and prompts with GDP.pdf PDFs and prompts. Anything above the threshold is dropped.
- Traces from GDP.pdf runs are tagged `eval_only` and filtered out of every training loader by construction.

## 9. Evaluation

| Metric | Definition | Tool |
|---|---|---|
| Strict pass rate | `all_pass`: share of tasks with every criterion passed | Official Inspect harness scorer (parity) |
| Mean rubric score | `mean_criteria` | Official scorer |
| Pass@5 / Pass^5 | 5 epochs, leaderboard configuration | Inspect reducers |
| Grounding precision | Share of cited spans that exist in the document and support the value | DeepEval custom metric |
| Calibration | ECE of field confidence on synthetic held-out data | DeepEval / custom |
| Latency | p50 / p95 per document; SLA violation rate | Queue timestamps |

- **Parity.** Our `gdpsolve.eval` adapter runs GDPSolve as an Inspect solver. The response passed to the judge is `ExtractionResult.answer`. The official `rubric_scorer` is used unchanged, with judge `google/gemini-3.5-flash`.
- **Day-to-day iteration** happens on synthetic held-out sets. GDP.pdf runs are reserved for milestone checks to limit overfitting through repeated test-set feedback.

## 10. Risks and open questions

| # | Item | Notes / proposed resolution |
|---|---|---|
| R1 | "Runs locally" vs. model size | Inkling-Small (276B) cannot run on a laptop. Proposal: everything except LLM sampling runs locally, and the worker calls Tinker. A fully offline path would need a smaller local VLM, at an accuracy cost. |
| R2 | SDPO++ hint design | Algorithm and hyperparameters come from the Scaling SDPO post (§8.3). The open risk is hint design: copying, and behaviors the model never samples. |
| R3 | Benchmark is QA, not pure extraction | The system must produce graded free-text answers (§2). Structured `fields` support grounding and confidence. |
| R4 | No encrypted PDFs in GDP.pdf | FR-5 is validated on synthetic encrypted variants only. |
| R5 | Official dataset unavailable | We use a pinned mirror. Re-verify file hashes against the official source before reporting results. |
| R6 | Local OCR throughput | PaddleOCR-VL on CPU/MPS may exceed the OCR budget on large scans. Mitigations: router page selection, caching, GGUF/llama.cpp, or a remote GPU OCR server. |
| R7 | Dependency friction | Known conflicts to handle when these are added back: deepeval 4.x caps `click<8.4` while huggingface-hub needs ≥8.4.2 (fix with a uv override); openenv needs `mcp<2`, which conflicts with verifiers. |
| Q1 | Training backend | Resolved: Tinker for both policies, verifiers as needed. prime-rl is dropped. |
| Q2 | SDPO++ hint source | Resolved: algorithm per §8.3. Still open: which hint (judge rationales, reference answer, or sibling rollout) works best. Needs an ablation on synthetic data. |
| Q3 | Router model | Size and base model, chosen from the models Tinker supports. |
| Q4 | Judge for training rewards | Using the leaderboard judge (Gemini 3.5 Flash) for training rewards risks overfitting to the judge. Proposal: train against an open judge and evaluate with the official one. |

## 11. CLI

All scripts, runs, and pipelines are subcommands of `gdpsolve` (`src/gdpsolve/cli.py`), grouped by stage. A group is added when its first command is implemented. Today only `eval download` exists; the table is the planned layout.

| Group | Stage | Commands |
|---|---|---|
| `data` | Training data | `synth`, `check-contamination` |
| `env` | RL environment | `serve` |
| `train` | SDPO++ RL | `worker`, `router` |
| `eval` | GDP.pdf | `download`, `run` |
| `serve` | Inference | `extract`, `worker` |

## 12. Milestones

| Milestone | Scope | Exit criterion |
|---|---|---|
| M0 | Repo, uv environment, dataset, TRD *(this commit)* | `uv run pytest` passes; 100 PDFs present |
| M1 | Baseline: Inkling-Small, single pass, PDF attached, through the Inspect harness | Baseline strict pass rate and mean rubric score recorded |
| M2 | Pipeline: SDK, queue, heuristic router, tool-using workers, grounding, metadata, passwords | All FRs pass integration tests; p95 < 5 min on GDP.pdf |
| M3 | Synthetic environment: OpenEnv server, generators for tables, floor plans, and encrypted docs, rubric authoring, contamination guard | 2k+ tasks; contamination check is clean |
| M4 | RL: worker SDPO++ on Tinker | Improvement on the synthetic held-out set; GDP.pdf milestone run |
| M5 | Co-training the router | Strict pass rate > 10% at p95 < 5 min |
