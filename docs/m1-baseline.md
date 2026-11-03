# M1 baseline spec

Milestone M1 (TRD §12): a baseline score on GDP.pdf, graded by the **official judge**. The same answers
are also scored by the untrained `verify()` ([`verifier-trd.md`](verifier-trd.md)), and the agreement is
shown in the eval report (`src/gdpsolve/eval/report_template.html`).

## Rules

- **GDP.pdf is evaluation only.** This run is evaluation. Do not change prompts, models, or `verify()`
  based on these results and then re-run to improve the number. Results and the report contain GDP.pdf
  text, so they live under `var/` (gitignored) and are never committed.
- **The official scorer is reused unchanged.** Same judge model, prompt, parsing, retries, and aggregation.
- **Spend cap.** Tinker ≤ $5 for this milestone. Gemini spend is printed. Every paid command prints an
  estimate first and aborts above `--budget`.

## Pipeline

```
gdpsolve eval gdp-pdf    --model thinkingmachines/Inkling-Small --out var/runs/m1   → results.jsonl
gdpsolve score gdp-pdf   var/runs/m1/results.jsonl                                  → grades.jsonl + summary
gdpsolve eval verifier   --checker rubric --data gdp-pdf --run var/runs/m1          → report.html + one number
```

### 1. `eval gdp-pdf`: baseline answers

- **Input:** `data/gdp_pdf/data.parquet` and `data/gdp_pdf/pdfs/` (from `gdpsolve download gdp-pdf`).
- **PDF to model input** (new core dependency `pypdfium2`):
  - Each page with a text layer becomes text: `"--- page {n} ---\n{text}"`.
  - Each page without text (scanned) is rendered at 100 DPI and sent as an image part.
  - The PDF content is truncated to a token budget (default 120k) by keeping pages in order. `truncated_at_page` is recorded.
- **Model:** `thinkingmachines/Inkling-Small`, single pass, through the cookbook renderer
  (`model_info.get_recommended_renderer_name`, `tml_v0`).
  - The message is the user prompt (GDP.pdf `prompt`) plus the document content.
  - The answer is the completion. Use the Tinker sampling client with a max-token cap (default 4,096).
- **Output:** `results.jsonl`, one `ExtractionResult` per task:
  - `request_id`: the GDP.pdf `task_id`
  - `answer`: the completion
  - `status`: `ok`, or `failed` with `error`
  - `latency_s`
  - `document`: `DocumentMetadata`
- **Resume:** skip tasks already present in `results.jsonl`.

### 2. `score gdp-pdf`: official grades

- **New extra `eval`:** `gdp-pdf @ git+https://github.com/surge-ai/gdp-pdf@7a72a514a6ab19c90babb00adc817e4ae86b9c1b`.
  This brings `inspect-ai` and `google-genai`.
- **Reuse the official scorer as is.** `gdp_pdf.scorer.rubric_scorer(judge_model="google/gemini-3.5-flash")`
  returns `score(state, target)`, which reads only `state.metadata["rubrics"]` and `state.output.completion`.
  Call it per task with a minimal state object carrying the answer and
  `rubrics=[{"id": str(N), "criterion": text}, …]`, built from the non-empty `rubric - N. criterion` columns.
  - If the Inspect decorator prevents direct calling, call `_score_criterion` / `_criterion_passes` from the
    same module and copy only the aggregation lines, with a comment naming the commit. Do not edit the
    prompt or the parse rules.
- **Keys:** `GEMINI_API_KEY` from `~/.zshrc`. Commands run in non-interactive shells, so source it.
- **Output:**
  - `grades.jsonl`: one row per (task, criterion) with `task_id`, `criterion_id = "{task_id}:{N}"`,
    criterion text, `result` (pass / fail / unscored), `rationale`.
  - `summary.json`: `all_pass` and `mean_criteria`, computed with the official aggregation over tasks.
    Also counts of unscored items, failed generations, and truncated documents.

### 3. `eval verifier --data gdp-pdf`: untrained verifier vs. official verdicts, plus the report

- **Scoring:** for every graded (task, criterion), run `verify(VerifyRequest(response=answer, criteria=[…]), CHECKERS["rubric"])`.
  This uses the untrained Qwen3.8-27B; `verify()` comes from `gdpsolve.verifier`.
  The label is the official verdict (pass = 1, fail = 0). Unscored verdicts are excluded.
- **Print one number:** the Brier reward of `verify()` against the official verdicts.
- **Fill the report.** Replace the JSON in `<script id="report-data">` of `src/gdpsolve/eval/report_template.html`
  (shape documented in the file's top comment) and write `var/runs/m1/report.html`.
  - `meta.sample = false`.
  - `models`: a single series, `{"ref": "Qwen/Qwen3.8-27B", "label": "Untrained verifier", "role": "verifier"}`.
  - `decisions`: one per (task, criterion). `type` and `severity` come from the dataset columns
    `rubric - N. criterion_type` and `criterion_severity`. They are report metadata, not `Criterion` fields.
- **Template changes, minimal:**
  - It must render with only one series: no zero-shot gain tile and no ceiling bars when those roles are absent.
  - Add an M1 headline from `summary.json` under `meta.baseline`: `{model, judge, all_pass, mean_criteria, n_tasks}`.
  - Update the top-of-file comment to match.

## Cost estimate

| Step | Est. |
|---|---|
| Inkling-Small answers: 100 tasks, about 25k prompt tokens (median 26.5 pages, capped at 120k) and about 1.5k output tokens | prefill about $1.50–3; sample about $0.25 |
| Untrained verify: about 1,280 decisions × 400 tokens (Qwen prefill $0.70/M) | about $0.40 |
| Gemini 3.5 Flash judge: about 1,280 calls | Price unknown. The command prints token counts and the estimate. |

## Acceptance

- `results.jsonl`: 100 rows, with failures counted and explained.
- `summary.json`: official `all_pass` and `mean_criteria`. These are the M1 numbers.
- The `verify()` Brier reward against the official verdicts is printed.
- `var/runs/m1/report.html` renders: baseline headline, verifier agreement, lowest-scoring decisions.
- Tests are offline (no network): PDF-to-input on a generated PDF, rubric extraction from a fake parquet
  row, scorer adapter with a mock judge (Inspect `mockllm/model`, as in the upstream offline test), and
  report-data shape validation.
- Tinker spend reported, ≤ $5.
