# Document service: API spec

Code: [`src/gdpsolve/service.py`](../src/gdpsolve/service.py) (server), [`src/gdpsolve/client.py`](../src/gdpsolve/client.py) (SDK).
Design source: [`architecture.md`](architecture.md). Baseline: [`m1-baseline.md`](m1-baseline.md).
The field names follow Reducto (https://docs.reducto.ai). This spec is written to about 80% of ASD-STE100.

## 1. Purpose

The service parses, extracts, splits, and classifies PDFs over HTTP. It runs as a separate process.
The GDP.pdf eval calls the service only through the SDK.

The first configuration (`--parser native`, the defaults below) gives the M1 baseline prompt token for token.
Thus a service run is the measured baseline. A test (`test_extract_prompt_matches_m1_baseline`) checks this.

## 2. Run it

```
gdpsolve serve [--host 127.0.0.1] [--port 8787] [--workers 8] [--parser native|paddle]
               [--model thinkingmachines/Inkling-Small] [--checkpoint TINKER_PATH]
               [--effort 0.3] [--max-tokens 8192] [--token-budget 120000]
gdpsolve eval gdp-pdf --system service --url http://127.0.0.1:8787 [--concurrency 8] [--limit N]
```

- `--checkpoint`: sample from this Tinker path (for example a trained LoRA) instead of `--model`.
- `--parser paddle`: starts the PaddleOCR-VL server (`run.ensure_ocr_server`) before the service starts.

## 3. Structure

The structure is the same as the DeepSeek Super-Agent: an always-on router and isolated workers.

- **Router.** The HTTP handler. It puts each POST on one job queue as one `Job` and waits for the result.
- **Workers.** `--workers` threads, each with a `while True: job = jobs.get()` loop. This is the concurrency cap.
- **Isolation.** Jobs share only the object store (`var/objects/<sha256>`, `verifier.store`) and the job
  results (`var/jobs/<job_id>.json`). A later request can use `jobid://<job_id>` to use a parse result again.
- **Model.** One Tinker sampling client, made at first use and shared by the workers. The prompt is made with
  the cookbook renderer, as `eval/baseline.py` makes it. Tests give a fake sampler and a fake recognizer.

## 4. Endpoints

All bodies are JSON, except the `/upload` body. Inputs are `gdpsolve://<sha256>` (a file) or `jobid://<job_id>` (a parse job).
Each POST response, except `/upload`, starts with `response_type`, `job_id`, and `duration` (seconds).

| Method and path | Request | Response |
|---|---|---|
| `POST /upload` | raw file bytes | `{file_id: "gdpsolve://<sha256>", presigned_url: null}` |
| `POST /parse` | `{input, settings: {page_range}}` | `{usage: {num_pages}, result: {type: "full", chunks}}` |
| `POST /extract` | `{input, instructions: {schema, system_prompt}, settings: {page_range}}` | `{usage: {num_pages, prompt_tokens, completion_tokens, cost_usd}, result, truncated_at_page, finish, prompt_sha256}` |
| `POST /split` | `{input, split_description: [{name, description}]}` | `{result: {splits: [{name, pages, conf}]}}` |
| `POST /classify` | `{input, classification_schema: [{category, criteria}], page_range}` | `{result: {category}}` |
| `GET /health` | | `{ok: true, workers, queued, config}` |

`page_range` is `{start, end}` (1-indexed, inclusive), a list of page numbers, or `null` (all pages).

### 4.1 parse

There is one chunk for each page: `{content, blocks: [block]}`. A block is
`{type, bbox: {left, top, width, height, page}, content, image_url}`. The bbox is the full page (0, 0, 1, 1).

| Page | `native` parser | `paddle` parser |
|---|---|---|
| Has a text layer | `Text` block with the page text | same as `native` |
| No text layer | `Figure` block, `content` "", `image_url` = PNG at 100 DPI in the store | `Text` block with the PaddleOCR-VL text (mode `text`) |

### 4.2 extract

1. Get the chunks: parse the file, or read the `jobid://` parse result.
2. Make the user message as `baseline.prepare` makes it: the system prompt as text, then each page in order
   (`--- page {n} ---` and its text, or the header and the page image), until `--token-budget` tokens.
   `pdf_input.parts` does this for both.
3. Sample once (`--effort`, `--max-tokens`) and keep only the text part of the completion (`baseline.answer_text`).

| `instructions.schema` | Prompt | `result` |
|---|---|---|
| `{}` (free form) | the system prompt | `[{"answer": text}]` |
| not empty | the system prompt and the line `Return only JSON matching this schema: <json>` | `[parsed JSON]`; if the JSON does not parse, `[{"answer": text}]` and `confidence: "low"` |

`finish` is `stop` or `length` (the completion hit `--max-tokens`). `prompt_sha256` is the decoded prompt in the store.

### 4.3 split and classify

- `split`: one sample for each page. The model gets the section list and returns a JSON list of section names.
  A section without pages is not in `splits`. `conf` is always `high`.
- `classify`: one sample over the pages in `page_range` (default: pages 1 to 5). `category` is the answer text.

### 4.4 Errors

| Status | When | Body |
|---|---|---|
| 404 | Unknown path | `{detail}` |
| 422 | Bad input: no `input`, unknown file or job, not a PDF, body is not JSON | `{detail}` |
| 500 | The job failed (for example, Tinker failed) | `{detail}` |

## 5. SDK

```python
from gdpsolve.client import GDPSolve

client = GDPSolve("http://127.0.0.1:8787", timeout=600)
upload = client.upload(Path("a.pdf"))                  # Path or bytes
client.parse.run(input=upload["file_id"])
client.extract.run(input=upload["file_id"], instructions={"schema": {}, "system_prompt": "..."})
client.split.run(input=..., split_description=[...])
client.classify.run(input=..., classification_schema=[...])
client.health()
```

Each call returns the response JSON as a dict. A status that is not 200 raises `GDPSolveError(status, detail)`.

## 6. Eval hook

`eval gdp-pdf --system service` does these steps for each task that has no `ok` row, `--concurrency` at a time:
upload the PDF, then `extract` with `{"schema": {}, "system_prompt": task["prompt"]}`.

- `results.jsonl` rows have the baseline format: `ExtractionResult` and `model`, `effort`, `truncated_at_page`,
  `prompt_tokens`, `completion_tokens`, `finish`, `prompt_sha256`. `latency_s` is the client wall time for the document.
  `gdpsolve score gdp-pdf` reads these rows.
- `run.json`: `system`, `n`, `wall_s`, `throughput_docs_per_min`, `latency_p50`, `latency_p95`, `cost_usd`, `ok`,
  `url`, and the server `config` (from `/health`).
- The default directory is `var/runs/service`.
