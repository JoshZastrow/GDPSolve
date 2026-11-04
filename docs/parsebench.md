# ParseBench

[ParseBench](https://github.com/run-llama/ParseBench) ([paper](https://arxiv.org/abs/2604.08538),
[data](https://huggingface.co/datasets/llamaindex/ParseBench)) scores document *parsing*: PDF page → Markdown
(tables as HTML, charts as tables) plus layout boxes. About 2,078 human-verified pages from insurance, finance,
and government documents, in five dimensions. Scoring is deterministic and rule-based: no LLM judge.

GDPSolve adds it as a second eval next to GDP.pdf. GDP.pdf asks "is the answer right?"; ParseBench asks "did the
parser keep the page's content and structure?", which is what the extraction workers depend on.
It is **evaluation only** by project policy, although the dataset is Apache-2.0.

## Commands

```bash
uv sync --all-extras                                   # the `parsebench` extra pins parse-bench to the leaderboard's commit
uv run gdpsolve download parsebench --test             # 3 documents per category -> data/parsebench-test (6 MB)
uv run gdpsolve download parsebench                    # full set at the leaderboard's revision -> data/parsebench (590 MB)
uv run gdpsolve eval parsebench <results> [--test] [--dimension tables ...]   # prints Overall (0-100)
uv run gdpsolve score parsebench <results>/<group>/<page>.result.json         # verifier on one page
```

`<results>` holds one ParseBench `InferenceResult` JSON per page, as `parse-bench` writes them:
`<results>/{table,chart,text,layout}/<page>.result.json` with `product_type: "parse"`.
`gdpsolve.verifier.parsebench.parse_result(case, markdown, layout_pages)` builds one from a parser's page output.
A page with no result file scores 0, as on the leaderboard.

## Dimensions and score methods

Each dimension's score is the mean, over its pages, of one headline metric (ParseBench
`scripts/leaderboard/common.py`). **Overall** is the unweighted mean of the five.

| Dimension | Group | Pages | Headline metric | How a page is scored |
|---|---|--:|---|---|
| Tables | `table` | 503 | `grits_trm_composite` (GTRM) | 0.5 × GriTS-Con + 0.5 × TableRecordMatch. GriTS-Con: F-score of the best 2D alignment of predicted and true cell grids, cell text compared by longest common subsequence. TableRecordMatch: each row becomes a header → value record; records are matched by the Hungarian algorithm with graduated cell scores. GriTS-Con alone when TRM does not apply. |
| Charts | `chart` | 568 | `rule_pass_rate` | Share of the page's chart rules that pass: each data point (series, axis label, value) must appear in the output's table; numbers match within 1% relative tolerance. |
| Content Faithfulness | `text_content` | 506 | `content_faithfulness` | (1 × text correctness + 0.5 × reading order) / 1.5, from text rules covering omissions, hallucinations, and order violations. |
| Semantic Formatting | `text_formatting` | 476 | `semantic_formatting` | Weighted mean of styling (bold, strikethrough, super/subscript) and title hierarchy at weight 1, LaTeX and code blocks at 1/5. |
| Visual Grounding | `layout` | 500 | `layout_element_rule_pass_rate` | Share of annotated layout elements that pass: a predicted box localizes it (IoU), has the right class, and carries the right content (attribution). Needs layout boxes in the output; a parse without boxes scores 0. |

Pages that fail in a way charged to the parser (no output, no layout, unparseable output) count as 0.
Harness errors and pages ParseBench skips are left out of the mean.

## Verifier

`gdpsolve.verifier.parsebench.score(result, case) -> float | None` is the leaderboard's per-page score, in [0, 1].
It calls ParseBench's own `EvaluationRunner` with the routing the runner uses (normal; cross-evaluation of a parse
output against layout ground truth; multi-task when a page mixes layout and text rules), so it can serve as an RL
reward for a parsing policy and its mean equals the eval score. `None` means unscored, matching the leaderboard:
the page is left out, not counted as 0.

`tests/test_parsebench.py` checks, per dimension on the test subset, that each page's verifier score equals the
harness's per-page metric and that their mean equals the harness's dimension score.

## How the harness works

`gdpsolve eval parsebench` runs, for each dimension, the command ParseBench's leaderboard uses to re-score
published outputs (`scripts/leaderboard/rescore.py`):

```
python -m parse_bench.cli evaluation run <results> -t <data> -g <group> -r <report>/<group> --force True
```

and reads `avg_<headline metric>` from `<report>/<group>/_evaluation_report.json`. The runner loads the group's
test cases, matches result files to them by test id, scores each page in a process pool, and scores pages without
a result as blank output. One addition: when no page produced the metric because every page failed against the
parser, the score is 0 rather than missing. Reports land in `<results>/_parsebench/` (per-page results in each
group's `_evaluation_report.json`, totals in `summary.json`).

Pins: `parse-bench` at commit `c38085d` (the leaderboard's `scored_with`; 21 commits after v1.0.7, including layout
scoring changes that move Visual Grounding by 10 points); dataset `llamaindex/ParseBench@2805a1d` (the leaderboard's
`data_revision`); test subset `test-data@68bbab2`. Move the pins together when the leaderboard is re-scored.

Re-scoring a provider's published outputs needs that provider's SDK when its layout adapter uses it, e.g.
`uv run --with llama-cloud gdpsolve eval parsebench ...` for LlamaParse rows. GDPSolve's own outputs
(`pipeline_name: "gdpsolve"`) use the default adapter and need nothing extra.

## Parity with the leaderboard

Checked 2026-11-02 by re-scoring a verified leaderboard row from its published outputs
(`llamaindex/ParseBench-results@0b3c833`, `results/llamaparse_cost_effective`, unpacked with that repo's
`tools/explode.py` into `var/parsebench/output/`):

| | Tables | Charts | Content Faith. | Sem. Format. | Visual Ground. | Overall |
|---|--:|--:|--:|--:|--:|--:|
| `gdpsolve eval parsebench` | 86.72 | 73.98 | 87.32 | 78.14 | 62.22 | 77.68 |
| leaderboard.csv | 86.72 | 73.98 | 87.32 | 78.14 | 62.22 | 77.68 |

2,553 page evaluations, about 3.5 minutes on a 10-core Mac. With `parse-bench==1.0.7` instead of the pinned commit,
Visual Grounding came out 52.10 (Overall 75.65); the other four were identical.
