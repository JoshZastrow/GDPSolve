# GDPSolve

A local document-extraction system (router → persistent SQLite queue → parallel tool-using workers) whose router and extraction model are co-trained with async RL to improve accuracy on the [GDP.pdf](https://github.com/surge-ai/gdp-pdf) benchmark.

Design: [`docs/TRD.md`](docs/TRD.md) · [`docs/architecture.md`](docs/architecture.md).

## Setup

```bash
uv sync --extra train            # core + Tinker training stack
uv run gdpsolve download gdp-pdf # GDP.pdf -> data/gdp_pdf (eval only, gitignored)
uv run pytest
```

| extra   | contents                                  |
|---------|-------------------------------------------|
| (core)  | pydantic, typer, huggingface-hub          |
| `train` | tinker, tinker-cookbook                   |
| `parsebench` | parse-bench @ c38085d (ParseBench's own scorer, as the leaderboard runs it) |

Dependencies are added when the code that needs them lands (TRD §8.4).

## CLI

Every script, run, and pipeline is a `gdpsolve <action> <resource>` subcommand (TRD §11).

| command                  | what it does                              |
|--------------------------|-------------------------------------------|
| `gdpsolve download gdp-pdf` | fetch the pinned GDP.pdf benchmark     |
| `gdpsolve download parsebench [--test]` | fetch ParseBench at the leaderboard's revision |
| `gdpsolve eval parsebench <results>` | ParseBench leaderboard scores via its own harness ([`docs/parsebench.md`](docs/parsebench.md)) |
| `gdpsolve score parsebench <result.json>` | ParseBench page verifier on one output |

## Layout

```
src/gdpsolve/
  cli.py          gdpsolve CLI (all entry points)
  schemas.py      request/result contract
  eval/gdp_pdf.py GDP.pdf download
  eval/parsebench.py   ParseBench download + harness
  verifier/parsebench.py  ParseBench page verifier
docs/TRD.md       technical requirements
docs/architecture.md  pipeline interfaces (+ diagrams/architecture.html)
```

> GDP.pdf and ParseBench are evaluation-only benchmarks. Do not train on them or on anything derived from them.
