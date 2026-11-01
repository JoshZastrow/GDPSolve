# GDPSolve

A local document-extraction system (router → persistent SQLite queue → parallel tool-using workers) whose router and extraction model are co-trained with async RL to improve accuracy on the [GDP.pdf](https://github.com/surge-ai/gdp-pdf) benchmark.

Design: [`docs/TRD.md`](docs/TRD.md).

## Setup

```bash
uv sync --extra train            # core + Tinker training stack
uv run gdpsolve eval download    # GDP.pdf -> data/gdp_pdf (eval only, gitignored)
uv run pytest
```

| extra   | contents                                  |
|---------|-------------------------------------------|
| (core)  | pydantic, typer, huggingface-hub          |
| `train` | tinker, tinker-cookbook, verifiers        |

Dependencies are added when the code that needs them lands (TRD §8.4 and R7 note known conflicts).

## CLI

Every script, run, and pipeline is a `gdpsolve` subcommand, grouped by system stage (TRD §11).

| command                  | what it does                              |
|--------------------------|-------------------------------------------|
| `gdpsolve eval download` | fetch the pinned GDP.pdf benchmark        |

## Layout

```
src/gdpsolve/
  cli.py          gdpsolve CLI (all entry points)
  schemas.py      request/result contract
  eval/gdp_pdf.py GDP.pdf download
docs/TRD.md       technical requirements
```

> GDP.pdf is an evaluation-only benchmark. Do not train on it or on anything derived from it.
