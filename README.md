# GDPSolve

A local document-extraction system (router → persistent SQLite queue → parallel tool-using workers) whose router and extraction model are co-trained with async RL to improve accuracy on the [GDP.pdf](https://github.com/surge-ai/gdp-pdf) benchmark.

Design: [`docs/TRD.md`](docs/TRD.md).

## Setup

```bash
uv sync --extra ocr --extra train --extra eval   # main dev environment
uv run python scripts/download_gdp_pdf.py        # GDP.pdf -> data/gdp_pdf (eval only, gitignored)
uv run pytest
```

Extras:

| extra   | contents                                                        |
|---------|-----------------------------------------------------------------|
| (core)  | pydantic, typer, pikepdf, pymupdf, HF hub/datasets, tinker      |
| `ocr`   | paddleocr, paddlepaddle (PaddleOCR-VL-1.6)                      |
| `train` | torch, transformers, tinker-cookbook, verifiers, prime          |
| `eval`  | deepeval, inspect-ai                                            |
| `env`   | openenv, which conflicts with `train`, so it runs in its own process |

`prime-rl` is not a dependency; clone it separately on the GPU training host (TRD §8).

## Layout

```
src/gdpsolve/
  schemas.py   request/result contract
  sdk/         client SDK
  router/      planner policy
  jobqueue/    SQLite queue
  worker/      tool-using extraction agent
  tools/       PDF / OCR / table / symbol tools
  models/      Tinker + PaddleOCR-VL clients
  env/         OpenEnv environment (synthetic docs)
  training/    SDPO++ async RL
  eval/        GDP.pdf scoring + DeepEval
  data/        loaders + synthetic generation
configs/       runtime config
scripts/       one-off utilities
docs/          TRD and design notes
```

> GDP.pdf is an evaluation-only benchmark. Do not train on it or on anything derived from it.
