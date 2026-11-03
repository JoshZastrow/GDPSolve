# GDPSolve — instructions for Claude

A local document-processing service (Reducto-shaped: upload, parse, extract, split, classify) with a client SDK,
improved by data, training, and model choice, and evaluated on the GDP.pdf benchmark. Design: `docs/TRD.md`,
`docs/architecture.md`, `docs/service-spec.md`.

## Design philosophy

1. **Think before coding.** State assumptions. If more than one reading exists, name them. If something is
   unclear, stop and ask.
2. **Simplicity first.** Write the minimum code that solves the problem. No speculative features, no
   single-use abstractions, no configurability nobody asked for. If 200 lines could be 50, rewrite it.
3. **Surgical changes.** Every changed line traces to the request. Match the existing style. Remove only the
   orphans your change made; mention other dead code, do not delete it.
4. **Goal-driven execution.** Turn each task into a check that can pass or fail (a test, a score), and loop
   until it passes. For multi-step work, write the plan as steps with a check for each.

## Project rules

- **GDP.pdf is for evaluation only.** Never train on it, never put its text or examples in data or prompts,
  and never tune a change on its results and then re-run it. Choose every change on dev data (ParseBench,
  doc-QA); run GDP.pdf only at the baseline and at milestones, and record the result whether better or worse.
- **No scaffolding to fix accuracy.** The router dispatches by job kind only. Do not add request
  classifiers, output validation and retry, per-task prompts, or best-of-N at serve time. Accuracy gains come
  from data, training, or models.
- **One prompt path.** The training `Env` (`docqa.py`) builds its prompt by calling the service functions
  (`service.handle`, `parse`, `content`). Keep effort, token budget, and max_tokens equal in both.
- **Spend ledger.** Log every Tinker phase with `docqa.log_spend`; check `docqa.spent()` against the cap
  before a run.
- **Test-driven.** Commit the failing tests first, then make them pass. Run `uv run pytest` and
  `uv run ruff check src tests`; check the pytest exit status before you commit.
- **Secrets.** API keys come from the environment (`source ~/.zshrc`). Never print them or write them in a
  file.
- **Run outputs** (`var/`, `data/`) are never committed.
- **Writing.** Docs and messages are about 80% ASD-STE100: one instruction per sentence, active voice, short
  sentences. Code identifiers are exempt.
- **Readable code.** The code also teaches Tinker, async RL, and router/worker design: keep docstrings that
  explain the concept, not the obvious.

## Commands

```bash
uv sync --extra train
uv run gdpsolve serve --port 8787 --workers 16 [--checkpoint tinker://.../sampler_weights/...]
uv run gdpsolve eval gdp-pdf --system service --url http://127.0.0.1:8787 --concurrency 16 --out var/runs/<name>
uv run gdpsolve score gdp-pdf var/runs/<name>/results.jsonl
uv run gdpsolve synth docqa | eval docqa | train docqa
```

Use Context7 for current library docs (tinker, tinker-cookbook, pypdfium2) before you write against their APIs.
