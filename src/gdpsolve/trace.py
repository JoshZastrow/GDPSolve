"""Run tracing, written into one run directory.

  config.json, code.diff, metrics.jsonl, logs.log   tinker_cookbook.utils.ml_log.setup_logging
  trace_events.jsonl                                tinker_cookbook.utils.trace (timing of every Tinker call; Perfetto)
  calls.jsonl                                       what each model was shown and what it returned (call())

call() is a no-op until start() runs, so library code can record unconditionally.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

_calls: TextIO | None = None


def start(run_dir: Path, config: dict[str, Any]):
    """Begin tracing a run; returns the ml_log logger for metrics."""
    global _calls
    from tinker_cookbook.utils import ml_log, trace

    run_dir.mkdir(parents=True, exist_ok=True)
    trace.trace_init(output_file=str(run_dir / "trace_events.jsonl"))
    logger = ml_log.setup_logging(str(run_dir), config={"argv": sys.argv, **config})
    _calls = (run_dir / "calls.jsonl").open("a")
    return logger


def call(kind: str, **fields: Any) -> None:
    """Append one model-call record: kind (e.g. "verify", "support", "baseline") plus its fields."""
    if _calls is not None:
        _calls.write(json.dumps({"ts": round(time.time(), 3), "kind": kind, **fields}, default=str) + "\n")
        _calls.flush()


def prompt_text(tokenizer, prompt) -> str:
    """A tinker.ModelInput as the model saw it, with each image chunk shown as <image>."""
    import tinker

    return "".join(
        tokenizer.decode(list(c.tokens)) if isinstance(c, tinker.EncodedTextChunk) else "<image>"
        for c in prompt.chunks
    )


def stop(logger=None) -> None:
    global _calls
    from tinker_cookbook.utils import trace

    trace.trace_shutdown()
    if logger is not None:
        logger.close()
    if _calls is not None:
        _calls.close()
        _calls = None
