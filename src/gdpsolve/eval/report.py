"""Untrained verify() vs. official GDP.pdf verdicts, and the eval report (docs/m1-baseline.md, step 3)."""

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

from gdpsolve.eval.gdp_pdf import criteria
from gdpsolve.schemas import CheckerConfig, Criterion, VerifyRequest

TEMPLATE = Path(__file__).with_name("report_template.html")
_DATA = re.compile(r'(<script type="application/json" id="report-data">\n).*?(\n</script>)', re.DOTALL)


def decisions(tasks: list[dict], grades: list[dict], results: dict[str, dict]) -> list[dict]:
    """One report decision per scored official verdict (unscored verdicts are excluded); `p` is empty.

    `results` are the ok results.jsonl rows by request_id; each decision cites its row and prompt.
    """
    meta = {f"{t['task_id']}:{c['n']}": c for t in tasks for c in criteria(t)}
    return [{"task_id": g["task_id"], "criterion_id": g["criterion_id"], "criterion": g["criterion"],
             "type": meta[g["criterion_id"]]["type"], "severity": meta[g["criterion_id"]]["severity"],
             "response": results[g["task_id"]]["answer"], "label": int(g["result"] == "pass"), "p": {},
             "request_id": results[g["task_id"]]["request_id"],
             "prompt_sha256": results[g["task_id"]].get("prompt_sha256")}
            for g in grades if g["result"] != "unscored"]


async def score(rows: list[dict], cfg: CheckerConfig, concurrency: int = 8) -> None:
    """Fill d["p"][cfg.model_ref] with verify() P(pass), one VerifyRequest per task."""
    from gdpsolve.verifier import verify

    by_task = defaultdict(list)
    for d in rows:
        by_task[d["task_id"]].append(d)
    gate = asyncio.Semaphore(concurrency)

    async def one(ds: list[dict]) -> None:
        req = VerifyRequest(response=ds[0]["response"],
                            criteria=[Criterion(id=d["criterion_id"], text=d["criterion"]) for d in ds])
        async with gate:
            result = await verify(req, cfg)
        p = {s.criterion_id: s.p_pass for s in result.scores}
        for d in ds:
            d["p"][result.checkpoint] = p.get(d["criterion_id"])

    await asyncio.gather(*(one(ds) for ds in by_task.values()))


def brier(rows: list[dict], ref: str) -> tuple[float | None, int]:
    """Brier reward 1 - mean((p - label)^2) over scored decisions, and how many were scored."""
    scored = [d for d in rows if d["p"].get(ref) is not None]
    if not scored:
        return None, 0
    return 1 - sum((d["p"][ref] - d["label"]) ** 2 for d in scored) / len(scored), len(scored)


def estimate(rows: list[dict], prefill_per_m: float = 0.70, overhead: int = 40) -> float:
    """Rough verify() prefill cost ($): about 4 characters per token, plus prompt overhead."""
    tokens = sum((len(d["response"]) + len(d["criterion"])) // 4 + overhead for d in rows)
    return tokens * prefill_per_m / 1e6


def render(data: dict, request_ids: set[str] = frozenset(), template: Path = TEMPLATE) -> str:
    """The template with its report-data block replaced by `data`.

    Real data (meta.sample false) must be traceable: every decision needs a prompt_sha256 and a
    request_id among `request_ids` (the results rows whose answers were scored) equal to its task_id.
    """
    if not data["meta"]["sample"]:
        bad = [d["criterion_id"] for d in data["decisions"] if not d.get("prompt_sha256")
               or d.get("request_id") not in request_ids or d["task_id"] != d["request_id"]]
        if bad:
            raise ValueError(f"{len(bad)} decisions lack provenance, e.g. {bad[:3]}")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html, n = _DATA.subn(lambda m: m.group(1) + payload + m.group(2), template.read_text())
    if n != 1:
        raise ValueError(f"{template}: report-data block not found")
    return html
