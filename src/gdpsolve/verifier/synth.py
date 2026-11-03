"""Synthetic check set for the untrained verifiers (docs/verifier-trd.md). Never reads GDP.pdf.

A teacher model writes a short fictional document excerpt with numeric facts and a correct answer.
Deterministic edits of that answer give responses whose labels are known by construction.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import re
from pathlib import Path

from gdpsolve.schemas import (
    BoundingBox,
    Claim,
    CriterionScore,
    Region,
    SourceSpan,
    SupportRequest,
    VerifyResult,
)

TEACHER = "thinkingmachines/Inkling-Small"
DOMAINS = [
    "corporate finance",
    "accounting",
    "legal contracts",
    "healthcare",
    "real estate",
    "construction",
    "insurance",
    "government procurement",
    "energy",
    "logistics",
]
PAD = (
    " For context, documents of this type are typically prepared by the responsible department and "
    "reviewed before distribution. Formatting conventions may vary between editions."
)
PROMPT = """Write a fictional excerpt from a {domain} document, 120-200 words, with exactly 4 distinct numeric facts \
(amounts, dates, percentages, counts). Each fact's value must appear exactly once in the excerpt, in its own sentence.
Then write a question about the excerpt that needs all 4 facts, and a correct answer that states each value verbatim.
Return only JSON:
{{"passage": "...", "question": "...", "answer": "...",
  "facts": [{{"id": "f1", "value": "<exact string from the passage>",
             "criterion": "The response should state <what>: <value>.",
             "statement": "<one-sentence claim containing the value>"}}, ...]}}"""


# --- edits: labels by construction ---------------------------------------------


def alter(value: str) -> str:
    """Change the first non-zero digit, so the number is plainly different."""
    for i, ch in enumerate(value):
        if ch.isdigit() and ch != "0":
            return value[:i] + str(int(ch) % 9 + 1) + value[i + 1 :]
    raise ValueError(f"no digit to alter in {value!r}")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def sentence_with(text: str, value: str) -> str | None:
    return next((s for s in _sentences(text) if value in s), None)


def drop(answer: str, value: str, others: list[str]) -> str:
    """Remove `value` only: its sentence if no other fact shares it, else just the value itself."""
    sentence = sentence_with(answer, value)
    if sentence and not any(o in sentence for o in others):
        return " ".join(s for s in _sentences(answer) if s != sentence)
    text = answer.replace(value, "")
    return re.sub(r"\s*,\s*(,\s*)+", ", ", re.sub(r"\s+([,.])", r"\1", text)).strip()


def edits(answer: str, value: str, others: list[str] = ()) -> dict[str, str]:
    """Responses that each break the criterion for `value` and leave the `others` intact."""
    return {
        "swap_number": answer.replace(value, alter(value)),
        "drop_fact": drop(answer, value, list(others)),
        "hedge": answer.replace(value, f"either {value} or {alter(value)}"),
    }


def valid(task: dict) -> bool:
    facts = task.get("facts") or []
    return len(facts) >= 3 and all(
        f.get("value")
        and any(c.isdigit() for c in f["value"])
        and task["passage"].count(f["value"]) == 1
        and f["value"] in task["answer"]
        and f["value"] in f.get("statement", "")
        and sentence_with(task["passage"], f["value"])
        for f in facts
    )


def rubric_rows(task: dict, rng: random.Random) -> list[dict]:
    """correct + pad (all yes); swap/drop/hedge on one random fact each (that criterion no)."""
    facts, answer = task["facts"], task["answer"]
    responses = [("correct", answer, None), ("pad", answer + PAD, None)]
    for edit in ("swap_number", "drop_fact", "hedge"):
        f = rng.choice(facts)
        others = [o["value"] for o in facts if o is not f]
        responses.append((edit, edits(answer, f["value"], others)[edit], f["id"]))
    return [
        {
            "task_id": task["task_id"],
            "edit": edit,
            "broken": broken,
            "response": text,
            "criterion_id": f"{task['task_id']}:{f['id']}",
            "criterion": f["criterion"],
            "label": 0 if f["id"] == broken else 1,
        }
        for edit, text, broken in responses
        for f in facts
    ]


def render_png(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) > 60:
            lines.append(line)
            line = ""
        line = f"{line} {word}".strip()
    lines.append(line)
    img = Image.new("RGB", (760, 34 * len(lines) + 24), "white")
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        draw.text((12, 12 + 34 * i), ln, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def support_rows(task: dict, images: bool, put=None) -> list[dict]:
    """True claim + its sentence (yes); swapped number or another fact's sentence (no).

    With images, evidence is a rendered PNG region; one extra negative per task: a blank image.
    """
    facts, passage, tid = task["facts"], task["passage"], task["task_id"]
    if images and put is None:
        from gdpsolve.verifier.store import put
    rows = []

    def request(claim: str, evidence: str, kind: str, label: int) -> None:
        span = SourceSpan(id="s1", page=1, text=evidence, method="text_layer")
        regions = []
        if images:
            region = Region(
                id="r1",
                doc_sha256=tid,
                page=1,
                dpi=100,
                image=put(render_png(evidence or " "), "image/png"),
                bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
            )
            span = span.model_copy(update={"method": "vlm", "region": "r1"})
            regions = [region]
        req = SupportRequest(claims=[Claim(id="k1", text=claim, source_ids=["s1"])], sources=[span], regions=regions)
        rows.append(
            {"task_id": tid, "kind": kind, "images": images, "request": req.model_dump(mode="json"), "label": label}
        )

    for i, f in enumerate(facts):
        own = sentence_with(passage, f["value"])
        other = sentence_with(passage, facts[(i + 1) % len(facts)]["value"])
        request(f["statement"], own, "true", 1)
        request(f["statement"].replace(f["value"], alter(f["value"])), own, "swapped_number", 0)
        request(f["statement"], other, "wrong_span", 0)
        if images and i == 0:
            request(f["statement"], "", "blank_image", 0)
    return rows


# --- generation (Tinker) -------------------------------------------------------


async def generate_tasks(n: int, seed: int = 0) -> list[dict]:
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.renderers import Message, get_text_content
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(TEACHER), get_tokenizer(TEACHER))
    client = await tinker.ServiceClient().create_sampling_client_async(base_model=TEACHER)
    params = tinker.SamplingParams(max_tokens=2000, temperature=0.9, stop=renderer.get_stop_sequences())

    async def one(i: int) -> dict | None:
        domain = DOMAINS[i % len(DOMAINS)]
        messages = [Message(role="user", content=PROMPT.format(domain=domain))]
        prompt = renderer.build_generation_prompt(messages, effort=0.0)  # no reasoning: it exhausts max_tokens
        out = await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        message, _ = renderer.parse_response(out.sequences[0].tokens)
        match = re.search(r"\{.*\}", get_text_content(message), re.DOTALL)
        try:
            task = json.loads(match.group()) if match else None
        except json.JSONDecodeError:
            return None
        if not task or not valid(task):
            return None
        return {"task_id": f"synth-{seed}-{i:04d}", "domain": domain, **task}

    tasks = await asyncio.gather(*(one(i) for i in range(n)))
    return [t for t in tasks if t]


def build(tasks: list[dict], out: Path, seed: int = 0) -> dict[str, int]:
    """Write tasks.jsonl, rubric.jsonl, support.jsonl; every other task uses image evidence."""
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    rubric = [row for t in tasks for row in rubric_rows(t, rng)]
    support = [row for i, t in enumerate(tasks) for row in support_rows(t, images=i % 2 == 1)]
    for name, rows in (("tasks", tasks), ("rubric", rubric), ("support", support)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"tasks": len(tasks), "rubric": len(rubric), "support": len(support)}


# --- evaluation ----------------------------------------------------------------


def brier_reward(pairs: list[tuple[float | None, int]]) -> float | None:
    scored = [(p, y) for p, y in pairs if p is not None]
    return 1 - sum((p - y) ** 2 for p, y in scored) / len(scored) if scored else None


async def score_rubric(rows: list[dict], cfg) -> list[float | None]:
    """P(pass) per row, one verify() call per (task, edit) response."""
    from gdpsolve.schemas import Criterion, VerifyRequest
    from gdpsolve.verifier import verify

    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["task_id"], r["edit"]), []).append(i)
    ps: list[float | None] = [None] * len(rows)
    sem = asyncio.Semaphore(8)

    async def one(idx: list[int]) -> None:
        criteria = [Criterion(id=rows[i]["criterion_id"], text=rows[i]["criterion"]) for i in idx]
        async with sem:
            v = await verify(VerifyRequest(response=rows[idx[0]]["response"], criteria=criteria), cfg)
        for i, s in zip(idx, v.scores, strict=True):
            ps[i] = s.p_pass

    await asyncio.gather(*(one(idx) for idx in groups.values()))
    return ps


async def score_support(rows: list[dict], cfg) -> list[float | None]:
    from gdpsolve.verifier import support

    sem = asyncio.Semaphore(8)

    async def one(r: dict) -> float | None:
        async with sem:
            return (await support(SupportRequest.model_validate(r["request"]), cfg)).scores["k1"]

    return list(await asyncio.gather(*(one(r) for r in rows)))


def rubric_report(rows: list[dict], ps: list[float | None]) -> dict:
    """Brier reward, plus the reward-ordering and pad-invariance checks from the spec."""
    from gdpsolve.verifier.reward import reward

    by_response: dict[tuple, list[CriterionScore]] = {}
    for r, p in zip(rows, ps, strict=True):
        by_response.setdefault((r["task_id"], r["edit"]), []).append(
            CriterionScore(criterion_id=r["criterion_id"], p_pass=p)
        )
    results = {k: VerifyResult(scores=v, checkpoint="") for k, v in by_response.items()}
    rewards = {k: reward(v, []) for k, v in results.items()}
    tasks = {t for t, _ in results}
    pairs = [(rewards[(t, "correct")], rewards[(t, e)]) for t in tasks for e in ("swap_number", "drop_fact", "hedge")]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    means = {k: v.mean_criteria for k, v in results.items()}
    pad = [
        abs(means[(t, "correct")] - means[(t, "pad")])
        for t in tasks
        if means[(t, "correct")] is not None and means[(t, "pad")] is not None
    ]
    return {
        "brier_reward": brier_reward(list(zip(ps, [r["label"] for r in rows], strict=True))),
        "reward_ordering": sum(a > b for a, b in pairs) / len(pairs) if pairs else None,
        "pad_delta": sum(pad) / len(pad) if pad else None,
        "n": len(rows),
        "unscored": sum(p is None for p in ps),
    }
