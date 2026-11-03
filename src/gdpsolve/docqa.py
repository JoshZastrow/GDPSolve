"""Doc-QA: training data, the RL Env, and the dev benchmark for Inkling-Small (docs/training.md). Never reads GDP.pdf.

A task is a 6-page PDF bundle (2 target pages + 4 distractors from ParseBench), a professional request that needs
facts from both target pages, and 6-10 rubric criteria. A teacher model (Inkling) writes the request and criteria
from the 2 target pages only. The policy (Inkling-Small) sees the whole bundle exactly as the service sends it.

Tinker concepts in this file:
- Renderer: turns chat messages into a token prompt (tinker.ModelInput) and parses sampled tokens back into a
  message. One renderer per model family; tml_v0 (Inkling) also takes a reasoning `effort`.
- Sampling client: a Tinker handle to a model (base weights or a tinker:// checkpoint) that samples tokens.
- Env: one single-use episode. initial_observation() gives the prompt; step(tokens) grades the completion.
- EnvGroupBuilder: group_size copies of one task's Env. The trainer centers rewards within each group, so a
  completion is pushed up only when it scores above its siblings on the same task (group-relative advantages).
- RLDataset / RLDatasetBuilder: batches of EnvGroupBuilders; the trainer asks for batch i at step i.
- BenchmarkBuilder: the same Env, one per dev row, for evaluation (run_benchmark, BenchmarkEvaluator).
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chz
import tinker
from tinker_cookbook.eval.benchmarks import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    register,
)
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)

from gdpsolve import service

PAGES = Path("data/parsebench/docs")  # ParseBench single-page PDFs (Apache-2.0)
OUT = Path("var/docqa")
SPEND = Path("var/train/spend.jsonl")
TEACHER = "thinkingmachines/Inkling"
POLICY = "thinkingmachines/Inkling-Small"
GRADER = "Qwen/Qwen3.8-27B"  # schemas.CHECKERS["rubric"]
CAP_USD = 15.0  # Tinker budget for all training-side work
# Tinker $ per 1M tokens
PRICES = {
    POLICY: {"prefill": 0.58, "sample": 1.44, "train": 1.73},
    TEACHER: {"prefill": 1.87, "sample": 4.68},
    GRADER: {"prefill": 0.6975},
}
MIN_CHARS, MAX_CHARS = 200, 6000  # target pages: a text layer with enough facts, and not too long

# --- data ---

TEACHER_PROMPT = """\
Below are two pages from a real document. You are writing one evaluation task for an assistant that answers \
questions about documents.

Write:
1. "prompt": a realistic request that a professional (for example a financial analyst, engineer, clinician, \
lawyer, or policy analyst) would send together with "the attached document". The request must need specific \
facts from BOTH pages: an analysis, a comparison, a calculation, or a summary. Write it in 2 to 5 sentences, \
the way that person would: state the goal and the deliverable, not a list of the individual values the answer \
must contain. The attached document will also contain other, unrelated pages, so do not mention page numbers \
and do not say where the facts are.
2. "criteria": 6 to 10 grading criteria. Each criterion is one checkable statement that starts with \
"The response should" and names one specific value or fact from the pages (a number with its unit, a name, a \
date, or a computed result with its value). Use facts from both pages, and only facts that a good answer to \
the request must contain. A grader will check each criterion against the response alone, without the document.

Return only JSON: {"prompt": "...", "criteria": ["The response should ...", ...]}"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _source(path: Path) -> tuple[str, int] | None:
    """('<document>', page) from a ParseBench file name '<document>_p<page>.pdf'."""
    m = re.fullmatch(r"(.*)_p(\d+)\.pdf", path.name)
    return (m.group(1), int(m.group(2))) if m else None


def eligible(pages_dir: Path = PAGES) -> tuple[list[Path], list[Path]]:
    """(target candidates, distractor pool). Each file name once; targets have MIN_CHARS..MAX_CHARS of text."""
    import pypdfium2 as pdfium

    from gdpsolve.eval import pdf_input

    seen, targets, pool = set(), [], []
    for path in sorted(pages_dir.glob("*/*.pdf")):
        if path.name in seen:
            continue
        seen.add(path.name)
        pdf = pdfium.PdfDocument(path)
        if len(pdf) == 1:
            pool.append(path)
            if _source(path) and MIN_CHARS <= len(pdf_input.page_text(pdf[0])) <= MAX_CHARS:
                targets.append(path)
        pdf.close()
    return targets, pool


def page_pairs(targets: list[Path], seed: int) -> list[tuple[Path, Path]]:
    """Disjoint pairs of neighbouring pages from the same source document, shuffled by `seed`."""
    by_doc: dict[str, list[tuple[int, Path]]] = {}
    for path in targets:
        if src := _source(path):
            by_doc.setdefault(src[0], []).append((src[1], path))
    pairs = []
    for doc in sorted(by_doc):
        ordered = [p for _, p in sorted(by_doc[doc])]
        pairs += [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered) - 1, 2)]
    random.Random(seed).shuffle(pairs)
    return pairs


def merge(paths: list[Path]) -> bytes:
    """One PDF with the pages of `paths`, in order. pdfium, as the parser: pypdf fails on some ParseBench files."""
    import pypdfium2 as pdfium

    out = pdfium.PdfDocument.new()
    for path in paths:
        src = pdfium.PdfDocument(path)
        out.import_pages(src)
        src.close()
    buf = io.BytesIO()
    out.save(buf)
    out.close()
    return buf.getvalue()


def make_bundle(targets: list[Path], pool: list[Path], seed: str, n_distractors: int = 4) -> tuple[bytes, list[dict]]:
    """The task document: the target pages and `n_distractors` other pages from `pool`, shuffled by `seed`.

    Returns (PDF bytes, [{page, source, target}] in page order).
    """
    rng = random.Random(seed)
    distractors = rng.sample([p for p in pool if p not in targets], n_distractors)
    order = [*targets, *distractors]
    rng.shuffle(order)
    sources = [{"page": n, "source": f"{p.parent.name}/{p.name}", "target": p in targets}
               for n, p in enumerate(order, start=1)]
    return merge(order), sources


def parse_teacher(text: str) -> dict | None:
    """{prompt, criteria} from the teacher's answer, or None when it is malformed."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        task = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        return None
    if not isinstance(task, dict) or not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
        return None
    criteria = task.get("criteria")
    if not isinstance(criteria, list) or not 6 <= len(criteria) <= 10:
        return None
    if not all(isinstance(c, str) and c.startswith("The response should") for c in criteria):
        return None
    return {"prompt": task["prompt"].strip(), "criteria": criteria}


def split(rows: list[dict], n_dev: int, seed: int) -> tuple[list[dict], list[dict]]:
    """(train, dev): the task ids sorted, shuffled by `seed`; the first n_dev are dev."""
    rows = sorted(rows, key=lambda r: r["task_id"])
    random.Random(seed).shuffle(rows)
    dev = sorted(rows[:n_dev], key=lambda r: r["task_id"])
    return sorted(rows[n_dev:], key=lambda r: r["task_id"]), dev


def teacher_prompts(pairs: list[tuple[Path, Path]]) -> tuple[Renderer, list[tinker.ModelInput]]:
    """The teacher's prompt for each pair: TEACHER_PROMPT and the two pages as the service parses them."""
    cfg = service.Config(parser="native", model=TEACHER)  # its sampler only counts tokens here
    renderer = cfg.sampler._renderer()
    prompts = []
    for pair in pairs:
        file_id = service.handle(cfg, "upload", merge(list(pair)))["file_id"]
        chunks = service.parse(cfg, {"input": file_id})["result"]["chunks"]
        content, _ = service.content(chunks, TEACHER_PROMPT, cfg.sampler.count_tokens, cfg.token_budget)
        # effort 0: at higher effort the teacher spends max_tokens on thinking before the JSON
        prompts.append(renderer.build_generation_prompt([{"role": "user", "content": content}], effort=0.0))
    return renderer, prompts


async def ask_teacher(renderer: Renderer, prompts: list[tinker.ModelInput], max_tokens: int = 2048,
                      concurrency: int = 32) -> list[tuple[str, int]]:
    """(answer text, completion tokens) for each prompt: one sample from the teacher's base weights."""
    from gdpsolve.eval import baseline

    client = await tinker.ServiceClient().create_sampling_client_async(base_model=TEACHER)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    gate = asyncio.Semaphore(concurrency)

    async def one(prompt: tinker.ModelInput) -> tuple[str, int]:
        async with gate:
            tokens = (await client.sample_async(prompt, num_samples=1, sampling_params=params)).sequences[0].tokens
        return baseline.answer_text(renderer, tokens)[0], len(tokens)

    return await asyncio.gather(*(one(p) for p in prompts))


def build(pairs: list[tuple[Path, Path]], pool: list[Path], answers: list[str], seed: int,
          out: Path = OUT, n_dev: int = 32) -> dict[str, int]:
    """Write the bundles (out/pdfs) and train.jsonl / dev.jsonl for the pairs whose teacher answer is well formed."""
    rows = []
    for i, (pair, answer) in enumerate(zip(pairs, answers, strict=True)):
        task = parse_teacher(answer)
        if task is None:
            continue
        task_id = f"docqa-{seed}-{i:04d}"
        pdf, sources = make_bundle(list(pair), pool, seed=task_id)
        path = out / "pdfs" / f"{task_id}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf)
        rows.append({"task_id": task_id, "pdf": str(path), **task, "source_pages": sources})
    train, dev = split(rows, n_dev, seed)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    return {"asked": len(pairs), "kept": len(rows), "train": len(train), "dev": len(dev)}


# --- spend ---


def cost(model: str, **tokens: float) -> float:
    """Tinker $ for token counts by kind, e.g. cost(POLICY, prefill=..., sample=..., train=...)."""
    return sum(n * PRICES[model][kind] for kind, n in tokens.items()) / 1e6


def spent(path: Path = SPEND) -> float:
    return read_jsonl(path)[-1]["total_usd"] if path.exists() and path.read_text().strip() else 0.0


def log_spend(phase: str, usd: float, path: Path = SPEND, **detail) -> float:
    """Append one phase's Tinker cost to var/train/spend.jsonl; returns the running total."""
    total = spent(path) + usd
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"phase": phase, "usd": round(usd, 4), "total_usd": round(total, 4), **detail}) + "\n")
    return total


def rl_cost(row: dict) -> float:
    """Tinker $ for one metrics.jsonl row of an RL run: the step's rollouts (sample, train, grade) and/or an
    inline dev eval. Prefill is priced uncached, so this is an upper bound."""
    usd = 0.0
    if "env/all/total_ob_tokens" in row:
        ob, ac = row["env/all/total_ob_tokens"], row["env/all/total_ac_tokens"]
        usd += cost(POLICY, prefill=ob, sample=ac, train=ob + ac)
        usd += cost(GRADER, prefill=row.get("env/all/grader_tokens", 0) * row["env/all/total_episodes"])
    e = f"eval/{DocQABenchmark.name}/"
    if e + "total_ob_tokens" in row:
        usd += cost(POLICY, prefill=row[e + "total_ob_tokens"], sample=row[e + "total_ac_tokens"])
        usd += cost(GRADER, prefill=row.get(e + "grader_tokens_total", 0))
    return usd


def headroom(records: list[dict]) -> dict[int, dict]:
    """Per training step, from the trainer's rollout records: mean@k (all rollouts), max@k (mean over groups of
    the best rollout), and frac_equal, the share of groups whose rewards are all within 0.01. Such a group has
    zero advantage after centering, so it teaches nothing."""
    groups: dict[int, dict[int, list[float]]] = {}
    for r in records:
        groups.setdefault(r["iteration"], {}).setdefault(r["group_idx"], []).append(r["total_reward"])
    out = {}
    for step, by_group in sorted(groups.items()):
        rewards = list(by_group.values())
        flat = [x for g in rewards for x in g]
        out[step] = {"mean@k": sum(flat) / len(flat), "max@k": sum(map(max, rewards)) / len(rewards),
                     "frac_equal": sum(max(g) - min(g) <= 0.01 for g in rewards) / len(rewards),
                     "groups": len(rewards)}
    return out


# --- env ---


class DocQAEnv(Env):
    """One doc-QA episode: the service's extract prompt for the bundle, graded by the rubric checker."""

    def __init__(self, row: dict, renderer: Renderer):
        self.row, self.renderer = row, renderer
        self.example_id = row["task_id"]  # stable id for the benchmark runner's results and resume

    def _prompt(self) -> tinker.ModelInput:
        """Upload, parse, and build the user message as service.extract does; render it at the service's effort."""
        cfg = service.Config(sampler=self)  # parse only reads `parser`; count_tokens is ours
        file_id = service.handle(cfg, "upload", Path(self.row["pdf"]).read_bytes())["file_id"]
        chunks = service.parse(cfg, {"input": file_id})["result"]["chunks"]
        content, _ = service.content(chunks, self.row["prompt"], self.count_tokens, cfg.token_budget)
        return self.renderer.build_generation_prompt([{"role": "user", "content": content}], effort=cfg.effort)

    def count_tokens(self, parts: list[dict]) -> int:
        """As service.Tinker.count_tokens: the user message with `parts`, without effort."""
        return self.renderer.build_generation_prompt([{"role": "user", "content": parts}]).length

    async def initial_observation(self):
        prompt = await asyncio.to_thread(self._prompt)  # pdfium and image rendering would block the event loop
        return prompt, self.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None) -> StepResult:
        """Reward = mean P(pass) of the rubric checker over the criteria it scored (0 for an empty answer)."""
        from gdpsolve.eval import baseline
        from gdpsolve.schemas import CHECKERS, Criterion, VerifyRequest
        from gdpsolve.verifier import checkers

        text, clean = baseline.answer_text(self.renderer, action)  # the text part, as the service returns it
        truncated = not clean or (extra or {}).get("stop_reason") == "length"
        criteria = [Criterion(id=f"{self.example_id}:{i}", text=c) for i, c in enumerate(self.row["criteria"])]
        score, grader_tokens = 0.0, 0
        if text.strip():
            result = await checkers.verify(VerifyRequest(response=text, criteria=criteria), CHECKERS["rubric"])
            score = result.mean_criteria or 0.0  # None when no criterion was scored
            grader_tokens = len(criteria) * (len(text) // 4 + 100)  # estimate: response + criterion + template
        return StepResult(
            reward=score,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"rubric_score": score, "completion_tokens": len(action), "truncated": float(truncated),
                     "grader_tokens": grader_tokens},
            logs={"task_id": self.example_id, "response": text[:2000]},
        )


@dataclass(frozen=True)
class DocQAGroupBuilder(EnvGroupBuilder):
    """group_size Envs for one task: their rewards are centered against each other."""

    row: dict
    renderer: Renderer
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        return [DocQAEnv(self.row, self.renderer) for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        return ["docqa"]


@dataclass(frozen=True)
class DocQADataset(RLDataset):
    """Batch i is rows[i * batch_size : (i + 1) * batch_size], one group each."""

    rows: list[dict]
    renderer: Renderer
    batch_size: int
    group_size: int

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        rows = self.rows[index * self.batch_size:(index + 1) * self.batch_size]
        return [DocQAGroupBuilder(r, self.renderer, self.group_size) for r in rows]

    def __len__(self) -> int:
        return len(self.rows) // self.batch_size


@chz.chz
class DocQADatasetBuilder(RLDatasetBuilder):
    """train.jsonl, shuffled by `seed`. No test set: the dev eval runs as a BenchmarkEvaluator."""

    model_name: str
    renderer_name: str
    batch_size: int
    group_size: int
    train_path: str = str(OUT / "train.jsonl")
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        rows = read_jsonl(Path(self.train_path))
        random.Random(self.seed).shuffle(rows)
        return DocQADataset(rows, renderer, self.batch_size, self.group_size), None


# --- benchmark ---


class DocQABenchmark(BenchmarkBuilder):
    """dev.jsonl as a benchmark. score = mean reward (mean rubric score); errors and timeouts count as 0."""

    name = "gdpsolve_docqa"
    recommended_timeout = 900  # 8192 sampled tokens plus up to 10 grader calls

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        rows = read_jsonl(OUT / "dev.jsonl")[: config.max_examples]
        return [DocQAEnv(row, renderer) for row in rows]

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        def mean(key: str) -> float:
            values = [m[key] for m in metrics_list if key in m]
            return sum(values) / len(values) if values else 0.0

        metrics = {k: mean(k) for k in ("rubric_score", "completion_tokens", "truncated")}
        metrics["grader_tokens_total"] = sum(m.get("grader_tokens", 0) for m in metrics_list)
        return BenchmarkResult(name=self.name, score=sum(rewards) / len(rewards) if rewards else 0.0,
                               num_examples=len(rewards), num_correct=sum(r > 0 for r in rewards), metrics=metrics)


register(DocQABenchmark())
