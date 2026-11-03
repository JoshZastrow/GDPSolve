"""Tests for the doc-QA data, Env, and benchmark (docs/training.md). Offline: fake verify, fake renderer."""

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from test_run import make_pdf

pytest.importorskip("tinker_cookbook")  # gdpsolve.docqa needs the train extra

from gdpsolve import docqa, service
from gdpsolve.schemas import CriterionScore, VerifyResult


@pytest.fixture
def var(tmp_path, monkeypatch):
    """Run with var/ (object store, jobs, docqa, spend) in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "var"


def pages(tmp_path, names: list[str]) -> list:
    """One single-page PDF for each name, in tmp_path/pages/<name>; the page text is the name."""
    (tmp_path / "pages").mkdir(exist_ok=True)
    paths = []
    for name in names:
        path = tmp_path / "pages" / name
        path.write_bytes(make_pdf([name.removesuffix(".pdf")]))
        paths.append(path)
    return paths


def page_texts(pdf: bytes) -> list[str]:
    from pypdf import PdfReader

    return [p.extract_text().strip() for p in PdfReader(io.BytesIO(pdf)).pages]


class FakeRenderer:
    """Each token is one character of the completion's text part."""

    def parse_response(self, tokens):
        return {"role": "assistant", "content": "".join(map(chr, tokens))}, SimpleNamespace(is_clean=True)


def row(task_id="t1", pdf="doc.pdf", criteria=("The response should say 42.", "The response should say 7.")):
    return {"task_id": task_id, "pdf": pdf, "prompt": "q", "criteria": list(criteria), "source_pages": []}


# --- data ---


def test_bundle_has_six_pages_two_targets_and_is_deterministic(tmp_path):
    paths = pages(tmp_path, [f"doc{i}_p1.pdf" for i in range(9)])
    targets, pool = paths[:2], paths
    pdf, order = docqa.make_bundle(targets, pool, seed="docqa-0-0001")
    texts = page_texts(pdf)
    assert len(texts) == 6 and len(set(texts)) == 6
    assert {"doc0_p1", "doc1_p1"} <= set(texts)
    assert [o["source"] for o in order] == [f"pages/{t}.pdf" for t in texts]
    assert [o["target"] for o in order].count(True) == 2
    assert [o["page"] for o in order] == [1, 2, 3, 4, 5, 6]
    again, again_order = docqa.make_bundle(targets, pool, seed="docqa-0-0001")
    assert page_texts(again) == texts and again_order == order  # the bytes hold a creation date
    orders = {tuple(o["source"] for o in docqa.make_bundle(targets, pool, seed=f"s{i}")[1]) for i in range(5)}
    assert len(orders) > 1


def test_page_pairs_are_adjacent_pages_of_one_source_and_disjoint(tmp_path):
    paths = pages(tmp_path, ["a_p1.pdf", "a_p2.pdf", "a_p3.pdf", "a_p7.pdf", "b_p4.pdf", "b_p5.pdf", "c_p1.pdf"])
    pairs = docqa.page_pairs(paths, seed=0)
    names = sorted((x.name, y.name) for x, y in pairs)
    assert names == [("a_p1.pdf", "a_p2.pdf"), ("a_p3.pdf", "a_p7.pdf"), ("b_p4.pdf", "b_p5.pdf")]
    assert docqa.page_pairs(paths, seed=0) == pairs


def test_parse_teacher_keeps_wellformed_tasks_only():
    good = {"prompt": "Compare the totals.", "criteria": [f"The response should state value {i}." for i in range(6)]}
    assert docqa.parse_teacher("Here:\n" + json.dumps(good)) == good
    assert docqa.parse_teacher("not json") is None
    assert docqa.parse_teacher(json.dumps(good | {"criteria": good["criteria"][:5]})) is None  # fewer than 6
    assert docqa.parse_teacher(json.dumps(good | {"criteria": good["criteria"] * 2})) is None  # more than 10
    bad = good | {"criteria": ["It should state 1."] + good["criteria"][1:]}
    assert docqa.parse_teacher(json.dumps(bad)) is None
    assert docqa.parse_teacher(json.dumps(good | {"prompt": ""})) is None


def test_jsonl_round_trip(tmp_path):
    rows = [row("a"), row("b", criteria=["The response should say ü."])]
    docqa.write_jsonl(tmp_path / "x" / "rows.jsonl", rows)
    assert docqa.read_jsonl(tmp_path / "x" / "rows.jsonl") == rows


def test_split_is_by_task_id_and_reproducible():
    rows = [row(f"docqa-0-{i:04d}") for i in range(40)]
    train, dev = docqa.split(rows, n_dev=8, seed=0)
    assert len(train) == 32 and len(dev) == 8
    assert not {r["task_id"] for r in train} & {r["task_id"] for r in dev}
    assert docqa.split(list(reversed(rows)), n_dev=8, seed=0) == (train, dev)


# --- spend ---


def test_spend_log_keeps_a_running_total(var):
    assert docqa.spent() == 0
    assert docqa.log_spend("synth", 1.25, prefill_tokens=10) == pytest.approx(1.25)
    assert docqa.log_spend("eval", 0.5) == pytest.approx(1.75)
    assert docqa.spent() == pytest.approx(1.75)
    lines = [json.loads(line) for line in (var / "train" / "spend.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in lines] == ["synth", "eval"] and lines[1]["total_usd"] == pytest.approx(1.75)


def test_rl_cost_counts_rollouts_training_grading_and_inline_eval():
    m = 1e6
    train_row = {"env/all/total_ob_tokens": m, "env/all/total_ac_tokens": m, "env/all/total_episodes": 10,
                 "env/all/grader_tokens": m / 10}
    # Inkling-Small prefill + sample, train on prompt and completion, Qwen grader prefill
    assert docqa.rl_cost(train_row) == pytest.approx(0.58 + 1.44 + 2 * 1.73 + 0.6975)
    eval_row = {"eval/gdpsolve_docqa/total_ob_tokens": m, "eval/gdpsolve_docqa/total_ac_tokens": m,
                "eval/gdpsolve_docqa/grader_tokens_total": m}
    assert docqa.rl_cost(eval_row) == pytest.approx(0.58 + 1.44 + 0.6975)
    assert docqa.rl_cost({"optim/lr": 1e-5}) == 0


# --- env ---


def test_env_reward_is_mean_p_pass_over_scored_criteria(monkeypatch):
    from gdpsolve.verifier import checkers

    seen = []

    async def fake_verify(req, cfg):
        seen.append((req, cfg))
        return VerifyResult(scores=[CriterionScore(criterion_id=c.id, p_pass=p)
                                    for c, p in zip(req.criteria, [1.0, 0.5, None], strict=True)], checkpoint="fake")

    monkeypatch.setattr(checkers, "verify", fake_verify)
    env = docqa.DocQAEnv(row(criteria=["The response should a.", "The response should b.", "The response should c."]),
                         FakeRenderer())
    result = asyncio.run(env.step([ord(c) for c in "The answer is 42."], extra={"stop_reason": "stop"}))
    assert result.episode_done and result.reward == pytest.approx(0.75)
    assert result.metrics["rubric_score"] == pytest.approx(0.75)
    assert result.metrics["completion_tokens"] == len("The answer is 42.") and result.metrics["truncated"] == 0
    req, cfg = seen[0]
    assert req.response == "The answer is 42." and req.criteria[0].text == "The response should a."
    assert cfg.base_model == "Qwen/Qwen3.8-27B"


def test_env_empty_answer_scores_zero_without_grading(monkeypatch):
    from gdpsolve.verifier import checkers

    async def fail(req, cfg):
        raise AssertionError("verify must not be called for an empty answer")

    monkeypatch.setattr(checkers, "verify", fail)
    result = asyncio.run(docqa.DocQAEnv(row(), FakeRenderer()).step([], extra={"stop_reason": "length"}))
    assert result.reward == 0 and result.metrics["truncated"] == 1 and result.metrics["rubric_score"] == 0


def test_env_prompt_equals_service_prompt(var, tmp_path):
    """PARITY: the Env's observation is token for token what the deployed service sends to extract."""
    model = "thinkingmachines/Inkling-Small"
    try:
        tinker = service.Tinker(model, None, effort=0.3, max_tokens=8192)
        renderer = tinker._renderer()
    except OSError as e:  # tokenizer not in the local cache
        pytest.skip(str(e))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(["Revenue 42", "Costs 7", "Other"]))
    ob, stop = asyncio.run(docqa.DocQAEnv(row(pdf=str(pdf)), renderer).initial_observation())

    cfg = service.Config(sampler=tinker, model=model)
    file_id = service.handle(cfg, "upload", pdf.read_bytes())["file_id"]
    chunks = service.parse(cfg, {"input": file_id})["result"]["chunks"]
    content, _ = service.content(chunks, "q", tinker.count_tokens, cfg.token_budget)
    expected = tinker.prompt(content)
    assert ob.chunks == expected.chunks
    assert stop == renderer.get_stop_sequences()


def test_group_builder_and_dataset():
    rows = [row(f"t{i}") for i in range(5)]
    dataset = docqa.DocQADataset(rows=rows, renderer=FakeRenderer(), batch_size=2, group_size=3)
    assert len(dataset) == 2
    batch = dataset.get_batch(1)
    assert [b.row["task_id"] for b in batch] == ["t2", "t3"]
    envs = asyncio.run(batch[0].make_envs())
    assert len(envs) == 3 and all(e.row["task_id"] == "t2" for e in envs) and len({id(e) for e in envs}) == 3


# --- benchmark ---


def test_benchmark_makes_one_env_per_dev_row(var):
    from tinker_cookbook.eval.benchmarks import REGISTRY, BenchmarkConfig

    docqa.write_jsonl(var / "docqa" / "dev.jsonl", [row(f"d{i}") for i in range(3)])
    bench = REGISTRY["gdpsolve_docqa"]
    envs = bench.make_envs(FakeRenderer(), BenchmarkConfig())
    assert [e.example_id for e in envs] == ["d0", "d1", "d2"]
    assert all(isinstance(e, docqa.DocQAEnv) for e in envs)
    assert len(bench.make_envs(FakeRenderer(), BenchmarkConfig(max_examples=2))) == 2

    metrics = [{"rubric_score": 1.0, "completion_tokens": 10, "truncated": 0, "grader_tokens": 100},
               {"rubric_score": 0.5, "completion_tokens": 30, "truncated": 1, "grader_tokens": 300}]
    result = bench.aggregate([1.0, 0.5], metrics)
    assert result.score == pytest.approx(0.75) and result.num_examples == 2
    assert result.metrics["rubric_score"] == pytest.approx(0.75)
    assert result.metrics["truncated"] == pytest.approx(0.5)
    assert result.metrics["grader_tokens_total"] == 400


def test_headroom_per_step_from_rollout_records():
    def rec(it, g, r):
        return {"split": "train", "iteration": it, "group_idx": g, "traj_idx": 0, "total_reward": r}

    records = [rec(1, 0, 0.2), rec(1, 0, 0.6), rec(1, 1, 0.5), rec(1, 1, 0.505),
               rec(2, 0, 1.0), rec(2, 0, 0.0)]
    steps = docqa.headroom(records)
    assert steps[1] == pytest.approx({"mean@k": 0.45125, "max@k": (0.6 + 0.505) / 2, "frac_equal": 0.5, "groups": 2})
    assert steps[2] == pytest.approx({"mean@k": 0.5, "max@k": 1.0, "frac_equal": 0.0, "groups": 1})
