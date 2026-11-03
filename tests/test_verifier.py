import pytest

from gdpsolve.schemas import CriterionScore, ExtractedField, SourceSpan, VerifyResult

pytest.importorskip("tinker_cookbook")  # gdpsolve.verifier needs the train extra

from gdpsolve.verifier.grounding import check_field_pages, evaluate, parse_number
from gdpsolve.verifier.reward import reward


def test_evaluate_allows_arithmetic_only():
    names = {"s1": 1200.0, "s2": -50.0}
    assert evaluate("s1 + s2 * 2", names) == 1100.0
    assert evaluate("sum(s1, s2, 10) / 2", names) == 580.0
    assert evaluate("-s2", names) == 50.0
    for bad in ("__import__('os')", "s1.real", "open('x')", "s3 + 1", "s1 ** 2"):
        with pytest.raises(ValueError):
            evaluate(bad, names)


@pytest.mark.parametrize(
    "text,expected",
    [("(1,204)", -1204.0), ("$1,240", 1240.0), ("12.5%", 12.5), ("total -3.5 units", -3.5), ("none", None)],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def _span(i, text, page=3):
    return SourceSpan(id=f"s{i}", page=page, text=text, method="text_layer")


PAGES = {3: "Revenue (in thousands)\nJan 1,200  Feb 1,250  Mar (100)\nTotal Q1 2,350"}


def test_check_field_extracted_and_computed():
    ok = ExtractedField(name="q1", value="2,350", confidence=1, sources=[_span(1, "Total Q1 2,350")])
    assert check_field_pages(ok, PAGES).ok
    wrong_page = ok.model_copy(update={"sources": [_span(1, "Total Q1 2,350", page=4)]})
    assert not check_field_pages(wrong_page, PAGES).ok
    spans = [_span(1, "Jan 1,200"), _span(2, "Feb 1,250"), _span(3, "Mar (100)")]
    ytd = ExtractedField(name="ytd", value=2350, confidence=1, sources=spans, kind="computed", formula="s1 + s2 + s3")
    assert check_field_pages(ytd, PAGES).ok
    off = ytd.model_copy(update={"value": 2450})
    result = check_field_pages(off, PAGES)
    assert not result.ok and "2350" in result.reason
    with pytest.raises(ValueError):
        check_field_pages(ytd.model_copy(update={"kind": "summary"}), PAGES)


def test_reward_and_unscored():
    v = VerifyResult(scores=[CriterionScore(criterion_id="a", p_pass=1.0), CriterionScore(criterion_id="b", p_pass=0.5)],
                     checkpoint="m")
    assert reward(v, []) == pytest.approx(0.75 + 0.5)
    assert reward(v, [1.0, 0.0], beta=2.0) == pytest.approx(0.75 + 1.0 + 0.5)
    unscored = VerifyResult(scores=[CriterionScore(criterion_id="a")], checkpoint="m")
    assert reward(unscored, [1.0]) is None


def test_render_places_placeholder_and_matches_cookbook():
    from tinker_cookbook import renderers
    from tinker_cookbook.recipes.decisions import Decision, DecisionRenderer
    from tinker_cookbook.recipes.decisions.forecasting import YesNoChoiceRenderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from gdpsolve.schemas import CHECKERS
    from gdpsolve.verifier.decision import render

    cfg = CHECKERS["rubric"]
    try:
        tokenizer = get_tokenizer(cfg.base_model)
    except Exception:  # noqa: BLE001 -- no network / no cached tokenizer
        pytest.skip("tokenizer not available offline")
    renderer = renderers.get_renderer(cfg.renderer, tokenizer)
    r = render("Is 2 + 2 = 4?", renderer)
    tokens = r.prompt.to_ints()
    assert len(tokens) == r.choice_pos + 1
    assert tokenizer.decode(tokens[-3:]).endswith("?")
    assert set(r.label_tokens) == {"yes", "no"}
    # Same answer placeholder handling as the cookbook decision renderer.
    ref = DecisionRenderer(renderer, YesNoChoiceRenderer(renderer)).render(Decision("q", [("yes", ""), ("no", "")]))
    assert ref.label_tokens == r.label_tokens


TASK = {
    "task_id": "t1",
    "passage": "Revenue was $7.4M in 2025. Headcount reached 120 staff. Margin was 12.5% for the year.",
    "question": "Summarize revenue, headcount and margin.",
    "answer": "Revenue was $7.4M. Headcount was 120 staff. Margin was 12.5%.",
    "facts": [
        {"id": "f1", "value": "$7.4M", "criterion": "State revenue: $7.4M.", "statement": "Revenue was $7.4M."},
        {"id": "f2", "value": "120", "criterion": "State headcount: 120.", "statement": "Headcount was 120."},
        {"id": "f3", "value": "12.5%", "criterion": "State margin: 12.5%.", "statement": "Margin was 12.5%."},
    ],
}


def test_synth_edits_and_labels():
    import random

    from gdpsolve.verifier.synth import alter, edits, rubric_rows, valid

    assert valid(TASK)
    assert alter("$7.4M") == "$8.4M" and alter("0.95") == "0.15"
    e = edits(TASK["answer"], "120")
    assert "120" not in e["swap_number"] and "120" not in e["drop_fact"] and "either 120 or 220" in e["hedge"]
    rows = rubric_rows(TASK, random.Random(0))
    assert len(rows) == 5 * 3  # correct, pad, 3 edits x 3 criteria
    for r in rows:
        assert r["label"] == (0 if r["criterion_id"] == f"t1:{r['broken']}" else 1)
    assert all(r["label"] == 1 for r in rows if r["edit"] in ("correct", "pad"))


def test_synth_support_rows(tmp_path):
    from gdpsolve.schemas import SupportRequest
    from gdpsolve.verifier import store
    from gdpsolve.verifier.synth import support_rows

    text = support_rows(TASK, images=False)
    assert [r["label"] for r in text].count(1) == 3 and len(text) == 9
    imgs = support_rows(TASK, images=True, put=lambda d, m: store.put(d, m, root=tmp_path))
    assert any(r["kind"] == "blank_image" for r in imgs)
    req = SupportRequest.model_validate(imgs[0]["request"])
    assert req.sources[0].region == "r1" and store.get(req.regions[0].image, root=tmp_path)[:4] == b"\x89PNG"


def test_drop_keeps_other_facts_in_shared_sentence():
    from gdpsolve.verifier.synth import drop

    answer = "The facts are $4.2 million, 6.5%, and 18."
    out = drop(answer, "6.5%", ["$4.2 million", "18"])
    assert "6.5%" not in out and "$4.2 million" in out and "18" in out
    assert drop("A is 5. B is 7.", "5", ["7"]) == "B is 7."


def test_trace_writes_run_files(tmp_path):
    import json

    from gdpsolve import trace

    trace.call("verify", id="before-start")  # no-op until start()
    logger = trace.start(tmp_path, {"checker": "rubric"})
    trace.call("verify", id="c1", p_yes=0.9, prompt="Response: ...")
    logger.log_metrics({"brier_reward": 0.97})
    trace.stop(logger)
    trace.call("verify", id="after-stop")  # no-op again
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [c["id"] for c in calls] == ["c1"] and calls[0]["kind"] == "verify"
    assert json.loads((tmp_path / "config.json").read_text())["checker"] == "rubric"
    assert "brier_reward" in (tmp_path / "metrics.jsonl").read_text()
