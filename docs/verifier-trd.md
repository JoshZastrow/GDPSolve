# Verifier spec: untrained `verify()`, `support()`, `reward()`

Recipe and contract: [`verifier.md`](verifier.md). M1 baseline that uses `verify()`: [`m1-baseline.md`](m1-baseline.md).

## Context

The RL loop needs a reward for each finished rollout. The official GDP.pdf judge is already an LLM
with a rubric. A trained verifier would only be a distilled copy of it, so **training is deferred**
until a measured gap justifies it.

We start with an **untrained Tinker model** answering a one-token yes/no decision per criterion. That
gives a calibrated-looking P(pass) that is cheap, deterministic, and runs on Tinker only. This spec
covers building it and proving it can score rollouts and produce a usable reward.

## Requirements

| # | Requirement |
|---|---|
| F1 | `verify(req, cfg) -> VerifyResult`: one P(pass) per criterion from yes/no token logprobs |
| F2 | `support(req, cfg) -> SupportResult`: one P(supported) per claim; text spans or region images |
| F3 | `check_field(field, document) -> FieldCheck`: rule-based grounding for `extracted` and `computed` fields |
| F4 | `reward(v, grounding, beta=1.0) -> float | None`: `mean_criteria + beta·all_pass + mean(grounding)`; None when `all_pass` is None |
| F5 | `gdpsolve score verifier --checker rubric|support [--checkpoint …]`: request JSON on stdin, result JSON on stdout |
| F6 | `gdpsolve synth tasks` and `gdpsolve eval verifier --data synthetic`: a small synthetic set with labels by construction, and one number (Brier reward) |
| N1 | `checkpoint=None` everywhere (untrained). Every result records `CheckerConfig.model_ref`. |
| N2 | Deterministic: probabilities come from logprobs, with no sampling |
| N3 | Unscored (`None`) on an API failure or input over the context budget, never a silent 0 |
| N4 | GDP.pdf is not used to write or tune prompts. It is only scored in M1. |
| N5 | Offline tests. Tinker is called only from CLI runs. |

## Models (`schemas.CHECKERS`, unchanged)

| Checker | Model | Renderer |
|---|---|---|
| rubric | `Qwen/Qwen3.8-27B` | `qwen3_8_disable_thinking` |
| support | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | `nemotron3_ultra_disable_thinking` |

## Implementation

Package `src/gdpsolve/verifier/`:

| Module | Contents |
|---|---|
| `decision.py` | Image-capable decision path, forked from `tinker_cookbook.recipes.decisions` (see below) |
| `checkers.py` | `verify()`, `support()`, prompt builders, and a per-`CheckerConfig` cache of the renderer and sampling client |
| `grounding.py` | `check_field()` and a restricted arithmetic evaluator (`ast`: numbers, `+ - * /`, unary minus, `sum`; no names, calls, or attributes) |
| `reward.py` | `reward()` |
| `synth.py` | Synthetic tasks and constructed responses for F6 |
| `store.py` | Content-addressed object store under `var/objects/`: `put(bytes, media_type) -> ObjectRef` and `get(ref) -> bytes` |

### `decision.py`

The cookbook flattens prompts to token ids (`decision.py:38,200`), which drops image chunks. The fork
keeps a `tinker.ModelInput`:

- `render(messages, renderer, choice_renderer) -> Rendering(prompt, choice_pos, label_tokens)`
  1. Build the conversation with `renderer.build_supervised_example`. Pass `effort=0.0` only for `TmlV0Renderer`.
  2. Find the answer-format tokens in the last text chunk.
  3. Cut the prompt just after the placeholder. This is the method proven in the image spike.
- `sample(client, rendering) -> dict[str, float]`
  - One `sample_async` call with `max_tokens=1`.
  - Sparse `target_prompt_logprobs` of shape `[prompt.length - 1, 2]`, with the label tokens at row `choice_pos - 1`.
  - Softmax over the two labels.
- Reused unchanged: `YesNoChoiceRenderer` (`decisions/forecasting.py:68`, `choice#Y`/`choice#N`), and
  `get_renderer(name, tokenizer, image_processor=…)`. The image processor is passed only when the model `is_vl` and is not Inkling.

### Prompts

- **Rubric:**
  ```
  Response:\n{response}\n\nCriterion:\n{criterion.text}\n\nDoes the response perfectly satisfy the criterion?
  ```
- **Support:**
  - Each cited span is written as `[{id}, page {page}] {text}`.
  - A span with `region` set becomes an image part (loaded with `store.get`) followed by `[{id}, page {page}]`.
  - Then: `\n\nClaim:\n{claim}\n\nIs the claim fully supported by the evidence?`

### `verify()` / `support()`

1. Look up or create `(renderer, client)` for `cfg`, using `create_sampling_client_async(model_path=cfg.checkpoint)`
   or `base_model=cfg.base_model`.
2. Render one decision per criterion or claim, and `asyncio.gather` the `sample` calls.
3. Map exceptions and over-budget prompts to `None`.
4. Return `VerifyResult(scores, checkpoint=cfg.model_ref)`, which derives `mean_criteria` and `all_pass`,
   or `SupportResult`.

### Synthetic check set (`synth.py`)

Purpose: prove the untrained checkers can tell right from wrong. This set is for evaluation only, not training.

1. **Generate 40 tasks** with Inkling-Small across the 10 TRD domains. Each task is JSON:
   `{domain, passage, question, facts: [{id, criterion, value}], answer}`. Drop any task where a `value`
   is missing verbatim from the passage or the answer.
2. **Build constructed responses** from the correct answer:
   - `swap_number`, `drop_fact`, `hedge` each target one fact: that criterion is `no`, the others are `yes`.
   - `pad` (irrelevant text) leaves every label unchanged.
3. **Build support claims:**
   - A fact paired with its passage sentence is `yes`.
   - A swapped number, or the fact paired with another fact's sentence, is `no`.
   - Half the claims render their evidence sentence to PNG with PIL (`method="vlm"`, `region` set).
     Image negatives are the wrong region and a blank image.
4. **Write the outputs:** `data/synth/tasks.jsonl` and `data/synth/{rubric,support}.jsonl` (gitignored).

### Acceptance (untrained)

| Check | Pass |
|---|---|
| `eval verifier --checker rubric --data synthetic` | Brier reward ≥ 0.80 |
| `eval verifier --checker support --data synthetic` | Brier reward ≥ 0.80 |
| Reward ordering | `reward(correct) > reward(degraded)` on ≥ 90% of (task, edit) pairs |
| `pad` invariance | mean \|Δ mean_criteria\| < 0.05 |

If a checker misses its bar, that is the measured gap that justifies training (deferred spec).

### Results (2026-11-02, untrained)

Synthetic set: 26 valid tasks out of 40 generated (`gdpsolve synth tasks --n 40`). Generation uses
Inkling-Small with reasoning off (`effort=0.0`); with reasoning on, it used up `max_tokens` before writing JSON.

| Check | Result | Bar |
|---|---|---|
| Rubric Brier reward (Qwen3.8-27B, 520 decisions) | **0.977** | ≥ 0.80 |
| Support Brier reward (Nemotron-3.5-Super-VL, 325 decisions, half with image evidence) | **0.994** | ≥ 0.80 |
| Reward ordering (correct > degraded) | **100%** | ≥ 90% |
| `pad` invariance (mean \|Δ mean_criteria\|) | **0.014** | < 0.05 |
| Unscored | 0 | — |

The untrained checkers can verify and rank rollouts on this set. The set is easy: short passages,
clean edits, explicit criteria. How well `verify()` matches the official judge on real GDP-style
criteria comes from M1 ([`m1-baseline.md`](m1-baseline.md)). Spend: about $0.40.

## Cost

Tinker prices per 1M tokens: Qwen prefill $0.70; Nemotron prefill $0.57; Inkling-Small sample $1.44.

| Step | Est. |
|---|---|
| 40 synthetic tasks (Inkling-Small) | $0.10 |
| Rubric eval: about 40 × 5 responses × 5 criteria = 1,000 decisions × 300 tokens | $0.20 |
| Support eval: about 600 decisions × 400 tokens | $0.15 |

Every Tinker command prints an estimate and aborts above `--budget` (default $1).

## Tests (`tests/test_verifier.py`, offline)

- **Evaluator:** arithmetic, parentheses for negatives, "in thousands", and rejection of names, calls, and attributes.
- **`check_field`:** extracted and computed, both passing and failing.
- **`reward()`:** value and the None path.
- **Synth:** the edits and the labels they produce.
- **Fork rendering** against the real Qwen tokenizer (skipped without `tinker_cookbook` or offline):
  - The placeholder sits at `choice_pos`.
  - The yes/no label tokens match the cookbook `DecisionRenderer`. The system prompt differs, so the
    full token sequence does not.

## Out of scope

- Training, soft labels, comparison models
- Real scans, charts, and floor plans
- Hookup to the RL trainer (`reward()` is its interface)
