# Verifier recipe (draft)

The RL reward for GDPSolve. It answers two questions about each rollout:

| Question | Checker | How |
|---|---|---|
| Is the answer right? | **Rubric verifier** | Learned decision model: P(response satisfies criterion) |
| Is the answer supported by the document? | **Grounding**: rule check + **support checker** | Rules for extracted and computed fields; learned decision model for summary claims |

Both learned checkers are built on the Tinker cookbook's decision-model recipe
(`tinker_cookbook.recipes.decisions`, [docs](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/decisions/)).
`decisions/forecasting.py` is the closest template: a yes/no decision trained with a Brier loss.
Everything runs through Tinker's API. Model weights may come from any vendor Tinker serves.

Schemas: `src/gdpsolve/schemas.py`, verifier section. Tests: `tests/test_smoke.py`.

## Rules

- **GDP.pdf is evaluation only.** No checker trains on GDP.pdf tasks, rubrics, responses, or
  judge verdicts. GDP.pdf is used only to measure how well the checkers agree with the official judge.
- **Synthetic tasks only for training** (TRD §8.1). Rubric and claim generators copy the GDP.pdf
  rubric *format* from hand-written exemplars, never GDP.pdf text. Criteria are written from the
  generator's known ground truth, not from a model reading the document. A wrong criterion trains
  the verifier to reward wrong answers.
- **Frozen during policy training.** A new checker version is a new Tinker checkpoint, adopted only between RL rounds.

## Models

| Role | Model (Tinker) | Inputs | Why |
|---|---|---|---|
| Rubric verifier | `Qwen/Qwen3.8-27B:peft:262144` | text | Same inputs as the official judge, which never sees images. Proven in the decisions recipe (AUC 0.81–0.83 on yes/no). A different family from the Inkling policy. |
| Support checker | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16:peft:262144`, renderer `nemotron3_ultra_disable_thinking` | text + region image | Built for document extraction, chart reading, and table parsing. 12B active parameters. A different family from the policy. |
| Labeling judge (training labels only) | `thinkingmachines/Inkling` with reasoning | text (+ image for claims) | The strongest model available. Runs offline, once per label. |

Planned comparison runs: `thinkingmachines/Inkling-Small` for both checkers.

**Image-decision check (2026-11-02).** One generated image showed "TOTAL 1,240" and another
"TOTAL 9,999". Each was given the same claim, "the image shows a total of 1,240", zero-shot,
with one-token yes/no logprobs:

| Model | P(yes), 1,240 image | P(yes), 9,999 image | Prompt tokens |
|---|---|---|---|
| Nemotron-3.5-Super-VL | 0.997 | 0.001 | 345 |
| Inkling-Small | 1.000 | 0.000 | 117 |

Tinker returns `target_prompt_logprobs` for prompts with image chunks, so image decisions work.
This proves the mechanism only, not accuracy on real crops. For the same input, Nemotron uses
about 3× the prompt tokens, from template overhead plus image encoding.

## 1. Rubric verifier

One decision per (response, criterion), with choices `yes` and `no`. It sees exactly what the
official judge sees: the response and one criterion's text. It never sees the task prompt, the PDF, or
criterion metadata (`surge-ai/gdp-pdf` `src/gdp_pdf/scorer.py`, commit `7a72a51`).

```python
Decision(
    text=f"Response:\n{req.response}\n\nCriterion:\n{c.text}\n\n"
         "Does the response perfectly satisfy the criterion?",
    choices=[("yes", "The response satisfies the criterion"),
             ("no", "The response does not satisfy the criterion")],
)
```

- Answer tokens: `YesNoChoiceRenderer` from `forecasting.py` (`choice#Y` / `choice#N`).
- The judge passes a criterion only on a "perfect" match, so the question wording follows that.
- "Dodged Bullet" criteria read "should avoid ...", so `yes` means the response avoided the trap.
- The verifier checks the answer against the rubric's answer key. It cannot tell whether a fact
  came from the document. Grounding (below) covers that.

## 2. Grounding

A field is grounded when every value it depends on traces to a cited span, and the step from
the spans to the value can be checked. `ExtractedField.kind` selects the check:

| Kind | Example | Cited | Check | Output |
|---|---|---|---|---|
| `extracted` | invoice total $48,210 | the span holding the value | Re-read the span (text layer or OCR); fuzzy-match text and number | `FieldCheck` |
| `computed` | YTD = Jan + … + Jun | each input span, plus unit and period context | Re-read inputs, parse numbers (signs, parentheses, "in thousands"), evaluate `formula` with a restricted arithmetic evaluator (numbers, `+ - * /`, `sum`; never `eval`), compare within tolerance | `FieldCheck` |
| `summary` | "Q2 margin fell because freight rose" | spans behind each `Claim` | Support checker | `SupportResult` |

- Reconciliation between sources, such as a monthly report against the accounting table,
  is a computed field: cite both sides, and the formula gives the gap.
- The rule check catches a wrong cell, wrong months, or wrong units. It does not catch the wrong
  method, such as an average where a sum was needed. The rubric verifier catches that, because the
  criterion states the expected number.
- The worker's `validate` tool runs the same recompute before it reports.

### Support checker

One decision per claim: the cited spans plus the claim, answered `yes` (supported) or `no`.

```python
Decision(
    text="Evidence:\n"
         + "\n".join(f"[{s.id}, page {s.page}] {s.text}" for s in cited)  # image part instead when s.region is set
         + f"\n\nClaim:\n{claim.text}\n\nIs the claim fully supported by the evidence?",
    choices=[("yes", "The evidence supports the claim"),
             ("no", "The evidence does not support the claim")],
)
```

- **Text spans:** the span text goes in the decision text.
- **Visual spans** (`SourceSpan.region` set: charts, floor-plan symbols, diagram segments, scanned
  table crops): the region image goes in the user message as an image part, followed by the claim text.
- The cookbook's `DecisionRenderer` flattens prompts to token ids (`ModelInput.to_ints()`), which
  fails on image chunks. Our fork keeps the `ModelInput`, finds the answer position with
  `ModelInput.length`, and sizes `target_prompt_logprobs` to match. The image-decision check above
  used this approach.
- **Cost control:** send an image only when the span is visual (`method == "vlm"` or a figure
  region). Otherwise send the OCR or text-layer text.

## 3. Data

```
synthetic task (doc, prompt, rubric, claims)   TRD §8.1 generators, ground truth known
  └─ responses from N policies                 diverse on purpose
       ├─ rubric decisions: one per criterion   label: P(pass) from the labeling judge, k = 4
       └─ support decisions: one per claim      label: generator truth (which fact is in which span)
```

| Item | Starting point |
|---|---|
| Tasks | 2,000 synthetic across 10 domains (M3) |
| Responses per task | 4: Inkling-Small baseline, Inkling, GDPSolve pipeline, one degraded |
| Rubric decisions | about 100k; train on about 20k first (the forecasting run used 3.6k) |
| Support decisions | positives from generator truth; negatives from edits: swapped number, reversed direction, unsupported cause, wrong region |

- **Degraded responses** are edits of a correct answer: drop a fact, swap a number, hedge, restate
  the criterion without evidence, pad with length. A checker trained only on baseline outputs will
  misjudge the policy once RL moves it, which is where reward hacking starts.
- **Refresh:** every R policy steps, label a sample of current rollouts and add them to the training set.
- **Splits** are by task (a task's criteria and claims stay together). One generator family,
  such as floor plans, is held out as an out-of-distribution test.

## 4. Training

Supervised, with one forward/backward pass per decision.

```python
renderer = DecisionRenderer(base_renderer, YesNoChoiceRenderer(base_renderer))
datums = decision_dist_datums(renderer, [(d, {"yes": p, "no": 1 - p}) for d, p in batch])
await training_client.forward_backward_custom_async(datums, probability_loss(brier_score_loss))
await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
```

| Setting | Starting point |
|---|---|
| Loss | Brier on soft labels. Ablation: cross-entropy on hard labels. |
| Batch / LR / epochs | 28 / 1e-4 / 1, as in the forecasting recipe |
| Eval cadence | every 16 steps |
| Class balance | Weight positives and negatives equally until baseline pass rates are known |

## 5. Interface and API contract

### Functions

```python
async def verify(req: VerifyRequest, cfg: CheckerConfig) -> VerifyResult
async def support(req: SupportRequest, cfg: CheckerConfig) -> SupportResult
def check_field(field: ExtractedField, document: Document) -> FieldCheck
```

These are in-process functions, not a service. Each learned checker builds its renderer and Tinker
sampling client from `cfg` and caches them per config.

### Choosing the model: `CheckerConfig`

| Field | Meaning |
|---|---|
| `base_model` | Tinker base model |
| `renderer` | cookbook renderer name, with reasoning off (one-token decisions) |
| `checkpoint` | `tinker://…` sampler path of a trained checker; `None` = zero-shot base model |
| `model_ref` (derived) | `checkpoint` or `base_model`. Written to every result's `checkpoint` field, so each score traces to an exact model. |

Defaults (`schemas.CHECKERS`; renderer names checked against `tinker_cookbook.model_info`):

| `--checker` | `base_model` | `renderer` |
|---|---|---|
| `rubric` | `Qwen/Qwen3.8-27B` | `qwen3_8_disable_thinking` |
| `support` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | `nemotron3_ultra_disable_thinking` |

### Running one example

`gdpsolve score verifier` reads a request as JSON on stdin and writes the result as JSON on stdout.

```bash
gdpsolve score verifier --checker rubric < request.json                                  # zero-shot
gdpsolve score verifier --checker rubric --checkpoint tinker://<run>/sampler_weights/000128 < request.json
gdpsolve score verifier --checker support < support_request.json
```

The same call in Python:

```python
cfg = CHECKERS["rubric"].model_copy(update={"checkpoint": "tinker://<run>/sampler_weights/000128"})
result = await verify(VerifyRequest.model_validate_json(open("request.json").read()), cfg)
```

Rubric input (`VerifyRequest`):

```json
{"response": "YTD revenue was $7.4M.",
 "criteria": [{"id": "c1", "text": "The response should state YTD revenue: $7.4M."}]}
```

Rubric output (`VerifyResult`; `mean_criteria` and `all_pass` are computed and included in the JSON):

```json
{"scores": [{"criterion_id": "c1", "p_pass": 0.97}],
 "checkpoint": "tinker://<run>/sampler_weights/000128",
 "mean_criteria": 0.97, "all_pass": 0.97}
```

Support input (`SupportRequest`) and output (`SupportResult`):

```json
{"claims": [{"id": "k1", "text": "Freight costs rose in Q2.", "source_ids": ["s7"]}],
 "sources": [{"id": "s7", "page": 5, "text": "freight cost bar rises Q1->Q2", "method": "vlm", "region": "r1"}],
 "regions": [{"id": "r1", "doc_sha256": "…", "page": 5, "dpi": 150,
              "image": {"sha256": "…", "media_type": "image/png", "size": 10240}}]}
```

```json
{"scores": {"k1": 0.94}, "checkpoint": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"}
```

Region images are read from the object store by `image.sha256`.

### Schemas

| Schema | Fields |
|---|---|
| `CheckerConfig` | `base_model`, `renderer`, `checkpoint`; derived `model_ref` |
| `Criterion` | `id` (matches the score back), `text` (what the model is asked). GDP.pdf's other criterion columns don't affect official scoring, so they stay in the dataset. |
| `VerifyRequest` | `response` (= `ExtractionResult.answer`), `criteria` |
| `VerifyResult` | `scores: list[CriterionScore]` (`criterion_id`, `p_pass` or None), `checkpoint`; derived `mean_criteria`, `all_pass` |
| `Claim` | `id`, `text`, `source_ids` |
| `SupportRequest` | `claims`, `sources: list[SourceSpan]`, `regions: list[Region]` (images for visual spans) |
| `SupportResult` | `scores: claim id -> P(supported)` or None, `checkpoint` |
| `FieldCheck` | `field`, `ok`, `reason` |
| `SourceSpan` (changed) | adds `id` (referenced by formulas and claims) and `region` (Region.id for visual evidence) |
| `ExtractedField` (changed) | adds `kind`, `formula`, `claims` |

- **One decision per criterion or claim,** all sent in parallel (`asyncio.gather` over
  `sample_decision`), so a rollout costs one round trip.
- **Deterministic:** probabilities come from yes/no token logprobs, not sampling, so the same input
  and checkpoint give the same score. Rewards can be replayed and audited.
- **Versioned:** every result carries `checkpoint`, and rollout transcripts store it.
- **Unscored handling matches the official scorer:** a failed call or an over-long input gives
  `p_pass=None`. That criterion leaves `mean_criteria`, and `all_pass` becomes None, so the trainer
  skips the rollout instead of counting it as a failure.

## 6. Reward

```
reward = mean_criteria + β·all_pass + mean(grounding)          (TRD §8.2)
grounding per field = FieldCheck.ok                     extracted / computed
                    = mean P(supported) over its claims  summary
```

- Low `p_pass` criteria double as the SDPO hint (TRD §8.3 hint 1). No judge rationale is needed.
- **Anti-hacking checks:**
  - Spot-check a fixed share of rollouts with the labeling judge.
  - If the checker and the judge disagree past a threshold, pause, relabel, and retrain.
  - Track response length against reward.

## 7. Evaluation

`gdpsolve eval verifier --checker rubric|support --data synthetic|gdp-pdf` prints one number.

**Metric: Brier reward** = 1 − mean((p − label)²) over every decision in the eval set. It runs
from 0 to 1, and higher is better.
- It is the forecasting recipe's metric, so the numbers compare directly.
- It penalizes both wrong and miscalibrated probabilities. The reward uses p itself (mean and
  product), so calibration matters.
- Both checkers are yes/no decisions, so they share the metric.

**One eval set per checker and dataset.** Robustness cases are ordinary labeled decisions mixed
into the set, not separate metrics, so failing them lowers the one number.

| Data | Rubric verifier set | Support checker set | Labels | When |
|---|---|---|---|---|
| `synthetic` | Held-out tasks (including one unseen generator family), plus degraded pairs (dropped fact, swapped number, hedge, restated criterion) labeled `no`, plus padded correct answers labeled with the original's label | Held-out claims, plus false claims (swapped number, reversed direction, unsupported cause), plus true claims shown the wrong region or a blank image, labeled `no` | Labeling judge (rubric); generator truth (support) | Every 16 training steps |
| `gdp-pdf` | Baseline responses on GDP.pdf | Not available. GDP.pdf has no claim labels. | Official Gemini 3.5 Flash verdicts, stored `eval_only` (TRD §8.5) | Before a checker version becomes the reward |

- **Reading the `gdp-pdf` number:** compare it with the labeling judge's Brier reward against the same
  Gemini verdicts. That is the ceiling, because the verifier is distilled from that judge.
- **The rule check** (`check_field`) is deterministic, so unit tests cover it instead of a metric.
- **During RL**, the same metric runs on the spot-checked rollouts (verifier p against labeling-judge
  verdicts). If it falls over policy steps, the policy is gaming the verifier: pause, relabel, retrain.

## 8. CLI (planned)

Commands are `gdpsolve <action> <resource>`.

| Command | Does |
|---|---|
| `gdpsolve score verifier --checker rubric\|support [--checkpoint …]` | Score one request: JSON on stdin, JSON on stdout (§5) |
| `gdpsolve label verifier` | Sample responses for synthetic tasks; label rubric decisions with the judge and support decisions from generator truth |
| `gdpsolve train verifier --checker rubric\|support` | Decision-model training on Tinker; saves a sampler checkpoint |
| `gdpsolve eval verifier --checker rubric\|support --data synthetic\|gdp-pdf` | Prints the Brier reward (§7) |

## Open questions

- **Labeling judge quality:** measure Inkling-judge vs. Gemini agreement on GDP.pdf baseline responses
  before training. The verifier can't be better than its labels.
- **Token cost:** measure Nemotron vs. Inkling-Small prompt tokens on full-page crops at 150 DPI, and check per-model Tinker pricing.
- **Long responses:** truncation policy, or mark them unscored, when response + criterion exceed the context budget.
