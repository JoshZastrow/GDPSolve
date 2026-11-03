"""Async RL of Inkling-Small on doc-QA (docs/training.md): builds tinker_cookbook.rl.train.Config and calls main.

Tinker concepts in this file:
- Training client: the LoRA adapter on Tinker. Each step runs forward_backward on the batch's rollouts (loss:
  importance-sampled policy gradient with group-centered advantages) and then optim_step (Adam).
- Sampling client: a snapshot of the training weights that samples the rollouts; the trainer makes a new one
  after each optim_step.
- Async off-policy: rollouts for the next step are sampled while the current step trains. A group sampled by
  weights more than max_steps_off_policy steps old is requeued instead of trained on.
- Checkpoints: every save_every steps the trainer saves the training state (to resume) and the sampler weights
  (to serve), and appends their tinker:// paths to <log_path>/checkpoints.jsonl. Running again with the same
  log_path resumes from the last checkpoint.
"""

from __future__ import annotations

from tinker_cookbook import hyperparam_utils, model_info
from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator
from tinker_cookbook.rl.train import AsyncConfig, Config

from gdpsolve import docqa

MODEL = docqa.POLICY
REFERENCE = "Qwen/Qwen3-8B"  # a model get_lr has a calibrated LoRA formula for


def learning_rate(model: str = MODEL) -> float:
    """LoRA learning rate. get_lr has no Inkling formula, so take the calibrated LR of REFERENCE and scale it by
    get_lora_lr_multiplier (1/sqrt(parameter count)), the transfer rule hyperparam_utils documents."""
    return (hyperparam_utils.get_lr(REFERENCE) * hyperparam_utils.get_lora_lr_multiplier(model)
            / hyperparam_utils.get_lora_lr_multiplier(REFERENCE))


def dev_evaluator(renderer_name: str, max_examples: int = 32, max_tokens: int = 8192) -> BenchmarkEvaluator:
    """The dev benchmark as an inline evaluator, sampled as the service samples (temperature 1)."""
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = get_renderer(renderer_name, get_tokenizer(MODEL))
    evaluator = BenchmarkEvaluator(docqa.DocQABenchmark.name, renderer, max_examples=max_examples,
                                   max_tokens=max_tokens, temperature=1.0)
    evaluator.config.timeout_seconds = docqa.DocQABenchmark.recommended_timeout
    return evaluator


def config(log_path: str, lr: float, groups_per_batch: int = 12, group_size: int = 4, max_steps: int | None = None,
           max_tokens: int = 4096, every: int = 4) -> Config:
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    return Config(
        model_name=MODEL,
        renderer_name=renderer_name,
        recipe_name="gdpsolve_docqa",
        lora_rank=32,
        learning_rate=lr,
        max_tokens=max_tokens,
        temperature=1.0,
        dataset_builder=docqa.DocQADatasetBuilder(model_name=MODEL, renderer_name=renderer_name,
                                                  batch_size=groups_per_batch, group_size=group_size),
        async_config=AsyncConfig(max_steps_off_policy=1, groups_per_batch=groups_per_batch),
        evaluator_builders=[lambda: dev_evaluator(renderer_name)],
        eval_every=every,
        save_every=every,
        max_steps=max_steps,
        log_path=log_path,
    )


def run(cfg: Config) -> None:
    import asyncio

    from tinker_cookbook.rl.train import main

    asyncio.run(main(cfg))
