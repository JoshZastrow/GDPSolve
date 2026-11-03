"""One-token yes/no decisions that may contain images (docs/verifier-trd.md).

Forked from tinker_cookbook.recipes.decisions, which flattens prompts to token ids and so drops
image chunks. Here the prompt stays a tinker.ModelInput, cut just after the answer placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinker
import torch
from tinker_cookbook.recipes.decisions.forecasting import (
    ANSWER_FORMAT,
    YesNoChoiceRenderer,
)
from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer

SYSTEM = (
    "Answer with only {fmt}, replacing ? with Y for yes or N for no.\n"
    "choice#Y = yes\nchoice#N = no"
)


@dataclass(frozen=True)
class Rendering:
    prompt: tinker.ModelInput  # ends at the placeholder token
    choice_pos: int  # absolute position of the placeholder
    label_tokens: dict[str, int]  # {"yes": id, "no": id}


def render(user_content: str | list, renderer: Renderer) -> Rendering:
    """Render a yes/no question (text, or a list of text/image parts) for scoring."""
    choices = YesNoChoiceRenderer(renderer).render_choices(["yes", "no"])
    conversation = [
        Message(role="system", content=SYSTEM.format(fmt=ANSWER_FORMAT)),
        Message(role="user", content=user_content),
        Message(role="assistant", content=ANSWER_FORMAT),
    ]
    if isinstance(renderer, TmlV0Renderer):
        model_input, _ = renderer.build_supervised_example(conversation, effort=0.0)
    else:
        model_input, _ = renderer.build_supervised_example(conversation)

    answer, n = choices.answer_tokens, len(choices.answer_tokens)
    offset, hit = 0, None
    for i, chunk in enumerate(model_input.chunks):
        if isinstance(chunk, tinker.EncodedTextChunk):
            tokens = list(chunk.tokens)
            for j in range(len(tokens) - n, -1, -1):
                if tokens[j : j + n] == answer:
                    hit = (i, j, offset)
                    break
        offset += chunk.length
    if hit is None:
        raise ValueError("answer format not found in the rendered conversation")
    i, j, chunk_offset = hit
    cut = j + choices.answer_position + 1
    last = tinker.EncodedTextChunk(tokens=list(model_input.chunks[i].tokens)[:cut])
    prompt = tinker.ModelInput(chunks=[*model_input.chunks[:i], last])
    return Rendering(prompt, chunk_offset + cut - 1, dict(choices.label_tokens))


async def sample(client: tinker.SamplingClient, r: Rendering) -> dict[str, float]:
    """P(yes), P(no) from the label-token logprobs at the placeholder. Deterministic."""
    labels = list(r.label_tokens)
    target = torch.full((r.prompt.length - 1, len(labels)), -1, dtype=torch.int64)
    target[r.choice_pos - 1] = torch.tensor([r.label_tokens[k] for k in labels])
    response = await client.sample_async(
        prompt=r.prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1),
        target_prompt_logprobs=tinker.TensorData.from_torch_sparse(target, pad_value=-1),
    )
    if response.target_prompt_logprobs is None:
        raise RuntimeError("no target_prompt_logprobs in the sampling response")
    logprobs = response.target_prompt_logprobs.to_torch()[r.choice_pos - 1].float()
    return dict(zip(labels, torch.softmax(logprobs, 0).tolist(), strict=True))
