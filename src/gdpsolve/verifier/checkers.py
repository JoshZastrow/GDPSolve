"""verify() and support(): one yes/no decision per criterion or claim (docs/verifier.md §5)."""

from __future__ import annotations

import asyncio
import io

import tinker
from PIL import Image
from tinker_cookbook import model_info, renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.tokenizer_utils import get_tokenizer

from gdpsolve import trace
from gdpsolve.schemas import (
    CheckerConfig,
    Criterion,
    CriterionScore,
    SourceSpan,
    SupportRequest,
    SupportResult,
    VerifyRequest,
    VerifyResult,
)
from gdpsolve.verifier import store
from gdpsolve.verifier.decision import Rendering, render, sample

MAX_PROMPT_TOKENS = 60_000  # longer decisions are unscored, not truncated

_service: tinker.ServiceClient | None = None
_cache: dict[tuple, tuple[renderers.Renderer, tinker.SamplingClient]] = {}
_lock = asyncio.Lock()


async def _model(cfg: CheckerConfig, vision: bool) -> tuple[renderers.Renderer, tinker.SamplingClient]:
    """Renderer and sampling client for a config, created once per process."""
    global _service
    key = (cfg.base_model, cfg.renderer, cfg.checkpoint, vision)
    async with _lock:
        if key not in _cache:
            processor = get_image_processor(cfg.base_model) if vision else None
            renderer = renderers.get_renderer(cfg.renderer, get_tokenizer(cfg.base_model), image_processor=processor)
            _service = _service or tinker.ServiceClient()
            if cfg.checkpoint:
                client = await _service.create_sampling_client_async(model_path=cfg.checkpoint)
            else:
                client = await _service.create_sampling_client_async(base_model=cfg.base_model)
            _cache[key] = (renderer, client)
    return _cache[key]


async def _p_yes(client: tinker.SamplingClient, r: Rendering, renderer: renderers.Renderer, kind: str,
                 checkpoint: str, item_id: str) -> float | None:
    p, error = None, None
    if r.prompt.length > MAX_PROMPT_TOKENS:
        error = f"prompt {r.prompt.length} tokens > {MAX_PROMPT_TOKENS}"
    else:
        try:
            p = (await sample(client, r))["yes"]
        except Exception as e:  # noqa: BLE001 -- any API failure leaves the decision unscored, like the official scorer
            error = f"{type(e).__name__}: {e}"
    trace.call(kind, checkpoint=checkpoint, id=item_id, p_yes=p, error=error,
               prompt_tokens=r.prompt.length, prompt=trace.prompt_text(renderer.tokenizer, r.prompt))
    return p


def rubric_question(response: str, criterion: Criterion) -> str:
    return (
        f"Response:\n{response}\n\nCriterion:\n{criterion.text}\n\n"
        "Does the response perfectly satisfy the criterion?"
    )


async def verify(req: VerifyRequest, cfg: CheckerConfig) -> VerifyResult:
    renderer, client = await _model(cfg, vision=False)
    renderings = [render(rubric_question(req.response, c), renderer) for c in req.criteria]
    ps = await asyncio.gather(
        *(_p_yes(client, r, renderer, "verify", cfg.model_ref, c.id) for r, c in zip(renderings, req.criteria, strict=True))
    )
    scores = [CriterionScore(criterion_id=c.id, p_pass=p) for c, p in zip(req.criteria, ps, strict=True)]
    return VerifyResult(scores=scores, checkpoint=cfg.model_ref)


def support_content(claim: str, spans: list[SourceSpan], images: dict[str, Image.Image]) -> str | list:
    """Evidence lines (or image parts for spans with a region), then the claim."""
    parts: list[dict] = [{"type": "text", "text": "Evidence:\n"}]
    for s in spans:
        if s.region in images:
            parts += [{"type": "image", "image": images[s.region]}, {"type": "text", "text": f"[{s.id}, page {s.page}]\n"}]
        else:
            parts.append({"type": "text", "text": f"[{s.id}, page {s.page}] {s.text}\n"})
    parts.append({"type": "text", "text": f"\nClaim:\n{claim}\n\nIs the claim fully supported by the evidence?"})
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


async def support(req: SupportRequest, cfg: CheckerConfig) -> SupportResult:
    vision = model_info.get_model_attributes(cfg.base_model).is_vl and bool(req.regions)
    renderer, client = await _model(cfg, vision=vision)
    images = {r.id: Image.open(io.BytesIO(store.get(r.image))).convert("RGB") for r in req.regions}
    spans = {s.id: s for s in req.sources}
    renderings = [
        render(support_content(c.text, [spans[i] for i in c.source_ids if i in spans], images), renderer)
        for c in req.claims
    ]
    ps = await asyncio.gather(
        *(_p_yes(client, r, renderer, "support", cfg.model_ref, c.id) for r, c in zip(renderings, req.claims, strict=True))
    )
    return SupportResult(scores={c.id: p for c, p in zip(req.claims, ps, strict=True)}, checkpoint=cfg.model_ref)
