"""Data contract for the pipeline (docs/architecture.md).

Public:   ExtractionRequest -> process() -> ExtractionResult
Internal: Document, Job, Report, Message, ObjectRef, Region, ToolCall (what the router, queue, and workers pass around)
Toolbox:  TOOLBOX, the worker's action space (one args model per tool)
Verifier: VerifyRequest -> verify(req, CheckerConfig) -> VerifyResult;
          SupportRequest -> support(req, CheckerConfig) -> SupportResult; FieldCheck (docs/verifier.md)
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, computed_field

# --- public -------------------------------------------------------------------


class ExtractionRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)  # eval uses the GDP.pdf task_id
    path: Path  # one PDF; a directory is expanded by the caller into one request per file
    instructions: str
    output_schema: dict | None = None  # optional JSON Schema the fields must satisfy
    passwords: list[SecretStr] = Field(default_factory=list)  # tried in order for encrypted PDFs
    deadline_s: int = 300


class BoundingBox(BaseModel):
    page: int  # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float


class SourceSpan(BaseModel):
    id: str = "s1"  # referenced by ExtractedField.formula and Claim.source_ids
    page: int
    bbox: BoundingBox | None = None
    text: str  # verbatim evidence from the document
    method: Literal["text_layer", "ocr", "vlm"]
    region: str | None = None  # Region.id showing the evidence; the support checker sees its image


class ExtractedField(BaseModel):
    """One extracted value and how it is grounded (docs/verifier.md, Grounding).

    extracted - the value appears in a source span
    computed  - the value is `formula` over source span ids, e.g. "s1 + s2 + s3"
    summary   - the value is free text; each of `claims` must be supported by its sources
    """

    name: str
    value: object
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SourceSpan] = Field(default_factory=list)
    kind: Literal["extracted", "computed", "summary"] = "extracted"
    formula: str | None = None  # computed only: arithmetic over SourceSpan ids
    claims: list[Claim] = Field(default_factory=list)  # summary only


class DocumentMetadata(BaseModel):
    filename: str
    sha256: str
    page_count: int
    encrypted: bool
    has_text_layer: bool
    title: str | None = None
    author: str | None = None
    created: datetime | None = None
    producer: str | None = None


class ExtractionResult(BaseModel):
    request_id: str
    document: DocumentMetadata
    status: Literal["ok", "partial", "failed", "password_required"]
    answer: str | None = None  # free-text response to the instructions (what GDP.pdf grades)
    fields: list[ExtractedField] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    latency_s: float | None = None
    error: str | None = None


# --- internal -----------------------------------------------------------------


class ObjectRef(BaseModel):
    """Content-addressed blob in the object store: page images, OCR output, transcripts."""

    sha256: str
    media_type: str  # e.g. application/pdf, image/png, application/json, application/x-ndjson
    size: int


class Document(BaseModel):
    """An ingested PDF: metadata plus the decrypted bytes in the object store."""

    metadata: DocumentMetadata
    pdf: ObjectRef


class Region(BaseModel):
    """A view of (part of) a page. Image tools return regions and take region ids, so steps
    compose like pipes: render -> crop -> correct_image -> recognize.

    `id` is a hash of (doc_sha256, page, bbox, dpi, ops): the same view always has the same id.
    """

    id: str
    doc_sha256: str
    page: int  # 1-indexed
    bbox: BoundingBox | None = None  # PDF points; None means the full page
    dpi: int
    ops: list[str] = Field(default_factory=list)  # corrections applied, in order
    image: ObjectRef


class ToolCall(BaseModel):
    """One step of a trajectory, and the memo record for it.

    Tools are pure functions of (document, args), so a call is memoized under
    key = sha256(doc_sha256, tool, canonical args). Any job, request, or RL rollout that
    repeats a call reuses the stored result instead of recomputing it.
    """

    key: str
    tool: str
    args: dict
    result: ObjectRef  # JSON output (text, layout blocks, table HTML, validation errors)
    regions: list[Region] = Field(default_factory=list)  # regions the call produced
    cost_ms: int  # wall time when first computed; part of the RL cost signal
    cached: bool = False


class Job(BaseModel):
    """One unit of worker work. Written only by the router.

    A request starts as one root job. If its worker calls `split`, the router adds child
    `extract` jobs over page shards and one `merge` job that runs `after` them.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    request_id: str
    parent_id: str | None = None  # the job whose `split` or `handback` made this one
    kind: Literal["extract", "merge"] = "extract"
    document: Document
    instructions: str
    pages: list[int] | None = None  # 1-indexed shard; None means the whole document
    after: list[str] = Field(default_factory=list)  # job ids that must end before this is claimable
    max_steps: int = 40  # tool-call budget
    deadline_at: datetime


class Message(BaseModel):
    """One entry on a request's Thread. Append-only; written only by workers, through tools.

    dsh-superagent analog: a Trigger on a Thread. The router acts on `split` and `handback`
    (it is the only writer of jobs); the other kinds are context for other workers. New
    messages are pushed into each worker's next observation (dsh: an Input delivered at the
    next step). A page's disposition is derived from the Thread, never stored.
    """

    thread: str  # the request_id: one Thread per request
    seq: int = 0  # assigned on append; total order within the Thread
    job_id: str  # the worker's job that wrote it
    kind: Literal["split", "handback", "unrelated", "follow", "note"]
    pages: list[int] = Field(default_factory=list)
    note: str = ""  # reason, finding, or pointer, e.g. "table continues from p. 37"
    pages_per_job: int | None = None  # split only


# --- toolbox: the worker's action space ---------------------------------------
# No fixed order. The policy chooses which tool to call next; the JSON Schemas of these
# models are the tool specs the model sees. Keep the set small and stable: RL learns over it.


class InspectPdf(BaseModel):
    """Page facts: size, rotation, has text layer, image coverage (scanned?)."""

    page: int


class Render(BaseModel):
    """Rasterize a full page. Returns a Region."""

    page: int
    dpi: int = 150


class DetectLayout(BaseModel):
    """Typed blocks with bboxes: text, heading, table, figure, footnote, header, footer."""

    page: int


class Crop(BaseModel):
    """A region of a page, re-rendered at `dpi`. Returns a Region."""

    page: int
    bbox: BoundingBox
    padding: float = 8.0  # points added on each side
    dpi: int = 300


class Tile(BaseModel):
    """Overlapping tiles over a large page (drawings, dense tables). Returns Regions."""

    page: int
    tile_size: int = 1024  # pixels
    overlap: int = 128


class CorrectImage(BaseModel):
    """Apply one correction to a region. Returns a new Region."""

    region: str  # Region.id
    operation: Literal["rotate", "deskew", "perspective", "dewarp", "denoise", "contrast"]


class Recognize(BaseModel):
    """Run a recognizer (PaddleOCR-VL-1.6) on a region."""

    region: str  # Region.id
    mode: Literal["text", "handwriting", "table", "formula", "chart", "symbol"]


class ReadNativeText(BaseModel):
    """Text-layer words with bboxes; free and exact when the page has a text layer."""

    page: int
    bbox: BoundingBox | None = None


class Validate(BaseModel):
    """Check an output: JSON Schema, required fields, types, sums, date/unit normalization."""

    output: dict
    constraints: dict


class Split(BaseModel):
    """Ask the router to fan this job's pages out to child jobs plus a merge job.

    dsh analog: `handback` queues work for the main session to decide; workers never create
    work directly. The router may cap or reject the split. This job should then report.
    """

    pages_per_job: int = 10
    pages: list[int] | None = None  # subset of this job's pages; None means all of them


class Post(BaseModel):
    """Append a Message to the request's Thread.

    kind (dsh analog):
      handback  - give pages back for the router to reassign (`handback`)
      unrelated - pages are irrelevant to the question; others may skip them (`markUnrelated`)
      follow    - pages elsewhere are related, e.g. a continued table (`followForContext`)
      note      - a finding other workers or the merge job should know (`note`)
    """

    kind: Literal["handback", "unrelated", "follow", "note"]
    pages: list[int] = Field(default_factory=list)
    note: str


class ReportArgs(BaseModel):
    """Publish to the router. `ready` or `failed` ends the job."""

    state: Literal["progress", "ready", "failed"]
    summary: str  # short status line for the router
    answer: str | None = None  # extract: notes for the merge job; merge: the final answer
    fields: list[ExtractedField] = Field(default_factory=list)  # each carries page-grounded sources
    pages_read: list[int] = Field(default_factory=list)  # pages actually examined (coverage)


TOOLBOX: dict[str, type[BaseModel]] = {
    "inspect_pdf": InspectPdf,
    "render": Render,
    "detect_layout": DetectLayout,
    "crop": Crop,
    "tile": Tile,
    "correct_image": CorrectImage,
    "recognize": Recognize,
    "read_native_text": ReadNativeText,
    "validate": Validate,
    "split": Split,
    "post": Post,
    "report": ReportArgs,
}


class Report(ReportArgs):
    """What a worker publishes back to the router. Written only by workers.

    A job's state is its latest report: `progress` keeps it running, `ready` and `failed` end it.
    """

    job_id: str
    artifacts: list[ObjectRef] = Field(default_factory=list)  # regions/outputs the worker produced
    transcript: ObjectRef | None = None  # NDJSON of model turns and ToolCalls (the RL rollout)
    error: str | None = None


# --- verifier: scoring functions for RL (docs/verifier.md) ----------------------
# Rubric verifier: does the response satisfy a criterion? Same inputs as the official GDP.pdf judge.
# Support checker: do the cited spans (text or image) support a claim? Rule check: extracted/computed fields.


class CheckerConfig(BaseModel):
    """Which model scores the decisions. verify()/support() build the renderer and Tinker
    sampling client from this."""

    base_model: str
    renderer: str  # tinker_cookbook renderer name; reasoning off for one-token decisions
    checkpoint: str | None = None  # tinker:// sampler path; None = zero-shot base model

    @property
    def model_ref(self) -> str:
        """What produced a score: the checkpoint, or the base model when zero-shot."""
        return self.checkpoint or self.base_model


CHECKERS: dict[str, CheckerConfig] = {
    "rubric": CheckerConfig(base_model="Qwen/Qwen3.8-27B", renderer="qwen3_8_disable_thinking"),
    "support": CheckerConfig(
        base_model="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        renderer="nemotron3_ultra_disable_thinking",
    ),
}


class Criterion(BaseModel):
    """One rubric criterion. The verifier model sees only `text`."""

    id: str
    text: str
    type: Literal["Primary Intent", "Dodged Bullet"]
    severity: Literal["Certain dealbreaker", "Possible dealbreaker", "Unlikely dealbreaker"] | None = None


class VerifyRequest(BaseModel):
    response: str  # ExtractionResult.answer
    criteria: list[Criterion]


class CriterionScore(BaseModel):
    criterion_id: str
    p_pass: float | None = Field(default=None, ge=0.0, le=1.0)  # None = unscored (call failed or input too long)


class VerifyResult(BaseModel):
    """Aggregates follow the official scorer: unscored criteria leave `mean_criteria`, and any
    unscored criterion makes `all_pass` None (the rollout is skipped, not counted as a failure)."""

    scores: list[CriterionScore]
    checkpoint: str  # CheckerConfig.model_ref that produced the scores

    @computed_field
    @property
    def mean_criteria(self) -> float | None:
        ps = [s.p_pass for s in self.scores if s.p_pass is not None]
        return sum(ps) / len(ps) if ps else None

    @computed_field
    @property
    def all_pass(self) -> float | None:
        if not self.scores or any(s.p_pass is None for s in self.scores):
            return None
        return math.prod(s.p_pass for s in self.scores)


class Claim(BaseModel):
    """One atomic claim of a summary field, backed by source spans."""

    id: str
    text: str
    source_ids: list[str]  # SourceSpan ids


class SupportRequest(BaseModel):
    claims: list[Claim]
    sources: list[SourceSpan]
    regions: list[Region] = Field(default_factory=list)  # images for sources with `region` set


class SupportResult(BaseModel):
    scores: dict[str, float | None]  # claim id -> P(supported); None = unscored
    checkpoint: str  # CheckerConfig.model_ref


class FieldCheck(BaseModel):
    """Rule-based grounding of an extracted or computed field."""

    field: str  # ExtractedField.name
    ok: bool
    reason: str = ""  # e.g. "s3 reads 1,204 not 1,240", "unit mismatch"
