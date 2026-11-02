"""Data contract for the pipeline (docs/architecture.md).

Public:   ExtractionRequest -> process() -> ExtractionResult
Internal: Document, Job, Report, ObjectRef (what the router, queue, and workers pass around)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr

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
    page: int
    bbox: BoundingBox | None = None
    text: str  # verbatim evidence from the document
    method: Literal["text_layer", "ocr", "vlm"]


class ExtractedField(BaseModel):
    name: str
    value: object
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SourceSpan] = Field(default_factory=list)


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


class Job(BaseModel):
    """One unit of worker work. Written only by the router."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    request_id: str
    document: Document
    instructions: str
    pages: list[int] | None = None  # 1-indexed pages to focus on; None means the whole document
    max_steps: int = 40  # tool-call budget
    deadline_at: datetime


class Report(BaseModel):
    """What a worker publishes back to the router. Written only by workers.

    A job's state is its latest report: `progress` keeps it running, `ready` and `failed` end it.
    """

    job_id: str
    state: Literal["progress", "ready", "failed"]
    summary: str  # short status line for the router
    answer: str | None = None
    fields: list[ExtractedField] = Field(default_factory=list)
    artifacts: list[ObjectRef] = Field(default_factory=list)  # images/OCR the worker produced
    transcript: ObjectRef | None = None  # NDJSON of model turns and tool calls (RL training data)
    error: str | None = None
