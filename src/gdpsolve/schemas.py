"""Public request/response contract shared by the SDK, router, workers, and eval (TRD section 5)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr


class ExtractionRequest(BaseModel):
    path: Path  # a file or a directory (directories are expanded into a batch)
    instructions: str
    output_schema: dict | None = None  # optional JSON Schema the payload must satisfy
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
    method: str  # "text_layer" | "ocr" | "vlm"


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
    job_id: str
    document: DocumentMetadata
    status: str  # "ok" | "partial" | "failed" | "password_required"
    answer: str | None = None  # free-text response to the instructions (what GDP.pdf grades)
    fields: list[ExtractedField] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    latency_s: float | None = None
    error: str | None = None
