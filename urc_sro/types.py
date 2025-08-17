from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime

class Metadata(BaseModel):
    source: Optional[str] = None         # e.g., "kb", "web", "api:docs"
    uri: Optional[str] = None            # canonical link or doc-id
    timestamp: Optional[datetime] = None # recency for temporal control
    score: Optional[float] = None        # retriever/reranker score
    extra: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    id: str
    text: str
    meta: Metadata = Field(default_factory=Metadata)

class EvidenceSpan(BaseModel):
    doc_id: str
    start: int
    end: int
    text: str
    meta: Metadata = Field(default_factory=Metadata)

class RetrievalParams(BaseModel):
    num_docs: int
    sources: List[str]

class StepSupport(BaseModel):
    entailed: bool
    spans: List[EvidenceSpan] = Field(default_factory=list)
    notes: Optional[str] = None

class AnswerCandidate(BaseModel):
    text: str
    citations: List[EvidenceSpan] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
