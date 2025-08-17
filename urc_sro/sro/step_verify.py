from typing import List
from ..types import Document, StepSupport, EvidenceSpan

class StepVerifier:
    """S6-G1 — Verify each reasoning step before continuing (stub)."""

    def verify(self, step_text: str, evidence: List[Document]) -> StepSupport:
        # TODO: plug in NLI / attribution checker
        ok = len(step_text.strip()) > 0 and len(evidence) > 0
        spans: List[EvidenceSpan] = []
        return StepSupport(entailed=ok, spans=spans, notes=None)
