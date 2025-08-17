from typing import List
from ..types import Document, EvidenceSpan

class CitationAuditor:
    """S5-G2 / S7-G1 / S7-G3 — Citation precision & recall of implicit support (stub)."""

    def audit(self, answer: str, evidence: List[Document]) -> List[EvidenceSpan]:
        # TODO: atomic-fact extraction + span alignment
        return []
