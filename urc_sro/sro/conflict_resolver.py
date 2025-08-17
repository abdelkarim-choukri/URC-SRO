from typing import List
from ..types import Document

class ConflictResolver:
    """S5-G4 — Handling conflicting evidence (stub)."""
    def resolve(self, query: str, evidence: List[Document]) -> List[Document]:
        # TODO: stance/NLI + recency trust
        return evidence
