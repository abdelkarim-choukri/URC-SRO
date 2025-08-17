from typing import List, Dict
from ..types import Document

class EvidenceMonitor:
    """S2-G2 / S3-G2 — Retrieval-failure + faithfulness/contradiction signals (stub)."""

    def score(self, claims: List[str], evidence: List[Document]) -> Dict[str, float]:
        # TODO: real metrics (RAGAS/AIS)
        if not claims or not evidence:
            return {"support": 0.0, "contradiction": 0.0}
        return {"support": 0.6, "contradiction": 0.0}

    def needs_retry(self, scores: Dict[str, float]) -> bool:
        return scores.get("support", 0.0) < 0.5 or scores.get("contradiction", 0.0) > 0.0
