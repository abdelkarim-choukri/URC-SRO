from typing import List
from .types import Document
from .urc import UnifiedRetrievalController
from .sro import SelfRegulationOrchestrator
from .config import Settings

class RAGPipeline:
    """End-to-end coordination of URC (retrieval) and SRO (reasoning/regulation)."""

    def __init__(self, urc: UnifiedRetrievalController, sro: SelfRegulationOrchestrator, settings: Settings):
        self.urc = urc
        self.sro = sro
        self.settings = settings

    def run(self, query: str) -> str:
        """Query → retrieve context (URC) → reason & verify (SRO) → finalize."""
        initial_docs: List[Document] = self.urc.retrieve(query)
        return self.sro.generate_response(query, initial_docs)
