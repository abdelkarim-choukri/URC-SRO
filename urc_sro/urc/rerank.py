from __future__ import annotations
from typing import List, Iterable
from ..types import Document

class ReRanker:
    """Simple passthrough (kept as a no-op stub)."""
    def rank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        return docs[:top_k]

class CrossEncoderReRanker(ReRanker):
    """
    Cross-encoder reranker (default: BAAI/bge-reranker-base).
    Scores (query, passage) pairs and returns the top_k docs by score.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder  # lazy import
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device, trust_remote_code=True)

    def rank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []
        pairs = [[query, d.text] for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
        scored = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:max(1, top_k)]]
