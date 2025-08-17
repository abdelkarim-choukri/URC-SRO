from __future__ import annotations
from typing import List, Tuple
from .retriever import BaseRetriever
from ..types import Document

class InMemoryRetriever(BaseRetriever):
    """
    Minimal, dependency-free retriever backed by an in-memory list of Documents.

    Scoring: simple token-overlap heuristic between query and doc text.
    This mirrors common "in-memory" patterns used for prototyping retrievers,
    keeping things fast and portable without vector DBs. 
    """

    def __init__(self, docs: List[Document]) -> None:
        self._docs = docs

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        q_tokens = _tokenize(query)
        scored: List[Tuple[float, Document]] = []
        for d in self._docs:
            score = _overlap(q_tokens, _tokenize(d.text))
            # optional: slight bump if a source score exists
            if d.meta.score is not None:
                score += 0.01 * float(d.meta.score)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:max(0, top_k)]]

def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]

def _overlap(a: List[str], b: List[str]) -> float:
    if not a or not b: 
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    # Jaccard-like score to avoid biasing toward longer docs
    return inter / float(len(A | B))
