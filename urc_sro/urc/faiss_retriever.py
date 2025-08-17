from __future__ import annotations
from typing import List
import numpy as np

from ..types import Document
from .retriever import BaseRetriever
from .embeddings import SBERTEmbedder

class FAISSRetriever(BaseRetriever):
    """
    FAISS-backed retriever (IndexFlatIP by default) over an in-memory corpus.
    Build once with build_index(); then retrieve(query, top_k).
    """

    def __init__(self, embedder: SBERTEmbedder, docs: List[Document], use_ip: bool = True) -> None:
        self.embedder = embedder
        self.docs = docs
        self.index = None
        self.vecs = None
        self.use_ip = use_ip

    def build_index(self) -> None:
        import faiss  # lazy import
        # Encode passages (normalized for IP search)
        self.vecs = self.embedder.encode_passages([d.text for d in self.docs])
        dim = int(self.vecs.shape[1])
        self.index = faiss.IndexFlatIP(dim) if self.use_ip else faiss.IndexFlatL2(dim)
        self.index.add(self.vecs)

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        if self.index is None:
            self.build_index()

        q = self.embedder.encode_query(query).reshape(1, -1).astype("float32")
        D, I = self.index.search(q, max(1, top_k))
        idxs = I[0].tolist()
        return [self.docs[i] for i in idxs if i >= 0]
