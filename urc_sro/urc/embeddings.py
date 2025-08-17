from __future__ import annotations
from typing import List
import numpy as np

class SBERTEmbedder:
    """
    Sentence-Transformers embedder wrapper (default: intfloat/e5-small-v2).
    Produces L2-normalized float32 vectors suitable for FAISS IP search.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        # Lazy import to avoid heavy import at module load
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=self.device)

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        # e5 family expects "passage: " prefix
        if "e5" in self.model_name:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        if "e5" in self.model_name:
            query = f"query: {query}"
        return self._encode([query])[0]

    # Kept for convenience if you want raw encoding
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)
