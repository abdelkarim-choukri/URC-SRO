from typing import List
import tiktoken
from ..types import Document

class ContextAllocator:
    """S4-G1 — Context Packing & Token-Budget Allocation (basic)."""

    def __init__(self, model_encoding: str = "cl100k_base", max_tokens: int = 2048):
        self.enc = tiktoken.get_encoding(model_encoding)
        self.max_tokens = max_tokens

    def _count(self, text: str) -> int:
        return len(self.enc.encode(text))

    def pack(self, docs: List[Document]) -> str:
        """Simple greedy packing by current score order."""
        out, budget = [], self.max_tokens
        for d in docs:
            take = d.text.strip()
            cost = self._count(take)
            if cost <= budget:
                out.append(f"[{d.id}] {take}")
                budget -= cost
            else:
                continue
        return "\n\n".join(out)
