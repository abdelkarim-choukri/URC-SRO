class CostAwareRetrievalPolicy:
    """S1-G3 — Cost-Aware Retrieval Policy (stub)."""

    def __init__(self, max_docs: int = 5):
        self.max_docs = max_docs

    def decide(self, complexity: float) -> int:
        """Scale #docs by complexity."""
        return max(1, min(self.max_docs, int(round(3 + 2 * complexity))))
