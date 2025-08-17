from typing import List, Dict

class SourceRouter:
    """S1-G1 — Blind Source Router (stub for learned routing)."""

    def __init__(self, available_sources: List[str]):
        self.available_sources = available_sources

    def select_sources(self, query: str, complexity: float) -> List[str]:
        """Choose sources for this query. For now: simple pass-through priority."""
        # TODO: plug in learned router (cost & reliability aware)
        return self.available_sources[:3] if len(self.available_sources) > 3 else self.available_sources
