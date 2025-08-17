from typing import List
from .manager import ToolCall

class ProvenanceGuard:
    """S6-G3 — Require claims referencing tool outputs to map to ledger entries (stub)."""
    def __init__(self, ledger_provider):
        self._ledger_provider = ledger_provider

    def validate_text(self, text: str) -> bool:
        # TODO: parse <tool_call_id=N> markers and match to ledger
        return True

    def get_ledger(self) -> List[ToolCall]:
        return self._ledger_provider()
