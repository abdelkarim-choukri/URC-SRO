from typing import Callable, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class ToolCall:
    id: int
    name: str
    args: Dict[str, Any]
    result_digest: str

class ToolManager:
    """S6-G3 — Tool invocation ledger (framework-agnostic)."""

    def __init__(self, tools: Dict[str, Callable[..., Any]]):
        self._tools = tools
        self._ledger: List[ToolCall] = []
        self._next_id = 1

    def invoke(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not registered")
        result = self._tools[name](**kwargs)
        digest = str(hash(str(result)))[-8:]
        self._ledger.append(ToolCall(self._next_id, name, kwargs, digest))
        self._next_id += 1
        return result

    def ledger(self) -> List[ToolCall]:
        return list(self._ledger)
