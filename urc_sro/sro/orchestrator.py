from typing import List
from ..types import Document
from ..llm_interfaces import LLM
from .step_verify import StepVerifier
from .evidence_monitor import EvidenceMonitor
from ..urc.allocator import ContextAllocator  # token-budgeted packing

class SelfRegulationOrchestrator:
    """Drives stepwise generation + verification + corrective retrieval."""

    def __init__(self, llm: LLM, verifier: StepVerifier, monitor: EvidenceMonitor, urc=None, max_iterations: int = 3, allocator: ContextAllocator | None = None):
        self.llm = llm
        self.verifier = verifier
        self.monitor = monitor
        self.urc = urc  # UnifiedRetrievalController (optional)
        self.max_iterations = max_iterations
        self.last_step_support = None  # set after verify()
        self.allocator = allocator or ContextAllocator()

    def _extract_claims(self, text: str) -> List[str]:
        return [s.strip() for s in text.split(".") if s.strip()]

    def _pack_for_llm(self, docs: List[Document]) -> List[Document]:
        packed = self.allocator.pack(docs)
        return [Document(id="packed-context", text=packed)]

    def _formulate_followup(self, query: str, draft: str, steps: List[str]) -> str:
        focus = (steps[-1] if steps else "").strip()
        if focus:
            return f"{query} — clarify: {focus}"
        claims = self._extract_claims(draft)
        return f"{query} — provide more evidence for: {claims[-1]}" if claims else query

    def _merge_docs(self, base: List[Document], new: List[Document]) -> List[Document]:
        seen = {d.id for d in base}
        merged = list(base)
        for d in new:
            if d.id not in seen:
                merged.append(d)
                seen.add(d.id)
        return merged

    def generate_response(self, query: str, context: List[Document]) -> str:
        evidence = list(context)
        for _ in range(max(1, self.max_iterations)):
            # Token-budgeted packed context for generation
            packed_for_llm = self._pack_for_llm(evidence)
            draft, steps = self.llm.generate_answer_with_steps(query, packed_for_llm)

            # Verify last reasoning step if present
            if steps:
                sup = self.verifier.verify(steps[-1], evidence)
                self.last_step_support = sup
                if not sup.entailed:
                    draft = self.llm.self_refine_answer(draft, "Last step unsupported; refine.")

            # Evidence sufficiency check
            scores = self.monitor.score(self._extract_claims(draft), evidence)
            if not self.monitor.needs_retry(scores):
                return draft

            # Retry path: fetch more evidence via URC if available
            if not self.urc:
                break
            followup = self._formulate_followup(query, draft, steps)
            if hasattr(self.urc, "retrieve_with_debug"):
                new_docs, _ = self.urc.retrieve_with_debug(followup)
            else:
                new_docs = self.urc.retrieve(followup)
            evidence = self._merge_docs(evidence, new_docs)

        return "I don’t have sufficient grounded evidence to answer precisely."
