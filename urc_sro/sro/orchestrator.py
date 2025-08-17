from typing import List
from ..types import Document
from ..llm_interfaces import LLM
from .step_verify import StepVerifier
from .evidence_monitor import EvidenceMonitor

class SelfRegulationOrchestrator:
    """Drives stepwise generation + verification + corrective retrieval (stub)."""

    def __init__(self, llm: LLM, verifier: StepVerifier, monitor: EvidenceMonitor):
        self.llm = llm
        self.verifier = verifier
        self.monitor = monitor
        self.last_step_support = None  # set to StepSupport after verify()

    def _extract_claims(self, text: str) -> List[str]:
        return [s.strip() for s in text.split(".") if s.strip()]

    def generate_response(self, query: str, context: List[Document]) -> str:
        draft, steps = self.llm.generate_answer_with_steps(query, context)
        if steps:
            sup = self.verifier.verify(steps[-1], context)
            self.last_step_support = sup
            if not sup.entailed:
                draft = self.llm.self_refine_answer(draft, "Last step unsupported; refine.")
        scores = self.monitor.score(self._extract_claims(draft), context)
        if self.monitor.needs_retry(scores):
            return "I don’t have sufficient grounded evidence to answer precisely."
        return draft
