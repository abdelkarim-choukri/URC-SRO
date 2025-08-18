from __future__ import annotations
from typing import List, Dict, Optional

from .config import Settings
from .types import Document, Metadata
from .pipeline import RAGPipeline

# URC parts
from .urc import (
    InMemoryRetriever,
    QueryComplexityEstimator,
    SourceRouter,
    CostAwareRetrievalPolicy,
    UnifiedRetrievalController,
    ReRanker,
    ContextAllocator,  
)

# SRO parts
from .adapters import TrivialLLM
from .sro import (
    StepVerifier,
    EvidenceMonitor,
    SelfRegulationOrchestrator,
)

def make_sample_docs() -> List[Document]:
    """Tiny in-memory corpus used by the InMemoryRetriever."""
    return [
        Document(
            id="doc-1",
            text="Unified Retrieval Controller (URC) selects sources and fetches relevant passages. It uses a cost-aware policy.",
            meta=Metadata(source="kb", uri="kb://urc", score=0.9),
        ),
        Document(
            id="doc-2",
            text="Self-Regulation Orchestrator (SRO) verifies each reasoning step and can refine drafts before finalizing.",
            meta=Metadata(source="kb", uri="kb://sro", score=0.85),
        ),
        Document(
            id="doc-3",
            text="The pipeline balances latency and recall by iterating retrieval only when evidence is insufficient.",
            meta=Metadata(source="kb", uri="kb://policy", score=0.8),
        ),
    ]

def make_mock_pipeline(
    docs: Optional[List[Document]] = None,
    source_names: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
) -> RAGPipeline:
    """
    Assemble a fully wired pipeline using the in-memory retriever and a trivial LLM.
    Returns a ready-to-use RAGPipeline.
    """
    settings = settings or Settings()
    docs = docs or make_sample_docs()
    source_names = source_names or ["kb"]

    # --- URC wiring ---
    retriever = InMemoryRetriever(docs)
    sources_map: Dict[str, InMemoryRetriever] = {name: retriever for name in source_names}

    complexity = QueryComplexityEstimator()
    router = SourceRouter(available_sources=source_names)
    policy = CostAwareRetrievalPolicy(max_docs=settings.urc_max_docs)
    reranker = ReRanker()  # pass-through for now

    urc = UnifiedRetrievalController(
        sources=sources_map,
        complexity=complexity,
        router=router,
        policy=policy,
        reranker=reranker,
    )

    # --- SRO wiring ---
    llm = TrivialLLM()
    verifier = StepVerifier()
    monitor = EvidenceMonitor()
    allocator = ContextAllocator(max_tokens=settings.urc_context_max_tokens)  # <-- NEW
    sro = SelfRegulationOrchestrator(
        llm=llm,
        verifier=verifier,
        monitor=monitor,
        urc=urc,
        max_iterations=settings.sro_max_iterations,
        allocator=allocator,  # <-- NEW
    )


    # --- Pipeline ---
    return RAGPipeline(urc=urc, sro=sro, settings=settings)
