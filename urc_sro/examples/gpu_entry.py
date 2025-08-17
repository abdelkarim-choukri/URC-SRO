from __future__ import annotations

from urc_sro import (
    RAGPipeline, Settings,
    QueryComplexityEstimator, SourceRouter, CostAwareRetrievalPolicy,
    UnifiedRetrievalController,
)
from urc_sro.types import Document, Metadata
from urc_sro.urc.embeddings import SBERTEmbedder
from urc_sro.urc.faiss_retriever import FAISSRetriever
from urc_sro.urc.rerank import ReRanker  # swap to CrossEncoderReRanker later
from urc_sro.adapters.hf_llm import HFGenerativeLLM
from urc_sro.sro.nli_step_verifier import NLIStepVerifier
from urc_sro.sro import EvidenceMonitor, SelfRegulationOrchestrator

def make_docs():
    return [
        Document(id="d1", text="URC selects sources and fetches passages using a cost-aware policy.", meta=Metadata(source="kb")),
        Document(id="d2", text="SRO verifies each reasoning step and can refine drafts before finalizing.", meta=Metadata(source="kb")),
        Document(id="d3", text="Latency vs. recall is balanced by iterating retrieval only when evidence is insufficient.", meta=Metadata(source="kb")),
    ]

def build_pipeline():
    settings = Settings()
    docs = make_docs()

    # URC
    embedder = SBERTEmbedder()
    retriever = FAISSRetriever(embedder, docs)
    retriever.build_index()

    urc = UnifiedRetrievalController(
        sources={"kb": retriever},
        complexity=QueryComplexityEstimator(),
        router=SourceRouter(["kb"]),
        policy=CostAwareRetrievalPolicy(max_docs=settings.urc_max_docs),
        reranker=ReRanker(),  # upgrade to CrossEncoderReRanker(...) later
    )

    # SRO
    llm = HFGenerativeLLM()
    verifier = NLIStepVerifier()
    monitor = EvidenceMonitor()
    sro = SelfRegulationOrchestrator(llm=llm, verifier=verifier, monitor=monitor)

    return RAGPipeline(urc=urc, sro=sro, settings=settings)

# Example usage (when you choose to run later):
# pipeline = build_pipeline()
# print(pipeline.run("What do URC and SRO do in this system?"))
