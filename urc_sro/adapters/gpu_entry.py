from __future__ import annotations
import json
from urc_sro import make_sample_docs
from urc_sro.config import Settings
from urc_sro.urc.embeddings import SBERTEmbedder
from urc_sro.urc.faiss_retriever import FAISSRetriever
from urc_sro.urc import (
    QueryComplexityEstimator,
    SourceRouter,
    CostAwareRetrievalPolicy,
    UnifiedRetrievalController,
)
from urc_sro.urc.rerank import ReRanker  # or CrossEncoderReRanker when you enable it
from urc_sro.sro.nli_step_verifier import NLIStepVerifier
from urc_sro.sro import EvidenceMonitor, SelfRegulationOrchestrator
from urc_sro.adapters.hf_llm import HFGenerativeLLM

def _load_cfg() -> dict:
    try:
        with open("examples/demo_config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"print_pre_post_rerank": True, "print_step_verifier_label": True, "print_citations": True}

def _inline_citations(answer: str, used_ids: list[str]) -> str:
    if not used_ids:
        return answer
    return f"{answer}  [sources: {', '.join(used_ids)}]"

def main():
    cfg = _load_cfg()
    settings = Settings()
    docs = make_sample_docs()

    # --- Build URC with FAISS (when you're ready to run) ---
    embedder = SBERTEmbedder()                # e5-small-v2 by default
    retriever = FAISSRetriever(embedder, docs)
    retriever.build_index()

    urc = UnifiedRetrievalController(
        sources={"kb": retriever},
        complexity=QueryComplexityEstimator(),
        router=SourceRouter(available_sources=["kb"]),
        policy=CostAwareRetrievalPolicy(max_docs=settings.urc_max_docs),
        reranker=ReRanker(),  # swap to CrossEncoderReRanker(...) later
    )

    # --- SRO with a real LLM + NLI verifier (when you're ready to run) ---
    llm = HFGenerativeLLM()                   # default Mistral-7B-Instruct wrapper
    verifier = NLIStepVerifier()              # DeBERTa-v3-base-MNLI
    monitor = EvidenceMonitor()
    sro = SelfRegulationOrchestrator(llm=llm, verifier=verifier, monitor=monitor)

    query = "What do URC and SRO each do in this pipeline?"

    docs, dbg = urc.retrieve_with_debug(query)

    if cfg.get("print_pre_post_rerank", False):
        print("— URC DEBUG —")
        print("complexity:", round(dbg["complexity"], 3), "| k:", dbg["k"], "| sources:", dbg["sources"])
        print("pre-rerank top-k:", dbg["pre_rerank_ids"])
        print("post-rerank top-k:", dbg["post_rerank_ids"])
        print()

    answer = sro.generate_response(query, docs)

    if cfg.get("print_step_verifier_label", False):
        sup = sro.last_step_support
        label = "entailed" if (sup and sup.entailed) else "not-entailed"
        print("— SRO DEBUG —")
        print("last-step label:", label)
        print()

    if cfg.get("print_citations", False):
        used_ids = dbg["post_rerank_ids"][:2]
        answer = _inline_citations(answer, used_ids)

    print("\n=== Final Answer ===\n")
    print(answer)

if __name__ == "__main__":
    main()
