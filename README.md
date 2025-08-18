URC-SRO Research Prototype

This repository hosts a research-oriented codebase for implementing a Unified Retrieval Controller (URC) and Self-Regulation Orchestrator (SRO) in a modern Retrieval-Augmented Generation (RAG) pipeline. The prototype demonstrates how an LLM-powered QA system can dynamically control its retrieval process and self-critique its answers for better accuracy and efficiency.

📌 Project Goal

Design, implement, and evaluate URC and SRO modules that:

Improve retrieval timing & relevance: Dynamically decide when and what to retrieve to answer a query.

Enhance reasoning quality via self-critique: Verify the answer’s supporting evidence step-by-step and refine if necessary.

Reduce compute costs while maintaining accuracy: Avoid unnecessary LLM calls by smarter retrieval and answer validation steps.

🛠 Tech Stack

Language Models: DeepSeek-R1 (conceptual), with a Hugging Face Transformers backend (default model: mistralai/Mistral-7B-Instruct-v0.3).

Retrieval: FAISS (vector search) and simple in-memory search for prototyping.

Embedding Models: SentenceTransformers (default: intfloat/e5-small-v2 for embedding passages and queries).

Adapters: LoRA (planned for lightweight fine-tuning; not yet integrated in this prototype).

Self-Reflection & Verification: Uses NLI (natural language inference) models (e.g., DeBERTa-v3 for step verification) and heuristic scoring for evidence support.

RL Training (Planned): PPO/DPO via TRL for refining policies (future work, not in this prototype).

📂 Folder Structure
URC-SRO/
├── urc_sro/              # Main Python package for URC & SRO code
│   ├── types.py          # Data models (Document, Metadata, EvidenceSpan, etc.)
│   ├── config.py         # Global configuration using pydantic (Settings)
│   ├── pipeline.py       # High-level RAGPipeline coordinating URC and SRO
│   ├── factory.py        # Helper to assemble a sample pipeline (mock or real)
│   ├── llm_interfaces.py # Protocol definitions for LLM behavior
│   ├── urc/              # **Unified Retrieval Controller** components
│   │   ├── complexity.py      # QueryComplexityEstimator (query difficulty heuristic)
│   │   ├── router.py          # SourceRouter (selects which sources to query)
│   │   ├── policy.py          # CostAwareRetrievalPolicy (decides `k` docs per source)
│   │   ├── retriever.py       # BaseRetriever interface and UnifiedRetrievalController logic
│   │   ├── in_memory.py       # InMemoryRetriever (simple token overlap search)
│   │   ├── embeddings.py      # SBERTEmbedder (sentence-transformer for vector embedding)
│   │   ├── faiss_retriever.py # FAISSRetriever (vector search using FAISS index)
│   │   ├── rerank.py          # ReRanker classes (stub and cross-encoder reranker)
│   │   └── allocator.py       # ContextAllocator (packs retrieved docs into token budget)
│   ├── sro/              # **Self-Regulation Orchestrator** components
│   │   ├── orchestrator.py        # SelfRegulationOrchestrator (coordinates generation & verification)
│   │   ├── step_verify.py         # StepVerifier (stub for step-by-step verification)
│   │   ├── nli_step_verifier.py   # NLIStepVerifier (uses an NLI model for step verification)
│   │   ├── evidence_monitor.py    # EvidenceMonitor (scores answer support/contradiction)
│   │   ├── conflict_resolver.py   # ConflictResolver (stub for handling conflicting evidence)
│   │   ├── citation_auditor.py    # CitationAuditor (stub for checking citation precision/recall)
│   │   ├── confidence_gate.py     # ConfidenceGate (stub for deciding to output or abstain)
│   │   └── coverage_auditor.py    # CoverageAuditor (stub for assessing answer completeness)
│   ├── adapters/          # Adapters for integrating with external models/LLMs
│   │   ├── hf_llm.py           # HFGenerativeLLM (wraps a HuggingFace causal LM for answer generation)
│   │   └── __init__.py         # Exposes adapter classes (e.g., TrivialLLM alias, if defined)
│   └── examples/          # Example scripts to run the pipeline
│       ├── mock_entry.py       # Runs a full pipeline with in-memory retrieval & trivial LLM (no heavy dependencies)
│       └── gpu_entry.py        # Runs the pipeline with FAISS and a HF model (requires transformers, torch, etc.)
├── README.md             # Project documentation (you are reading this)
└── requirements.txt      # (If provided, would list required packages – not auto-generated in this prototype)

🚀 Getting Started

Prerequisites: Ensure you have Python 3.10+ and install the necessary libraries. The core requirements include:

pip install pydantic transformers sentence-transformers faiss-cpu torch


(You may use faiss-gpu instead of faiss-cpu if you have a CUDA-capable system and want GPU acceleration for vector search.)

Once dependencies are installed, you can try out the pipeline using the example scripts:

Quick Mock Demo (CPU-only, no heavy model):
Run the mock pipeline with an in-memory retriever and a trivial LLM logic:

python urc_sro/examples/mock_entry.py


This will use a small built-in document set and a simple retrieval + reasoning loop. It prints debug info about retrieval (pre/post re-rank document IDs, etc.), then outputs a final answer.

Full Pipeline Demo (FAISS + HF model):
If you have the resources and want to see the pipeline with a real embedding model and LLM:

python urc_sro/adapters/gpu_entry.py


This will: build a FAISS index of sample docs, use a SentenceTransformer model (e5-small-v2) for embeddings, and run a HuggingFace generative model (by default a 7B parameter model) to answer the question. It also uses an NLI-based verifier (NLIStepVerifier) to check the final reasoning step’s support. Debug information and the final answer will be printed to the console. (Ensure you have a GPU or adjust the model/device if running on CPU for this demo.)

Both demos use the query: “What do URC and SRO each do in this pipeline?” as an example, and demonstrate how the system retrieves relevant info and formulates an answer.

⚙️ How It Works (Module Overview)

Unified Retrieval Controller (URC) – this module manages the retrieval phase (when and what to retrieve):

Query Complexity Estimation: First, the QueryComplexityEstimator (urc/complexity.py) assigns a complexity score (0.0 to 1.0) to the query based on heuristics (e.g. query length). This score influences how aggressive retrieval should be.

Source Routing: The SourceRouter (urc/router.py) picks which data sources to query. In this prototype, it’s a stub that simply selects all available sources (or the top N if many).

Retrieval Policy: The CostAwareRetrievalPolicy (urc/policy.py) decides how many documents to retrieve from each source, k, based on the query complexity and a max cap (configurable, default 5). More complex queries may warrant retrieving more docs.

Retrieval Execution: The UnifiedRetrievalController (urc/retriever.py) brings it all together. It uses the chosen sources and calls their respective retrievers to gather up to k documents each:

InMemoryRetriever (urc/in_memory.py) – a simple built-in retriever that scores documents by keyword overlap (used in the mock demo).

FAISSRetriever (urc/faiss_retriever.py) – a vector similarity retriever using FAISS (used in the GPU demo, with SBERT embeddings from urc/embeddings.py).

(You can extend BaseRetriever to integrate other backends like web search or API calls.)

Re-Ranking: Optionally, a ReRanker (urc/rerank.py) can reorder the retrieved documents. By default, the pipeline uses a trivial ReRanker that just truncates the list to top-k. We also include a CrossEncoderReRanker implementation that uses a cross-encoder model (e.g., BAAI/bge-reranker-base) to score (query, doc) pairs for more precise ranking. (This is not enabled by default in the demos but can be swapped in.)

Context Allocation: The retrieved docs can be concatenated for the LLM. The ContextAllocator (urc/allocator.py) is responsible for packing the top documents into a prompt context under a token limit (e.g., 2048 tokens). It greedily adds docs until the budget is full. (In the current pipeline, the HFGenerativeLLM handles simple context formatting internally, but this allocator is intended for future use to finely control token usage.)

URC’s output is a set of top relevant documents (context) for the query, along with some debug info (e.g., which sources were used, pre- vs post-rerank IDs, etc.).

Self-Regulation Orchestrator (SRO) – this module manages the reasoning + self-check phase after initial retrieval:

LLM Answer Generation: The SelfRegulationOrchestrator (sro/orchestrator.py) takes the query and retrieved context and calls the language model to draft an answer with reasoning steps. Our LLM interface (LLM Protocol in llm_interfaces.py) defines generate_answer_with_steps, which should return both an answer and a list of reasoning steps the LLM took. In our prototype:

The Trivial/HF LLM (adapters/hf_llm.py) simply returns a final answer and a static list of placeholder steps (e.g., “Identify relevant passages”, “Compose answer”, “Validate claims”). In a real system, the LLM could be prompted to produce a chain-of-thought or reasoning log.

Step Verification: The last reasoning step (essentially the answer justification) is then checked by the StepVerifier (sro/step_verify.py). This component evaluates if the step is supported by the retrieved evidence. Currently, the StepVerifier stub just ensures both the step text and evidence list are non-empty (a placeholder for real logic). However, we have an advanced verifier implemented:

NLIStepVerifier (sro/nli_step_verifier.py), which uses a pretrained NLI model (DeBERTa-v3-base-MNLI) to determine if the evidence entails the claim in the step. It returns a StepSupport object indicating whether the step is supported (entailed=True/False), and can provide an evidence span from the most supportive document.

(The GPU demo uses NLIStepVerifier instead of the trivial verifier to illustrate this verification.)

Self-Reflection & Refinement: If the step verification finds the last step not entailed by the evidence (meaning the answer may have an unsupported claim), the SRO triggers a self-reflection: the LLM is asked to refine its answer. In our orchestrator, if sup.entailed is False, we call llm.self_refine_answer(...) with feedback (e.g., “Last step unsupported; refine.”). The LLM then amends the answer (in our dummy implementation, it appends a note that it was refined). This mimics how the system could iteratively improve an answer when a reasoning check fails.

Evidence Monitoring: After getting a (possibly refined) answer, the EvidenceMonitor (sro/evidence_monitor.py) assesses the overall answer for support. It computes simple scores for support vs. contradiction by comparing the answer’s claims to the evidence. In this prototype, it returns a fixed support score (0.6 if there is any evidence, else 0.0) and 0.0 for contradiction as a stub.

Final Answer or Abstain: The SRO uses the monitor’s scores to decide if the answer is sufficiently supported. The needs_retry() method checks if support is below a threshold or if any contradiction is detected. If so, instead of returning a potentially incorrect answer, the orchestrator will output: “I don’t have sufficient grounded evidence to answer precisely.” (an abstention indicating uncertainty). If the scores are acceptable, the answer is returned as-is.

Additional Audits (Planned): We have placeholder classes for further answer quality checks that are not yet wired into the pipeline:

ConflictResolver (sro/conflict_resolver.py) – would handle conflicting evidence sources by reconciling differences or choosing the most credible source.

CitationAuditor (sro/citation_auditor.py) – would ensure that all factual claims in the answer are backed by citations, and that provided citations are relevant (precision and recall of supporting facts).

ConfidenceGate (sro/confidence_gate.py) – a final threshold check on the answer’s confidence. This could use the LLM’s own uncertainty or an external calibration to decide whether to present the answer or abstain.

CoverageAuditor (sro/coverage_auditor.py) – would gauge if the answer fully addresses the query (no important aspect missed) using completeness metrics or follow-up question generation.

Together, the URC and SRO form a loop: the URC retrieves supporting info, the LLM generates an answer, the SRO verifies it. In future iterations, the SRO could decide to ask the URC for additional retrieval if needed (e.g., if support is lacking, go fetch more documents and then continue the reasoning). This prototype is a first step toward that kind of closed-loop RAG system.

🔮 Roadmap

This is an early prototype. The following enhancements are planned or could be explored next:

Iterative Retrieval & Multi-step Reasoning: Extend the SelfRegulationOrchestrator to perform multiple reasoning rounds. For example, if the EvidenceMonitor flags low support, the orchestrator could invoke the URC again to retrieve more data for a second attempt at answering, up to a max_iterations limit.

Learning-Based Components: Replace heuristics with learned models:

Train the QueryComplexityEstimator on a dataset of queries vs. optimal k to better predict complexity.

Implement a smarter SourceRouter that uses a classifier or policy to pick sources based on query features (e.g., route finance questions to a finance knowledge base, etc.).

Learn a retrieval policy that considers latency and past success (reinforcement learning to balance speed vs. thoroughness).

Enhanced Verification: Integrate the ConflictResolver, CitationAuditor, etc., into the answer finalization step. For instance, use a second LLM pass or rule-based system to resolve conflicts in evidence, and attach source attributions to each sentence of the answer.

Confidence Calibration: Utilize the ConfidenceGate with a calibrated model score or an ensemble of signals (e.g., NLI confidence, retrieval coverage) to decide when the system should abstain from answering.

Adapters and Integration: Add adapters for other models and tools:

Plug in a browser or API retriever to fetch live information.

Use LoRA to fine-tune the LLM (as hinted in Tech Stack) on domain-specific data for better in-context reasoning.

Evaluation: Rigorous evaluation on benchmark datasets for QA, measuring how the URC-SRO pipeline improves factual accuracy and reduces hallucinations compared to a baseline RAG system.

We welcome feedback and contributions. This project is a sandbox for trying out ideas in controllable retrieval and self-reflective reasoning with LLMs. By iterating on these components, we aim to inch closer to RAG systems that know when they don't know – and either find the needed information or admit uncertainty, leading to more trustworthy AI assistants.