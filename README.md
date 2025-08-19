# URC-SRO Research Prototype

This repository hosts the codebase for implementing **Unified Retrieval Controller (URC)** and **Self-Regulation Orchestrator (SRO)** modules in a modern RAG pipeline.
The goal is a controllable, auditable system that retrieves *just enough* evidence and validates reasoning before answering.

---

## 📌 Project Goal

* Improve retrieval timing & relevance
* Enhance reasoning quality via self-critique
* Reduce compute costs while maintaining accuracy

---

## 🛠 Tech Stack

| **Component**       | **Stack**                                                   |
| ------------------- | ----------------------------------------------------------- |
| **Language Models** | DeepSeek-R1 (or any HF model) via Hugging Face Transformers |
| **Retrieval**       | FAISS vector search (LlamaIndex optional)                   |
| **Adapters**        | LoRA (PEFT) for lightweight fine-tuning                     |
| **RL Training**     | PPO/DPO via TRL library                                     |

---

## 📂 Folder Structure

```
URC-SRO/
├── urc_sro/                      # Main Python package
│   ├── types.py                  # Data models (Document, Metadata, etc.)
│   ├── config.py                 # Pydantic settings
│   ├── pipeline.py               # High-level coordinator (URC → SRO → answer)
│   ├── factory.py                # Helpers to assemble mock or GPU pipelines
│   ├── llm_interfaces.py         # Protocol for LLM adapters
│   ├── urc/                      # Unified Retrieval Controller
│   │   ├── complexity.py         # QueryComplexityEstimator
│   │   ├── router.py             # SourceRouter
│   │   ├── policy.py             # Cost-Aware Retrieval Policy
│   │   ├── retriever.py          # BaseRetriever + UnifiedRetrievalController
│   │   ├── in_memory.py          # InMemoryRetriever (mock)
│   │   ├── embeddings.py         # Sentence-Transformers wrapper
│   │   ├── faiss_retriever.py    # FAISSRetriever + save/load helpers
│   │   ├── rerank.py             # Re-rankers (stub & cross-encoder)
│   │   └── allocator.py          # Token-budgeted ContextAllocator
│   ├── sro/                      # Self-Regulation Orchestrator
│   │   ├── orchestrator.py       # Iterative generate → verify → retry loop
│   │   ├── step_verify.py        # Simple verifier stub
│   │   ├── nli_step_verifier.py  # NLI-based verifier (e.g., DeBERTa-v3)
│   │   ├── evidence_monitor.py   # Heuristic support/contradiction scoring
│   │   ├── conflict_resolver.py  # (planned) Resolve conflicting evidence
│   │   ├── citation_auditor.py   # (planned) Citation precision/recall
│   │   ├── confidence_gate.py    # (planned) Abstain thresholding
│   │   └── coverage_auditor.py   # (planned) Completeness checks
│   ├── adapters/                 # LLM adapters (mock + HF)
│   │   ├── hf_llm.py             # HFGenerativeLLM + TrivialLLM
│   │   └── __init__.py
│   ├── io/                       # Corpus loaders
│   │   ├── __init__.py
│   │   └── loaders.py
│   └── eval/                     # Evaluation adapters
│       ├── __init__.py
│       └── ragas_adapter.py      # RAGAS harness (faithfulness, relevancy)
├── README.md
└── requirements.txt
```

---

## ⚡ 30-Second Quickstart

```bash
pip install -r requirements.txt

# Mock-only demo (no heavy models)
python -m urc_sro.examples.mock_entry

# GPU demo with FAISS
python -m urc_sro.examples.ingest_corpus --input_dir ./data --output_dir ./faiss_store
python -m urc_sro.examples.gpu_entry
```

---

## 🧭 The Journey (Plain Words)

| **Step** | **Who** | **What happens**                                                                                                    |
| -------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| 1️⃣      | **URC** | Estimates query difficulty → selects sources → decides how many docs to fetch.                                      |
| 2️⃣      | **URC** | Retrieves candidates → (optionally) re-ranks → packs them into a token-budgeted context.                            |
| 3️⃣      | **SRO** | Drafts an answer (plus brief reasoning steps) grounded in the retrieved context.                                    |
| 4️⃣      | **SRO** | Verifies a key step via NLI; if support is weak, loops back to **URC** to fetch more evidence (bounded iterations). |
| 5️⃣      | **SRO** | Returns a grounded answer—or politely abstains if evidence remains insufficient.                                    |

---

## ✅ What Works Today

* **Retrievers:** In-memory (keyword) and FAISS (vector)
* **Re-Ranker:** Optional cross-encoder
* **LLMs:** TrivialLLM (mock) and a Hugging Face wrapper (real)
* **Verification:** NLI-based step verifier
* **Evaluation:** RAGAS harness (faithfulness & relevancy)

---

## 🔮 Next Steps

| **Phase**       | **Deliverable**                                 |
| --------------- | ----------------------------------------------- |
| Learned Routing | Replace heuristics with trained policies        |
| Audits          | Conflict resolution & citation precision/recall |
| Fine-Tuning     | LoRA adapters + PPO/DPO via TRL                 |

---

## 📚 References

* [DeepSeek-R1](https://huggingface.co/deepseek-ai)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers)
* [FAISS](https://github.com/facebookresearch/faiss)
* [LlamaIndex](https://docs.llamaindex.ai/)
* [PEFT LoRA](https://huggingface.co/docs/peft)
* [TRL PPO/DPO](https://huggingface.co/docs/trl)
