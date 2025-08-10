# URC-SRO Research Prototype

This repository hosts the codebase for implementing **Unified Retrieval Controller (URC)** and **Self-Reflection Optimizer (SRO)** modules in a modern RAG pipeline.

## 📌 Project Goal
To design, implement, and evaluate URC and SRO modules that:
- Improve retrieval timing & relevance
- Enhance reasoning quality via self-critique
- Reduce compute costs while maintaining accuracy

## 🛠 Tech Stack
- **Language Models:** DeepSeek-R1, compatible with HuggingFace Transformers
- **Retrieval:** FAISS / LlamaIndex
- **Adapters:** LoRA for lightweight fine-tuning
- **RL Training:** PPO/DPO (TRL library)

## 📂 Folder Structure
urc_sro_project/
├── docs/ # Design docs, diagrams
├── src/ # URC & SRO code
├── data/ # Datasets
├── tests/ # Unit tests
