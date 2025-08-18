from __future__ import annotations
import argparse, os
from urc_sro.io import load_corpus_from_dir
from urc_sro.urc.embeddings import SBERTEmbedder
from urc_sro.urc.faiss_retriever import FAISSRetriever

def main():
    ap = argparse.ArgumentParser(description="Ingest a corpus and build a FAISS index (IndexFlatIP).")
    ap.add_argument("--input_dir", required=True, help="Directory containing .txt and/or .jsonl files")
    ap.add_argument("--output_dir", required=True, help="Directory to write index.faiss and docs.json")
    ap.add_argument("--model", default="intfloat/e5-small-v2", help="Sentence-Transformers model for embeddings")
    args = ap.parse_args()

    docs = load_corpus_from_dir(args.input_dir)
    if not docs:
        raise SystemExit(f"No documents found under: {args.input_dir}")

    embedder = SBERTEmbedder(model_name=args.model)
    retriever = FAISSRetriever(embedder, docs, use_ip=True)
    retriever.build_index()
    os.makedirs(args.output_dir, exist_ok=True)
    retriever.save_local(args.output_dir)
    print(f"Saved FAISS index + docs to: {args.output_dir}")

if __name__ == "__main__":
    main()
