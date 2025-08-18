from __future__ import annotations
from urc_sro.factory import make_mock_pipeline  # or wire your GPU pipeline here
from urc_sro.eval import build_eval_dataset, run_ragas_eval

def main():
    # 1) Build a pipeline (mock by default; swap to your GPU build if desired)
    pipeline = make_mock_pipeline()
    urc, sro = pipeline.urc, pipeline.sro

    # 2) Define a few demo questions (extend as needed)
    queries = [
        "What are the roles of URC and SRO?",
        "How does the pipeline reduce hallucinations?",
        "When does URC decide to fetch more documents?"
    ]

    # 3) Collect traces -> HF Dataset
    ds = build_eval_dataset(queries, urc, sro)

    # 4) Run RAGAS evaluation
    result = run_ragas_eval(ds)

    # 5) Print a compact summary
    print("\n=== RAGAS (faithfulness & answer_relevancy) ===")
    print(result)       # pretty summary per-metric
    print("\nAverages:", result.to_pandas().mean(numeric_only=True).to_dict())

if __name__ == "__main__":
    main()
