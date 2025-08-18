from __future__ import annotations
from typing import List, Tuple, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy  # v0.1.x naming
# Columns required by these two metrics: question, contexts, answer (no ground_truths needed)

def build_eval_dataset(queries: List[str], urc, sro) -> Dataset:
    """
    Runs URC->SRO for each query and collects:
      - question: str
      - contexts: list[str] (texts of retrieved docs actually sent to SRO)
      - answer: str (final SRO answer)
    Returns a HF Dataset with required columns for RAGAS.
    """
    rows: Dict[str, List[Any]] = {"question": [], "contexts": [], "answer": []}
    for q in queries:
        # Retrieve with debug to get the exact contexts we would send to SRO
        if hasattr(urc, "retrieve_with_debug"):
            docs, _dbg = urc.retrieve_with_debug(q)
        else:
            docs = urc.retrieve(q)
        # SRO generates the final answer using the retrieved docs
        answer = sro.generate_response(q, docs)
        rows["question"].append(q)
        rows["contexts"].append([d.text for d in docs])
        rows["answer"].append(answer)
    return Dataset.from_dict(rows)

def run_ragas_eval(dataset: Dataset):
    """
    Evaluates a dataset with RAGAS over:
      - faithfulness (answer vs. contexts)
      - answer_relevancy (answer vs. question/contexts)
    Returns the EvaluationResult.
    """
    # Default metrics if not supplied (we pass explicitly for clarity)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        show_progress=False,
    )
    return result
