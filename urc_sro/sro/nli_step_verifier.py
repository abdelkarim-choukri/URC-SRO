from __future__ import annotations
from typing import List, Dict
from ..types import Document, StepSupport, EvidenceSpan

class NLIStepVerifier:
    """
    MNLI classifier wrapper (default: MoritzLaurer/DeBERTa-v3-base-mnli).
    Aggregates entailment over evidence docs and returns StepSupport.
    """

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli", device: int | str | None = None, threshold: float = 0.5) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        import torch

        self.threshold = threshold
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        # device: int index for CUDA, -1 for CPU
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self._pipe = TextClassificationPipeline(
            model=self._mdl,
            tokenizer=self._tok,
            task="text-classification",
            device=device,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )

    def _entailment_prob(self, premise: str, hypothesis: str) -> float:
        # Pair input via text/text_pair
        scores = self._pipe({"text": premise, "text_pair": hypothesis})[0]
        # labels are usually 'entailment', 'neutral', 'contradiction'
        lbl = {s["label"].lower(): float(s["score"]) for s in scores}
        return lbl.get("entailment", 0.0)

    def verify(self, step_text: str, evidence: List[Document]) -> StepSupport:
        if not step_text or not evidence:
            return StepSupport(entailed=False, spans=[],
                               notes="missing step_text or evidence")
        best_p = 0.0
        best_doc = None
        for d in evidence:
            p = self._entailment_prob(d.text, step_text)
            if p > best_p:
                best_p, best_doc = p, d

        entailed = best_p >= self.threshold
        spans: List[EvidenceSpan] = []
        if entailed and best_doc:
            snippet = best_doc.text[:128]
            spans.append(EvidenceSpan(doc_id=best_doc.id, start=0, end=len(snippet), text=snippet, meta=best_doc.meta))
        return StepSupport(entailed=entailed, spans=spans, notes=f"best_entailment={best_p:.3f}")
