from typing import Optional

class QueryComplexityEstimator:
    """S1-G4 — Zero-Label Complexity Estimator (stub).
    Replace heuristics with a learned head in a later step."""
    def __init__(self) -> None:
        pass

    def estimate(self, query: str) -> float:
        """Return normalized complexity in [0,1]."""
        # placeholder heuristic: length + basic cues
        tokens = query.split()
        score = min(1.0, 0.1 + 0.02 * len(tokens))
        '''Imagine a slider:
            1-word query → 0.1 + 0.02 * 1 = 0.12
            10-word query → 0.1 + 0.20 = 0.30
            45-word query → 0.1 + 0.90 = 1.00 (ceiling hit)
            60-word query → still capped at 1.00'''
        return score
