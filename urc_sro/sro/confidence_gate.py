class ConfidenceGate:
    """S5-G3 — Final calibrated abstention (stub)."""
    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def allow(self, confidence: float) -> bool:
        return confidence >= self.threshold
