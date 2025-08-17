from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional, Dict

class Settings(BaseSettings):
    """Global configuration for URC-SRO. Extend in future steps."""
    # URC
    urc_max_docs: int = Field(default=5, description="Default cap per source before re-ranking")
    urc_context_max_tokens: int = Field(default=2048, description="Pack budget for context allocator")

    # SRO
    sro_confidence_threshold: float = Field(default=0.85, description="Final answer confidence gate")
    sro_max_iterations: int = Field(default=3, description="Upper bound on reason→verify cycles")

    # Sources registry (logical names → connection strings)
    sources: Dict[str, str] = Field(default_factory=dict)

    class Config:
        env_prefix = "URC_SRO_"
        extra = "ignore"
