from typing import List, Dict, Tuple
from ..types import Document
from .complexity import QueryComplexityEstimator
from .router import SourceRouter
from .policy import CostAwareRetrievalPolicy
from .rerank import ReRanker

class BaseRetriever:
    """Pluggable retriever interface (vector/lexical/API)."""
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        raise NotImplementedError

class UnifiedRetrievalController:
    """Stages S1–S3: choose sources, fetch, early filter."""

    def __init__(
        self,
        sources: Dict[str, BaseRetriever],
        complexity: QueryComplexityEstimator,
        router: SourceRouter,
        policy: CostAwareRetrievalPolicy,
        reranker: ReRanker | None = None,
    ):
        self.sources = sources
        self.complexity = complexity
        self.router = router
        self.policy = policy
        self.reranker = reranker

    def retrieve(self, query: str) -> List[Document]:
        """Default non-debug path used by RAGPipeline."""
        docs, _dbg = self.retrieve_with_debug(query)
        return docs

    def retrieve_with_debug(self, query: str) -> Tuple[List[Document], Dict]:
        """Return results plus a small dict with pre/post re-rank IDs and control signals."""
        c = self.complexity.estimate(query)
        k = self.policy.decide(c)
        chosen = self.router.select_sources(query, c)

        pre: List[Document] = []
        for name in chosen:
            retr = self.sources.get(name)
            if retr:
                pre.extend(retr.retrieve(query, k))

        pre_ids = [d.id for d in pre]
        post = pre
        if self.reranker:
            post = self.reranker.rank(query, pre, k)
        post_ids = [d.id for d in post]

        debug = {
            "complexity": c,
            "k": k,
            "sources": chosen,
            "pre_rerank_ids": pre_ids,
            "post_rerank_ids": post_ids,
        }
        return post, debug
