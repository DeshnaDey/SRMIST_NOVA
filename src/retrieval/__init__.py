"""Stage 3 - retrieval, fusion and reranking.  (owner: retrieval)

Only ``dense`` runs. The other three are gated off and kept as evidence.

``dense``   SHIPS - bi-encoder + FAISS semantic search
``bm25``    Gated out before building. The pre-build diagnostic put the oracle
            ceiling on any dense+BM25 fusion at +0.027 recall@100
            (scripts/bm25_diagnostic.py). Stub; ``ENABLE_BM25=False``.
``fusion``  Gated out with BM25 - with one ranked list there is nothing to
            fuse. Stub; reached only when ``ENABLE_BM25`` is True.
``rerank``  BUILT and measured, then disabled: cross-encoder reordering LOSES
            NDCG@10 at every pool depth (-0.153 at 25, -0.258 at 100).
            ``CrossEncoderReranker`` is a real implementation and carries the
            NaN guard for ms-marco-MiniLM-L-6-v2; ``ENABLE_RERANK=False``.
"""

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import get_fusion_strategy, reciprocal_rank_fusion
from src.retrieval.rerank import CrossEncoderReranker, NoOpReranker, get_reranker

__all__ = [
    "BM25Retriever",
    "CrossEncoderReranker",
    "DenseRetriever",
    "NoOpReranker",
    "get_fusion_strategy",
    "get_reranker",
    "reciprocal_rank_fusion",
]
