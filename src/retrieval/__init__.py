"""Stage 3 - retrieval, fusion and reranking.  (owner: retrieval)

``dense``   bi-encoder + FAISS semantic search
``bm25``    lexical search over tokenized code
``fusion``  RRF / weighted-sum merge of the two ranked lists
``rerank``  cross-encoder reordering of the fused candidates
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
