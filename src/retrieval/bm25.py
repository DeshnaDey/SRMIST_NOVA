"""Lexical BM25 retrieval.  (owner: retrieval)

The sparse half of the hybrid. Catches exact matches the embedding model
blurs: specific function names, library calls, error strings, rare identifiers.
Cheap to run and a reliable NDCG contributor on code corpora, where queries
often quote the exact API they want.
"""

from __future__ import annotations

from src.corpus.index import BM25IndexBuilder
from src.interfaces import ProcessedQuery, RankedList, Retriever


class BM25Retriever(Retriever):
    """Okapi BM25 scoring over the tokenized corpus.

    TODO(retrieval): implement.

    Retrieve outline
    ----------------
    1. Tokenize ``query.text`` with ``src.corpus.index.tokenize_code`` - the
       SAME function used to build the index. Do not hand-roll a second
       tokenizer here; a mismatch between index-side and query-side
       tokenization silently destroys recall and no assertion will catch it.
    2. ``scores = bm25.get_scores(query_tokens)`` - one score per corpus doc.
    3. Take the ``top_k`` highest via ``numpy.argpartition`` (O(N), versus
       sorting all ~100k scores per query).
    4. Map positions through ``index_builder.ids``.

    Return ``[]`` for a query that tokenizes to nothing rather than raising.

    Note on scores
    --------------
    BM25 scores are unbounded and corpus-dependent - typically 0-30, not 0-1.
    They are NOT comparable to cosine similarities, which is precisely why the
    default fusion strategy is rank-based RRF rather than a score sum.
    """

    def __init__(self, index_builder: BM25IndexBuilder) -> None:
        """Parameters
        ----------
        index_builder:
            An already-built :class:`BM25IndexBuilder` carrying the BM25
            statistics and the position -> corpus id mapping.
        """
        self.index_builder = index_builder

    def retrieve(self, query: ProcessedQuery, top_k: int) -> RankedList:
        """Return the ``top_k`` highest-scoring snippets. See class docstring."""
        # TODO(retrieval): implement.
        raise NotImplementedError("TODO(retrieval): BM25Retriever.retrieve")
