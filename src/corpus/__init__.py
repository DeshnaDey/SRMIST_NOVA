"""Stage 2 - snippet preprocessing and index construction.  (owner: corpus)

``preprocess`` would normalize snippet text while preserving corpus ids, but
the stage is gated out (``ENABLE_SNIPPET_PREPROCESSING=False``) and the shipped
path runs ``NoOpSnippetProcessor``; the variants were measured and gated out
before building (scripts/corpus_variants.py).

``index`` builds the dense vector index that SHIPS. ``BM25IndexBuilder`` is a
stub for the gated-out lexical half.
"""

from src.corpus.index import BM25IndexBuilder, DenseIndexBuilder
from src.corpus.preprocess import NoOpSnippetProcessor, get_snippet_processor

__all__ = [
    "BM25IndexBuilder",
    "DenseIndexBuilder",
    "NoOpSnippetProcessor",
    "get_snippet_processor",
]
