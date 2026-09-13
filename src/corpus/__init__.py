"""Stage 2 - snippet preprocessing and index construction.  (owner: corpus)

``preprocess`` normalizes snippet text while preserving corpus ids;
``index`` builds the FAISS vector index and the BM25 index over the result.
"""

from src.corpus.index import BM25IndexBuilder, DenseIndexBuilder
from src.corpus.preprocess import NoOpSnippetProcessor, get_snippet_processor

__all__ = [
    "BM25IndexBuilder",
    "DenseIndexBuilder",
    "NoOpSnippetProcessor",
    "get_snippet_processor",
]
