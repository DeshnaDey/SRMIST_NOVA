"""PRISM Agentic Code Intelligence - natural-language retrieval over code.

Samsung PRISM GenAI Hackathon 2026, Theme 1.

Package layout
--------------
``config``      Every tunable in the system. Start here.
``interfaces``  The stage contracts all four workstreams build against.
``query``       Stage 1: query cleaning + classification.
``corpus``      Stage 2: snippet normalization + index construction.
``retrieval``   Stage 3: dense search, BM25, RRF fusion, cross-encoder rerank.
``pipeline``    The MTEB v2 adapters that expose the above to the benchmark.
``versioning``  Content-hashed embedding cache.
"""

__version__ = "0.1.0"
