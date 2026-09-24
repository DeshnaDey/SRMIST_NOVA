"""PRISM Agentic Code Intelligence - natural-language retrieval over code.

Samsung PRISM GenAI Hackathon 2026, Theme 1.

Package layout
--------------
``config``      Every tunable in the system. Start here.
``interfaces``  The stage contracts all four workstreams build against.
``query``       Stage 1 - gated out. Passthrough on the shipped path.
``corpus``      Stage 2 - snippet passthrough (gated out) + dense index build.
``retrieval``   Stage 3 - dense search SHIPS. BM25, fusion and cross-encoder
                rerank are all gated out; see their modules for the measured
                bound that gated each one.
``pipeline``    The MTEB v2 adapters that expose the above to the benchmark.
``versioning``  Content-hashed embedding cache. The one optional stage that
                ships ON.

WHAT ACTUALLY RUNS
------------------
    query -> instruction prefix + token budget -> bi-encoder -> FAISS -> top-k

One dense stage. Every other stage in the tree is present, flag-gated off, and
kept as the evidence behind the ablation table in ``experiments.md``.
"""

__version__ = "0.1.0"
