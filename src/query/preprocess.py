"""Query cleaning and classification.  (owner: query)

The pipeline always calls :func:`get_query_processor`, which returns a no-op
passthrough while ``config.ENABLE_QUERY_PREPROCESSING`` is False. That keeps
the end-to-end run green from day one and makes this stage a single-flag A/B.

Ideas worth measuring (log each one in experiments.md):
  - strip boilerplate framing: "Write a Python function that ...", "Given ..."
  - drop the worked examples / sample I/O blocks that APPS prompts carry
  - normalize unicode punctuation and collapse whitespace
  - pull out identifier-like tokens to help the BM25 half of the hybrid
  - classify into config.QUERY_CATEGORIES to allow per-category routing
"""

from __future__ import annotations

from src import config
from src.interfaces import ProcessedQuery, QueryProcessor


class NoOpQueryProcessor(QueryProcessor):
    """Passthrough baseline: the cleaned text *is* the raw text.

    This is the control condition. Any real processor has to beat it on
    NDCG@10, and surprisingly often the naive cleanups lose - the boilerplate
    carries signal the embedding model uses.
    """

    def process(self, raw_query: str) -> ProcessedQuery:
        """Return ``raw_query`` unchanged, with no category."""
        return ProcessedQuery(text=raw_query, raw=raw_query, category=None)


class PrismQueryProcessor(QueryProcessor):
    """The real query preprocessor.

    TODO(query): implement. Replace the passthrough body below.

    Expected behaviour
    ------------------
    Input : one raw query string, possibly empty, possibly several KB of
            problem statement with example I/O blocks.
    Output: ProcessedQuery(text=<cleaned>, raw=<input verbatim>,
                           category=<label from config.QUERY_CATEGORIES or None>,
                           metadata=<optional extras, e.g. {"keywords": [...]}>)

    Hard requirements
    -----------------
    * Never raise - a bad query must not kill a two-hour eval run.
    * Never return empty ``text``; fall back to ``raw_query`` if cleaning
      would strip everything (see config.MIN_QUERY_CHARS).
    * Deterministic: same input, same output, every run.
    * Respect config.MAX_QUERY_CHARS.
    """

    def __init__(self) -> None:
        # TODO(query): load whatever the classifier needs (keyword lists, a
        # small model, regex tables). Keep it CPU-cheap - this runs per query.
        pass

    def process(self, raw_query: str) -> ProcessedQuery:
        """Clean and classify one query. See class docstring for the contract."""
        # TODO(query): replace this passthrough with the real implementation.
        return ProcessedQuery(text=raw_query, raw=raw_query, category=None)


def get_query_processor() -> QueryProcessor:
    """Return the processor selected by config.

    The pipeline calls this rather than constructing a class directly, so the
    whole stage can be toggled from ``config.ENABLE_QUERY_PREPROCESSING``
    without touching pipeline code.
    """
    if config.ENABLE_QUERY_PREPROCESSING:
        return PrismQueryProcessor()
    return NoOpQueryProcessor()
