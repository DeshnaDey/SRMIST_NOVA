"""Query cleaning and classification.  (owner: query)

The pipeline always calls :func:`get_query_processor`, which returns a no-op
passthrough while ``config.ENABLE_QUERY_PREPROCESSING`` is False. That keeps
the end-to-end run green from day one and makes this stage a single-flag A/B.

THIS IS THE HIGH-VALUE STAGE. Queries here are not short natural-language
asks - they are full competitive-programming problem statements, median ~1,050
characters. MEASURED (data/inspection_report.md): **61.9% of queries overflow
the current model's 254-token window**, against 23.5% of snippets. Whatever we
recover here is worth roughly 2.6x the equivalent work on the corpus side.

Ideas worth measuring (log each one in experiments.md):
  - DROP THE Input/Output FORMAT SECTIONS AND WORKED EXAMPLES. These are
    boilerplate-shaped, they sit at the END of the statement, and they are
    eating the token budget that the actual problem description needs. This is
    the one to try first.
  - normalize unicode punctuation and collapse whitespace
  - pull out identifier-like tokens to help the BM25 half of the hybrid
  - classify into config.QUERY_CATEGORIES to allow per-category routing

DO NOT bother stripping leading boilerplate framing
---------------------------------------------------
An earlier version of this file suggested stripping openers like "Write a
Python function that ..." on the assumption they appear in every query. That
assumption was tested on the train split and is false:

    longest common prefix across all 5,000 train queries : "" (none)
    queries opening with "Write a function"              : 2.0%
    queries opening with "You are given"                 : 4.4%
    queries opening with "Given"                         : 11.5%

There is no shared prefix to strip. A regex aimed at these would touch a small
minority of queries and cannot move the metric. The budget problem is the
trailing example blocks, not the opening sentence.
"""

from __future__ import annotations

from src import config
from src.interfaces import ProcessedQuery, QueryProcessor


class NoOpQueryProcessor(QueryProcessor):
    """Passthrough baseline: the cleaned text *is* the raw text.

    This is the control condition. Any real processor has to beat it on
    NDCG@10, and surprisingly often the naive cleanups lose - the framing
    carries signal the embedding model uses. Measure, do not assume.
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
    * Respect config.MAX_QUERY_TOKENS - a TOKEN budget, resolved from the
      active checkpoint. Measure length with that model's tokenizer; a
      character count is the wrong unit and the old cap never fired.
    """

    def __init__(self) -> None:
        # TODO(query): load whatever the classifier needs (keyword lists, a
        # small model, regex tables). Keep it CPU-cheap - this runs per query.
        # Start with example-block removal, not prefix stripping: see the
        # module docstring for why the prefix idea was dropped.
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
