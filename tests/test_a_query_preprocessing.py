"""Category A - query preprocessing.  (owner: query)

Placeholders below are marked ``skip``; drop the marker as you implement each
one. The contract tests at the top already pass against the no-op processor and
should keep passing against the real one - they encode the rules from
``src/interfaces.py``, not any particular cleaning strategy.
"""

from __future__ import annotations

import pytest

from src.interfaces import ProcessedQuery, QueryProcessor
from src.query.preprocess import NoOpQueryProcessor, get_query_processor


# =============================================================================
# Contract tests - must hold for EVERY QueryProcessor implementation
# =============================================================================


@pytest.fixture(params=[NoOpQueryProcessor])
def processor(request: pytest.FixtureRequest) -> QueryProcessor:
    """Every implementation, one at a time.

    TODO(query): add ``PrismQueryProcessor`` to ``params`` once implemented -
    the contract tests then cover it for free.
    """
    return request.param()


def test_returns_processed_query(processor: QueryProcessor) -> None:
    """process() returns a ProcessedQuery, not a bare string."""
    result = processor.process("sort a list")
    assert isinstance(result, ProcessedQuery)


def test_raw_is_preserved_verbatim(processor: QueryProcessor) -> None:
    """.raw must be the untouched input - reranking depends on it."""
    raw = "  Write a function that sorts a list.  \n"
    assert processor.process(raw).raw == raw


def test_never_raises_on_degenerate_input(
    processor: QueryProcessor, raw_queries: list[str]
) -> None:
    """Empty and whitespace-only queries must degrade, not crash.

    A raise here kills a two-hour evaluation at query 9,000.
    """
    for query in raw_queries:
        processor.process(query)


def test_is_deterministic(processor: QueryProcessor) -> None:
    """Same input, same output - otherwise experiments.md is not reproducible."""
    query = "compute the nth fibonacci number"
    first, second = processor.process(query), processor.process(query)
    assert first.text == second.text
    assert first.category == second.category


def test_batch_preserves_length_and_order(
    processor: QueryProcessor, raw_queries: list[str]
) -> None:
    """process_batch is a 1:1 map over its input."""
    results = processor.process_batch(raw_queries)
    assert len(results) == len(raw_queries)
    assert [r.raw for r in results] == raw_queries


def test_get_query_processor_respects_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory returns the no-op when the stage is disabled."""
    from src import config

    monkeypatch.setattr(config, "ENABLE_QUERY_PREPROCESSING", False)
    assert isinstance(get_query_processor(), NoOpQueryProcessor)


# =============================================================================
# Behaviour tests - TODO(query)
# =============================================================================


@pytest.mark.skip(reason="TODO(query): implement PrismQueryProcessor")
def test_strips_boilerplate_framing() -> None:
    """"Write a Python function that reverses a string" -> "reverses a string"."""


@pytest.mark.skip(reason="TODO(query): implement PrismQueryProcessor")
def test_drops_example_io_blocks() -> None:
    """APPS prompts carry sample input/output blocks that add no retrieval signal."""


@pytest.mark.skip(reason="TODO(query): implement PrismQueryProcessor")
def test_cleaning_never_empties_a_nonempty_query() -> None:
    """Falls back to raw when cleaning would strip everything.

    See config.MIN_QUERY_CHARS. An unclean query retrieves far better than an
    empty one.
    """


@pytest.mark.skip(reason="TODO(query): implement classification")
def test_category_is_from_the_known_taxonomy() -> None:
    """.category is None or a member of config.QUERY_CATEGORIES."""


@pytest.mark.skip(reason="TODO(query): implement PrismQueryProcessor")
def test_respects_max_query_chars() -> None:
    """Output honours config.MAX_QUERY_CHARS."""
