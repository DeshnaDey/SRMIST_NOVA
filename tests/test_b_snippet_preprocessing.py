"""Category B - snippet preprocessing.  (owner: corpus)

The id-preservation tests here are the most important in the suite. A mangled
corpus id does not raise, does not warn, and scores zero - the only thing that
catches it is a test like ``test_ids_are_preserved_exactly``.
"""

from __future__ import annotations

import pytest

from src.corpus.preprocess import NoOpSnippetProcessor, get_snippet_processor
from src.interfaces import Snippet, SnippetProcessor


# =============================================================================
# Contract tests - must hold for EVERY SnippetProcessor implementation
# =============================================================================


@pytest.fixture(params=[NoOpSnippetProcessor])
def processor(request: pytest.FixtureRequest) -> SnippetProcessor:
    """TODO(corpus): add ``PrismSnippetProcessor`` here once implemented."""
    return request.param()


def test_ids_are_preserved_exactly(
    processor: SnippetProcessor, raw_snippets: list[Snippet]
) -> None:
    """THE critical invariant: ids survive byte-for-byte.

    MTEB joins our output to its qrels by this string. Mangle it and the score
    silently drops to zero with no error anywhere.
    """
    for snippet in raw_snippets:
        assert processor.process(snippet)["id"] == snippet["id"]


def test_batch_preserves_length_and_order(
    processor: SnippetProcessor, raw_snippets: list[Snippet]
) -> None:
    """Preprocessing is a text transform, never a filter.

    Index positions map back to ids by offset, so dropping or reordering one
    row shifts every id after it.
    """
    results = processor.process_batch(raw_snippets)
    assert len(results) == len(raw_snippets)
    assert [r["id"] for r in results] == [s["id"] for s in raw_snippets]


def test_output_shape(processor: SnippetProcessor, raw_snippets: list[Snippet]) -> None:
    """Output has exactly the ProcessedSnippet keys, with a string body."""
    result = processor.process(raw_snippets[0])
    assert set(result) == {"id", "processed_text"}
    assert isinstance(result["processed_text"], str)


def test_handles_empty_snippet(processor: SnippetProcessor) -> None:
    """An empty document is kept, not dropped."""
    result = processor.process({"id": "empty", "text": ""})
    assert result["id"] == "empty"
    assert isinstance(result["processed_text"], str)


def test_never_raises_on_unparseable_code(processor: SnippetProcessor) -> None:
    """Syntactically invalid code falls back to the original text."""
    processor.process({"id": "bad", "text": "def broken(:\n  this is not python"})


def test_get_snippet_processor_respects_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory returns the no-op when the stage is disabled."""
    from src import config

    monkeypatch.setattr(config, "ENABLE_SNIPPET_PREPROCESSING", False)
    assert isinstance(get_snippet_processor(), NoOpSnippetProcessor)


# =============================================================================
# Behaviour tests - TODO(corpus)
# =============================================================================


@pytest.mark.skip(reason="TODO(corpus): implement PrismSnippetProcessor")
def test_truncates_to_max_snippet_tokens() -> None:
    """Long snippets are cut to config.MAX_SNIPPET_TOKENS.

    Measured with the tokenizer of config.DENSE_MODEL_NAME.
    """


@pytest.mark.skip(reason="TODO(corpus): implement comment stripping")
def test_strips_comments_when_enabled() -> None:
    """config.STRIP_COMMENTS removes comments and docstrings.

    Worth A/B-ing rather than assuming: comments are often the only
    natural-language bridge between a query and a code snippet.
    """


@pytest.mark.skip(reason="TODO(corpus): implement comment stripping")
def test_does_not_strip_comments_when_disabled() -> None:
    """The flag actually gates the behaviour."""


@pytest.mark.skip(reason="TODO(corpus): implement chunking")
def test_chunking_preserves_snippet_identity() -> None:
    """Chunking must still expose one ProcessedSnippet per input id."""
