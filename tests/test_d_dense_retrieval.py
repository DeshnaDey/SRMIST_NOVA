"""Category D - dense bi-encoder retrieval.  (owner: retrieval)

Tests that need a real checkpoint must carry ``@pytest.mark.slow`` so the
default ``pytest -m "not slow"`` run stays offline and fast. Prefer a fake
index with hand-written vectors for the logic tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(retrieval): implement DenseRetriever.retrieve")
def test_returns_at_most_top_k() -> None:
    """Returning more than top_k distorts fusion downstream."""


@pytest.mark.skip(reason="TODO(retrieval): implement DenseRetriever.retrieve")
def test_results_are_sorted_descending() -> None:
    """Ranked lists are always sorted descending by score."""


@pytest.mark.skip(reason="TODO(retrieval): implement DenseRetriever.retrieve")
def test_returns_known_corpus_ids_only() -> None:
    """Every returned id came from the indexed corpus."""


@pytest.mark.skip(reason="TODO(retrieval): implement DenseRetriever.retrieve")
def test_filters_faiss_padding_positions() -> None:
    """FAISS pads with -1 when the index holds fewer than top_k vectors.

    An unfiltered -1 indexes ids[-1], which is a real id and a wrong answer.
    Build an index with 2 documents, ask for top_k=10.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement DenseRetriever.retrieve")
def test_empty_index_returns_empty_list() -> None:
    """Degrade, don't raise."""


@pytest.mark.skip(reason="TODO(retrieval): implement retrieve_batch")
def test_batch_matches_sequential_results() -> None:
    """retrieve_batch is an optimization, not a behaviour change."""


@pytest.mark.skip(reason="TODO(retrieval): implement prompt prefixes")
def test_query_prompt_prefix_is_applied() -> None:
    """E5/BGE/GTE-family models need their prefix and degrade quietly without it."""


@pytest.mark.slow
@pytest.mark.skip(reason="TODO(retrieval): needs a downloaded checkpoint")
def test_retrieves_semantically_relevant_snippet() -> None:
    """"find an element in a sorted array" ranks doc_1 (binary_search) first.

    The end-to-end sanity check: shares almost no literal tokens with the
    query, so only the dense side can find it.
    """
