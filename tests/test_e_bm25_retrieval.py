"""Category E - lexical BM25 retrieval.  (owner: retrieval)

BM25 needs no model download, so this category can be tested fully offline -
no ``slow`` marker needed. It should end up the best-covered retrieval file.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(retrieval): implement BM25Retriever.retrieve")
def test_returns_at_most_top_k() -> None:
    """Length cap is part of the Retriever contract."""


@pytest.mark.skip(reason="TODO(retrieval): implement BM25Retriever.retrieve")
def test_results_are_sorted_descending() -> None:
    """Ranked lists are always sorted descending by score."""


@pytest.mark.skip(reason="TODO(retrieval): implement BM25Retriever.retrieve")
def test_exact_identifier_match_ranks_first() -> None:
    """Query "bubble_sort" puts doc_2 at rank 1.

    This is precisely the case the dense retriever blurs and BM25 exists to
    rescue.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement BM25Retriever.retrieve")
def test_query_with_no_matching_tokens_returns_empty_or_low_scores() -> None:
    """No shared vocabulary must not raise."""


@pytest.mark.skip(reason="TODO(retrieval): implement BM25Retriever.retrieve")
def test_empty_query_returns_empty_list() -> None:
    """A query that tokenizes to nothing returns [], it does not crash."""


@pytest.mark.skip(reason="TODO(retrieval): implement BM25Retriever.retrieve")
def test_uses_shared_tokenizer() -> None:
    """Query-side tokenization goes through corpus.index.tokenize_code.

    Monkeypatch it and assert it was called, so a future refactor cannot
    quietly introduce a second tokenizer.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement BM25Retriever.retrieve")
def test_top_k_selection_is_correct_not_just_fast() -> None:
    """argpartition avoids a full sort; assert it still returns the true top-k."""
