"""Category G - cross-encoder reranking.  (owner: retrieval)

The no-op reranker is real and tested here. For ``CrossEncoderReranker``,
prefer a stub model that returns scripted scores over downloading a real
checkpoint - the logic worth testing (ordering, truncation, id integrity,
failure fallback) is all independent of the model.
"""

from __future__ import annotations

import pytest

from src.interfaces import ProcessedQuery
from src.retrieval.rerank import NoOpReranker, get_reranker


# =============================================================================
# No-op reranker - the control condition
# =============================================================================


def test_noop_preserves_order(
    sample_query: ProcessedQuery, dense_ranking: list[tuple[str, float]]
) -> None:
    """The passthrough returns candidates untouched."""
    result = NoOpReranker().rerank(sample_query, dense_ranking, top_k=10)
    assert result == dense_ranking


def test_noop_respects_top_k(
    sample_query: ProcessedQuery, dense_ranking: list[tuple[str, float]]
) -> None:
    """Even the passthrough honours the length cap."""
    assert len(NoOpReranker().rerank(sample_query, dense_ranking, top_k=2)) == 2


def test_noop_handles_empty_candidates(sample_query: ProcessedQuery) -> None:
    """No candidates is not an error."""
    assert NoOpReranker().rerank(sample_query, [], top_k=10) == []


def test_get_reranker_returns_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """config.ENABLE_RERANK gates the stage."""
    from src import config

    monkeypatch.setattr(config, "ENABLE_RERANK", False)
    assert isinstance(get_reranker(), NoOpReranker)


def test_get_reranker_requires_lookup_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabling rerank without snippet text fails at construction.

    Much cheaper than discovering it 40 minutes into an evaluation.
    """
    from src import config

    monkeypatch.setattr(config, "ENABLE_RERANK", True)
    with pytest.raises(ValueError, match="snippet_lookup"):
        get_reranker(None)


# =============================================================================
# Cross-encoder behaviour - TODO(retrieval)
# =============================================================================


@pytest.mark.skip(reason="TODO(retrieval): implement CrossEncoderReranker.rerank")
def test_reorders_by_cross_encoder_score() -> None:
    """With a stub model scoring doc_2 highest, doc_2 comes back at rank 1."""


@pytest.mark.skip(reason="TODO(retrieval): implement CrossEncoderReranker.rerank")
def test_never_invents_corpus_ids() -> None:
    """Output ids are a subset of the input candidate ids."""


@pytest.mark.skip(reason="TODO(retrieval): implement CrossEncoderReranker.rerank")
def test_only_scores_rerank_top_n() -> None:
    """Only config.RERANK_TOP_N candidates reach the model.

    This is the dominant CPU cost - assert the stub saw exactly that many pairs.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement CrossEncoderReranker.rerank")
def test_unreranked_tail_keeps_fused_order() -> None:
    """Candidates past RERANK_TOP_N stay in fused order at the end of the list."""


@pytest.mark.skip(reason="TODO(retrieval): implement CrossEncoderReranker.rerank")
def test_falls_back_to_input_order_on_model_failure() -> None:
    """A raising stub model yields candidates[:top_k], not an exception.

    A degraded ranking beats a crashed evaluation.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement CrossEncoderReranker.rerank")
def test_missing_snippet_text_is_skipped_not_fatal() -> None:
    """A candidate id absent from snippet_lookup must not crash the query."""
