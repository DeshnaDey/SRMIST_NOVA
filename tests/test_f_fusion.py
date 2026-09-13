"""Category F - rank fusion.  (owner: retrieval)

Pure functions over hand-written ranked lists: no models, no indexes, no
network. This is the cheapest category to test properly, and RRF has enough
edge cases to deserve it.
"""

from __future__ import annotations

import pytest

from src.retrieval.fusion import get_fusion_strategy


def test_get_fusion_strategy_returns_configured_callable() -> None:
    """The factory resolves config.FUSION_STRATEGY to a callable."""
    assert callable(get_fusion_strategy())


def test_get_fusion_strategy_rejects_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown strategy fails loudly at startup, not silently at query time."""
    from src import config

    monkeypatch.setattr(config, "FUSION_STRATEGY", "not_a_strategy")
    with pytest.raises(ValueError, match="Unknown FUSION_STRATEGY"):
        get_fusion_strategy()


# =============================================================================
# RRF behaviour - TODO(retrieval)
# =============================================================================


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_matches_hand_computed_scores() -> None:
    """Verify the formula directly: RRF(d) = sum 1 / (k + rank).

    With k=60 and doc_1 at rank 1 in dense, rank 2 in bm25:
        1/61 + 1/62 = 0.032922...
    Hand-compute two or three documents from the dense_ranking and
    bm25_ranking fixtures and assert the exact values.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_deduplicates() -> None:
    """A document in both input lists appears exactly once in the output."""


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_includes_documents_present_in_only_one_list() -> None:
    """Partial overlap is the normal case, not an error.

    ``apps/train/0042`` is BM25-only and must survive fusion.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_ignores_score_magnitudes() -> None:
    """RRF depends only on ranks.

    Multiply every BM25 score by 1000 and the fused order must not change -
    that scale-invariance is the entire reason RRF is the default.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_respects_top_k() -> None:
    """Output length <= top_k."""


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_empty_inputs() -> None:
    """[] and [[], []] both return [] without raising."""


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_single_list_preserves_order() -> None:
    """One input list comes back re-scored but in the same order."""


@pytest.mark.skip(reason="TODO(retrieval): implement reciprocal_rank_fusion")
def test_rrf_ties_break_deterministically() -> None:
    """Two runs produce byte-identical output.

    Non-deterministic tie-breaking makes experiments.md rows irreproducible -
    a 0.001 NDCG wobble that nobody can explain.
    """


@pytest.mark.skip(reason="TODO(retrieval): implement weighted_score_fusion")
def test_weighted_fusion_normalizes_before_summing() -> None:
    """Without per-list normalization, BM25's larger numbers win by default."""
