"""Rank fusion.  (owner: retrieval)

Merges the dense and BM25 ranked lists into one. Implements
:class:`src.interfaces.FusionStrategy` (a Protocol - plain functions qualify).

Why rank-based fusion is the default
------------------------------------
Cosine similarities sit in roughly [-1, 1]; BM25 scores are unbounded and
corpus-dependent. Summing them directly lets BM25 dominate purely because its
numbers are bigger. Reciprocal Rank Fusion sidesteps the problem by throwing
away the scores and keeping only the ranks, which is why it is both the default
here and a persistently hard baseline to beat.

MTEB v2 ships its own ``mteb.HybridSearch(models=[...], fusion_strategy="rrf")``.
We implement fusion ourselves so that the preprocessing and reranking stages
sit inside the same pipeline - but HybridSearch is a useful sanity check to
compare our RRF against if the numbers look wrong.
"""

from __future__ import annotations

from src import config
from src.interfaces import FusionStrategy, RankedList


def reciprocal_rank_fusion(
    rankings: list[RankedList],
    top_k: int,
    k: int | None = None,
) -> RankedList:
    """Fuse ranked lists by Reciprocal Rank Fusion.

    TODO(retrieval): implement.

    The formula
    -----------
    For each document d::

        RRF(d) = sum over each ranking r containing d of  1 / (k + rank_r(d))

    where ``rank`` is 1-based (the top hit is rank 1) and ``k`` damps the
    influence of the very top positions. ``k = 60`` is the value from the
    original RRF paper and the usual default; it lives in ``config.RRF_K``.

    Parameters
    ----------
    rankings:
        One :data:`RankedList` per retriever, each sorted descending. The
        pipeline passes ``[dense_results, bm25_results]``. Any list may be
        empty. The lists overlap only partially - a document appearing in one
        and not the other is the normal case and contributes only its one term.
    top_k:
        Length cap on the fused output.
    k:
        RRF damping constant; defaults to ``config.RRF_K``.

    Returns
    -------
    RankedList
        Sorted descending by fused score, length ``<= top_k``, NO duplicate
        corpus ids. Deduplication happens here and nowhere else.

    Edge cases that must not raise
    ------------------------------
    * ``rankings == []`` or every list empty -> return ``[]``.
    * A single non-empty list -> return it re-scored, order preserved.
    * Ties -> break them deterministically (e.g. by corpus id), so that two
      runs of the same config produce byte-identical output. Non-deterministic
      tie-breaking makes experiments.md rows irreproducible.
    """
    raise NotImplementedError("TODO(retrieval): reciprocal_rank_fusion")


def weighted_score_fusion(
    rankings: list[RankedList],
    top_k: int,
    weights: tuple[float, ...] | None = None,
) -> RankedList:
    """Fuse ranked lists by weighted sum of NORMALIZED scores.

    TODO(retrieval): implement, only if RRF is measurably beaten.

    Unlike RRF this keeps score magnitudes, so it can express "dense is much
    more confident here" - but it only works if each list's scores are first
    normalized to a common range (min-max per query is the usual choice).
    Skipping that normalization silently hands the ranking to whichever
    retriever has the larger numbers.

    Parameters
    ----------
    weights:
        One weight per input ranking, same order as ``rankings``. Defaults to
        ``config.FUSION_WEIGHTS``. Conventionally sums to 1.0.

    Returns
    -------
    RankedList
        Sorted descending, deduplicated, length ``<= top_k``.
    """
    raise NotImplementedError("TODO(retrieval): weighted_score_fusion")


def get_fusion_strategy() -> FusionStrategy:
    """Return the fusion callable named by ``config.FUSION_STRATEGY``.

    Raises
    ------
    ValueError
        If the configured name is unknown - fail loudly at startup rather than
        silently falling back to a strategy nobody chose.
    """
    strategy = config.FUSION_STRATEGY.lower()
    if strategy == "rrf":
        return reciprocal_rank_fusion
    if strategy == "weighted_sum":
        return weighted_score_fusion
    raise ValueError(
        f"Unknown FUSION_STRATEGY {config.FUSION_STRATEGY!r}; "
        "expected 'rrf' or 'weighted_sum'"
    )
