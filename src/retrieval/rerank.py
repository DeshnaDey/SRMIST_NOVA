"""Cross-encoder reranking.  (owner: retrieval)

The last stage, and usually the largest single NDCG@10 jump - reordering the
top candidates is exactly what NDCG@10 measures.

Bi-encoder vs cross-encoder
---------------------------
The bi-encoder embeds query and snippet separately, so snippet vectors can be
precomputed once - cheap, but the two never interact. A cross-encoder runs the
query and snippet through the model *together*, so it can attend across them,
which is far more accurate and far too slow to run over the whole corpus.
Hence: bi-encoder retrieves ~100 candidates, cross-encoder reorders them.

On CPU, ``config.RERANK_TOP_N`` is the dominant latency knob. Every candidate
is a full forward pass. Start at 25-50 and measure wall-clock before raising
it - a config that cannot finish the evaluation scores nothing.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src import config
from src.interfaces import CorpusId, ProcessedQuery, RankedList, Reranker

logger = logging.getLogger(__name__)


class NoOpReranker(Reranker):
    """Passthrough baseline: returns the fused order untouched.

    The control condition for measuring what reranking actually buys.
    """

    def rerank(
        self,
        query: ProcessedQuery,
        candidates: RankedList,
        top_k: int,
    ) -> RankedList:
        """Return ``candidates[:top_k]`` unchanged."""
        return candidates[:top_k]


class CrossEncoderReranker(Reranker):
    """Reorders candidates with a cross-encoder.

    Rerank outline
    --------------
    1. Take the first ``config.RERANK_TOP_N`` candidates - the tail is not
       worth the forward passes.
    2. Look up each candidate's snippet text by corpus id (see
       ``snippet_lookup`` below - the reranker needs the text, and only ids
       flow through the ranked lists).
    3. Build pairs ``[(query_text, snippet_text), ...]``. Try ``query.raw`` as
       well as ``query.text``: cross-encoders are trained on natural questions
       and often prefer the unstripped phrasing.
    4. ``model.predict(pairs, batch_size=config.RERANK_BATCH_SIZE)``.
    5. Sort descending by the new scores, return ``top_k``.
    6. Candidates beyond ``RERANK_TOP_N`` keep their fused order and go at the
       end, if ``top_k`` reaches that far.

    Hard requirements
    -----------------
    * Never invent a corpus id that was not in ``candidates``.
    * On model failure, return ``candidates[:top_k]`` rather than raising. A
      degraded ranking beats a crashed run.
    """

    def __init__(
        self,
        snippet_lookup: dict[str, str],
        model_name: str | None = None,
    ) -> None:
        """Parameters
        ----------
        snippet_lookup:
            ``{corpus_id: snippet_text}`` for every indexed document. The
            ranked lists carry only ids, so the reranker needs this to
            reconstruct the text. Use the PROCESSED text, so the reranker sees
            what the retriever saw.
        model_name:
            Cross-encoder checkpoint; defaults to ``config.RERANK_MODEL_NAME``.
        """
        self.snippet_lookup = snippet_lookup
        self.model_name = model_name or config.RERANK_MODEL_NAME
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """The CrossEncoder, loaded on first use.

        Lazy for the same reason the bi-encoder is: constructing this object
        in a test, or to read its metadata, must not pull weights off the Hub.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s on %s",
                        self.model_name, config.DEVICE)
            self._model = CrossEncoder(self.model_name, device=config.DEVICE)
        return self._model

    def rerank(
        self,
        query: ProcessedQuery,
        candidates: RankedList,
        top_k: int,
    ) -> RankedList:
        """Reorder ``candidates``. See class docstring for the outline.

        The tail beyond ``RERANK_TOP_N`` keeps its incoming order and is
        appended after the rescored head, so this can only ever permute the
        ranking - it never drops or invents a candidate.
        """
        if not candidates:
            return []

        head = candidates[: config.RERANK_TOP_N]
        tail = candidates[config.RERANK_TOP_N :]

        # Cross-encoders are trained on natural questions, and the stub notes
        # the raw phrasing often reads better to them than a stripped one.
        # ``raw`` is the untouched query; it falls back to ``text`` when a
        # processor did not keep the original around.
        query_text = getattr(query, "raw", None) or query.text

        pairs: list[tuple[str, str]] = []
        scored_ids: list[CorpusId] = []
        missing: list[tuple[CorpusId, float]] = []
        for cid, score in head:
            snippet = self.snippet_lookup.get(cid)
            if snippet is None:
                # An id with no text cannot be rescored. Keep it rather than
                # drop it - "never invent an id" cuts both ways.
                missing.append((cid, score))
                continue
            pairs.append((query_text, snippet))
            scored_ids.append(cid)

        if not pairs:
            return candidates[:top_k]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=config.RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 - a degraded ranking beats a crash
            logger.warning("Cross-encoder failed (%s); keeping the fused order",
                           exc)
            return candidates[:top_k]

        # NaN GUARD - do not remove.
        #
        # cross-encoder/ms-marco-MiniLM-L-6-v2, the obvious default and the
        # checkpoint this constant used to name, returns NaN for EVERY pair on
        # the pinned stack (finite fp32 weights, NaN out of encoder layer 0,
        # under both sdpa and eager attention). Sorting by NaN is a no-op that
        # silently preserves the input order, so the pipeline would report a
        # perfectly clean "reranking changed nothing" instead of an error -
        # exactly the silent failure this repo keeps getting bitten by.
        # Sibling checkpoints (L-4, L-12, TinyBERT-L-2) and bge-reranker-base
        # are all finite, so this is checkpoint-specific, not stack-wide.
        finite = [(cid, float(sc)) for cid, sc in zip(scored_ids, scores)
                  if math.isfinite(float(sc))]
        if len(finite) != len(scored_ids):
            logger.warning(
                "%s returned %d non-finite scores of %d; keeping the fused "
                "order. This model is unusable for reranking here.",
                self.model_name, len(scored_ids) - len(finite), len(scored_ids),
            )
            return candidates[:top_k]

        reordered = sorted(finite, key=lambda pair: pair[1], reverse=True)
        return (reordered + missing + tail)[:top_k]


def get_reranker(snippet_lookup: dict[str, str] | None = None) -> Reranker:
    """Return the reranker selected by ``config.ENABLE_RERANK``.

    Parameters
    ----------
    snippet_lookup:
        Required when reranking is enabled; ignored otherwise.

    Raises
    ------
    ValueError
        If reranking is enabled but no lookup was supplied - catching that at
        construction is much cheaper than discovering it mid-evaluation.
    """
    if not config.ENABLE_RERANK:
        return NoOpReranker()
    if snippet_lookup is None:
        raise ValueError(
            "ENABLE_RERANK is True but no snippet_lookup was provided; "
            "the cross-encoder needs snippet text, not just ids"
        )
    return CrossEncoderReranker(snippet_lookup)
