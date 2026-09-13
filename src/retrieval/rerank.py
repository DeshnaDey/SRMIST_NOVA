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

from src import config
from src.interfaces import ProcessedQuery, RankedList, Reranker


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

    TODO(retrieval): implement.

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
        self._model = None  # TODO(retrieval): lazy-load CrossEncoder

    def rerank(
        self,
        query: ProcessedQuery,
        candidates: RankedList,
        top_k: int,
    ) -> RankedList:
        """Reorder ``candidates``. See class docstring for the outline."""
        # TODO(retrieval): implement.
        raise NotImplementedError("TODO(retrieval): CrossEncoderReranker.rerank")


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
