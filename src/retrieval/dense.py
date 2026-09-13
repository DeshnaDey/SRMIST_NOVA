"""Dense bi-encoder retrieval over the FAISS index.  (owner: retrieval)

Encodes the query with the same checkpoint used for the corpus and takes the
nearest neighbours by inner product (== cosine, given normalized vectors).

This is the semantic half of the hybrid: it matches "sort a list without
built-ins" to a bubble-sort implementation that shares no tokens with the
query. It is also the half that misses exact identifier matches, which is what
BM25 is there to cover.
"""

from __future__ import annotations

from src import config
from src.corpus.index import DenseIndexBuilder
from src.interfaces import ProcessedQuery, RankedList, Retriever


class DenseRetriever(Retriever):
    """Nearest-neighbour search over snippet embeddings.

    TODO(retrieval): implement.

    Retrieve outline
    ----------------
    1. Prepend ``config.QUERY_PROMPT_PREFIX`` if the checkpoint needs it.
       E5/BGE/GTE-family models lose a lot of accuracy without their prefix,
       and fail silently - nothing errors, the numbers are just worse.
    2. Encode ``query.text`` with the SAME model instance that built the index.
       A different checkpoint here produces vectors in an unrelated space and
       returns confident nonsense.
    3. L2-normalize if ``config.NORMALIZE_EMBEDDINGS``.
    4. ``scores, positions = index.search(query_vec, top_k)``
    5. Map positions back through ``index_builder.ids`` and return
       ``[(corpus_id, float(score)), ...]`` sorted descending.

    Watch out
    ---------
    FAISS returns ``-1`` for padding positions when the index holds fewer than
    ``top_k`` vectors. Filter those out or you will emit ``ids[-1]``, which is
    a real id and a wrong answer.
    """

    def __init__(
        self,
        index_builder: DenseIndexBuilder,
        model_name: str | None = None,
    ) -> None:
        """Parameters
        ----------
        index_builder:
            An already-built :class:`DenseIndexBuilder`, carrying both the
            FAISS index and the position -> corpus id mapping.
        model_name:
            Bi-encoder checkpoint; must match the one used to build the index.
            Defaults to ``config.DENSE_MODEL_NAME``.
        """
        self.index_builder = index_builder
        self.model_name = model_name or config.DENSE_MODEL_NAME
        self._model = None  # TODO(retrieval): lazy-load SentenceTransformer

    def retrieve(self, query: ProcessedQuery, top_k: int) -> RankedList:
        """Return the ``top_k`` nearest snippets. See class docstring."""
        # TODO(retrieval): implement.
        raise NotImplementedError("TODO(retrieval): DenseRetriever.retrieve")

    def retrieve_batch(
        self, queries: list[ProcessedQuery], top_k: int
    ) -> list[RankedList]:
        """Batched search - encode all queries in one forward pass.

        TODO(retrieval): implement. Worth doing properly: on CPU this is the
        difference between minutes and an hour over the full query set. FAISS
        ``search`` is natively batched, so pass the whole query matrix at once.
        """
        # TODO(retrieval): implement; falling back to the per-query loop is
        # correct but slow.
        return super().retrieve_batch(queries, top_k)
