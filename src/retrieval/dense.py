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

    @property
    def encoder(self):
        """The shared bi-encoder, loaded once per retriever."""
        if self._model is None:
            from src.pipeline.baseline import BaselineEncoder

            self._model = BaselineEncoder(self.model_name)
        return self._model

    def retrieve(self, query: ProcessedQuery, top_k: int) -> RankedList:
        """Return the ``top_k`` nearest snippets. See class docstring."""
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(
        self, queries: list[ProcessedQuery], top_k: int
    ) -> list[RankedList]:
        """Batched search - encode all queries in one forward pass.

        Done properly rather than as a per-query loop: on CPU this is the
        difference between minutes and an hour over the full query set, and
        FAISS ``search`` is natively batched.

        The query prompt prefix is applied by ``BaselineEncoder`` via
        ``prompt_type="query"`` - it is NOT applied here as well. Prefixing
        twice is silent and costs accuracy, which is the failure mode called
        out in guide.md.
        """
        import numpy as np

        if not queries:
            return []

        vecs = self.encoder.encode(
            [q.text for q in queries], prompt_type="query",
            batch_size=config.BATCH_SIZE,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        )
        vecs = np.ascontiguousarray(np.asarray(vecs, dtype=np.float32))
        if config.NORMALIZE_EMBEDDINGS:
            # numpy, not faiss.normalize_L2: importing faiss alongside torch
            # segfaults this build (see src/corpus/index.py _use_exact_numpy).
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)

        index = self.index_builder.index
        ids = self.index_builder.ids
        k = min(top_k, index.ntotal)
        scores, positions = index.search(vecs, k)

        out: list[RankedList] = []
        for row_scores, row_positions in zip(scores, positions):
            ranked: RankedList = []
            for score, pos in zip(row_scores, row_positions):
                # FAISS pads with -1 when the index holds fewer than k
                # vectors. ids[-1] is a REAL id and a wrong answer.
                if pos < 0:
                    continue
                ranked.append((ids[pos], float(score)))
            out.append(ranked)
        return out
