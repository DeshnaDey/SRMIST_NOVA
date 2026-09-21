"""Index construction over the processed corpus.  (owner: corpus)

Two indexes, one interface (:class:`src.interfaces.IndexBuilder`):

``DenseIndexBuilder``  FAISS index over bi-encoder embeddings -> semantic match.
``BM25IndexBuilder``   rank_bm25 over tokenized code -> exact identifier match.

They are deliberately separate objects: the retrieval stage queries both and
fuses the results, and either one can be disabled from config.

THE ID MAPPING IS THE WHOLE GAME
--------------------------------
Both backends work in dense integer positions internally. The mapping from
position -> corpus id is what makes the output joinable to MTEB's qrels, so it
is built once, in corpus order, and must be saved and loaded alongside the
index itself. An index restored without its id map is worse than useless: it
returns confident, plausible, entirely wrong ids.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from src import config
from src.interfaces import CorpusId, IndexBuilder, ProcessedSnippet

logger = logging.getLogger(__name__)


class DenseIndexBuilder(IndexBuilder):
    """FAISS index over snippet embeddings.

    Build outline
    -------------
    1. Encode ``[s["processed_text"] for s in snippets]`` with the shared
       bi-encoder, in batches of ``config.BATCH_SIZE``, through the embedding
       cache (src/versioning/cache.py) so reruns are cheap.
    2. L2-normalize if ``config.NORMALIZE_EMBEDDINGS`` - required for cosine
       similarity via FAISS ``IndexFlatIP``.
    3. Build via ``faiss.index_factory(dim, config.FAISS_INDEX_FACTORY, metric)``.
       "Flat" is exact and O(N); only move to IVF/HNSW if latency forces it,
       and re-measure NDCG afterwards because ANN is lossy.
    4. Record ``self._ids`` in corpus order.

    Prefer the CPU-only faiss build; nothing here may request a GPU.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Parameters
        ----------
        model_name:
            Bi-encoder checkpoint. Defaults to ``config.DENSE_MODEL_NAME``.
        """
        self.model_name = model_name or config.DENSE_MODEL_NAME
        #: Position -> corpus id, in corpus order. Populated by build()/load().
        self._ids: list[CorpusId] = []
        #: The underlying faiss.Index. None until built.
        self._index = None

    def build(self, snippets: list[ProcessedSnippet]) -> None:
        """Encode and index the corpus. See class docstring for the outline.

        Encoding goes through :class:`~src.pipeline.baseline.BaselineEncoder`
        rather than a fresh SentenceTransformer, so the vectors are produced by
        exactly the code path every measured number in ``experiments.md`` came
        from - same prompt prefix, same window cap, same normalisation. A
        second, subtly different encoder here would put the corpus in one
        vector space and the queries in another and still return confident
        nonsense.
        """
        import numpy as np

        self._ids = [s["id"] for s in snippets]
        texts = [s["processed_text"] for s in snippets]
        logger.info("Encoding %d snippets with %s", len(texts), self.model_name)

        vecs = _encode_cached(texts, self.model_name)
        vecs = np.ascontiguousarray(np.asarray(vecs, dtype=np.float32))

        # Inner product == cosine only because the vectors are unit length.
        # BaselineEncoder already normalises when NORMALIZE_EMBEDDINGS is set;
        # this is the belt-and-braces pass for a future encoder that does not.
        if config.NORMALIZE_EMBEDDINGS:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)

        if _use_exact_numpy():
            # "Flat" is exact brute force, so a matmul is not an approximation
            # of it - it IS it, to the last bit. Taking this path deliberately
            # because importing faiss alongside torch SEGFAULTS this build.
            # See _use_exact_numpy for the measurement.
            self._index = _ExactIndex(vecs)
            logger.info("Built exact (numpy) index: %d vectors, dim %d",
                        self._index.ntotal, vecs.shape[1])
        else:
            import faiss

            self._index = faiss.index_factory(
                vecs.shape[1], config.FAISS_INDEX_FACTORY,
                faiss.METRIC_INNER_PRODUCT,
            )
            if not self._index.is_trained:  # IVF/PQ need a training pass
                self._index.train(vecs)
            self._index.add(vecs)
            logger.info("Built %s (faiss) index: %d vectors, dim %d",
                        config.FAISS_INDEX_FACTORY, self._index.ntotal,
                        vecs.shape[1])

        if self._index.ntotal != len(self._ids):
            raise ValueError(
                f"Index holds {self._index.ntotal} vectors for "
                f"{len(self._ids)} ids; positions would map to wrong ids."
            )

    @property
    def index(self):
        """The built FAISS index. Raises if ``build``/``load`` has not run."""
        if self._index is None:
            raise RuntimeError("DenseIndexBuilder used before build()/load()")
        return self._index

    def save(self, path: str) -> None:
        """Write the FAISS index and the id map side by side.

        Both artifacts or neither - a half-saved index is a silent scoring bug.
        """
        # TODO(corpus): faiss.write_index(self._index, path) + dump self._ids
        raise NotImplementedError("TODO(corpus): DenseIndexBuilder.save")

    def load(self, path: str) -> None:
        """Restore index + id map. Must be equivalent to a fresh ``build``.

        Validate that ``len(self._ids) == self._index.ntotal`` and fail loudly
        if not - that mismatch is exactly the failure this method exists to
        catch.
        """
        # TODO(corpus): implement.
        raise NotImplementedError("TODO(corpus): DenseIndexBuilder.load")

    @property
    def ids(self) -> list[CorpusId]:
        """Position -> corpus id mapping, in corpus order."""
        return self._ids


class BM25IndexBuilder(IndexBuilder):
    """Lexical BM25 index over tokenized snippets.

    TODO(corpus): implement.

    Build outline
    -------------
    1. Tokenize each ``processed_text``. Tokenization is the main lever here:
       plain ``.split()`` is a weak baseline for code. Consider splitting
       snake_case and camelCase into subtokens so a query saying "binary
       search" can match ``binary_search``, and consider keeping both the
       compound and its parts.
    2. ``BM25Okapi(corpus_tokens, k1=config.BM25_K1, b=config.BM25_B)``.
    3. Record ``self._ids`` in the same corpus order as the dense index.

    BM25 is what rescues exact identifier and API-name matches that the
    bi-encoder blurs away, so it earns its place despite being unglamorous.
    """

    def __init__(self) -> None:
        self._ids: list[CorpusId] = []
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []

    def build(self, snippets: list[ProcessedSnippet]) -> None:
        """Tokenize and index the corpus. See class docstring for the outline."""
        # TODO(corpus): implement.
        raise NotImplementedError("TODO(corpus): BM25IndexBuilder.build")

    def save(self, path: str) -> None:
        """Persist tokenized corpus + id map (pickle is fine; it's rebuildable)."""
        # TODO(corpus): implement.
        raise NotImplementedError("TODO(corpus): BM25IndexBuilder.save")

    def load(self, path: str) -> None:
        """Restore the tokenized corpus and rebuild the BM25 statistics."""
        # TODO(corpus): implement.
        raise NotImplementedError("TODO(corpus): BM25IndexBuilder.load")

    @property
    def ids(self) -> list[CorpusId]:
        """Position -> corpus id mapping, in corpus order."""
        return self._ids


def tokenize_code(text: str) -> list[str]:
    """Tokenize a code snippet for BM25.

    TODO(corpus): implement.

    Input : raw or processed snippet text.
    Output: lowercase token list.

    Shared by :class:`BM25IndexBuilder` (corpus side) and
    ``src.retrieval.bm25`` (query side). Both sides MUST use this same
    function - a tokenizer mismatch between index and query is a silent
    recall killer that no test will catch unless you write it.
    """
    raise NotImplementedError("TODO(corpus): tokenize_code")


def _cache_path(model_name: str) -> "Path":
    """Where the document-vector cache for ``model_name`` lives."""
    tag = model_name.replace("/", "__") + f"_w{config.ENCODER_WINDOW_CAP}"
    return config.CACHE_DIR / f"{tag}_docvecs_{config.CACHE_VERSION}.npz"


def _encode_cached(texts: list[str], model_name: str):
    """Encode ``texts`` as documents, reusing a content-hash disk cache.

    Why this exists: the corpus encode is ~10 minutes of CPU and it is
    IDENTICAL on every run, because the corpus is the same 8,765 documents in
    the same order on both splits. Re-paying it per evaluation is the single
    biggest avoidable cost in the loop, and DenseIndexBuilder's own build
    outline calls for going through the embedding cache.

    Keyed by sha1 of the exact text, namespaced by checkpoint, window cap and
    CACHE_VERSION. Normalisation is deliberately NOT part of the key: the
    vectors are stored exactly as the encoder returned them, and the caller
    normalises. A cached vector can therefore only be reused for a text that
    is byte-identical under the same checkpoint and window - which is the
    whole correctness condition.

    Degrades to a plain encode on any cache I/O failure: a slow run beats a
    failed one, and this box has already lost a read to an iCloud timeout.
    """
    import numpy as np

    from src.pipeline.baseline import BaselineEncoder

    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    cached: dict[str, Any] = {}
    path = _cache_path(model_name)
    if config.ENABLE_EMBEDDING_CACHE:
        try:
            if path.exists():
                with np.load(path) as z:
                    cached = {k: z[k] for k in z.files}
                logger.info("Document vector cache: %d entries at %s",
                            len(cached), path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s (%s); encoding from scratch",
                           path, exc)
            cached = {}

    todo = {k: t for k, t in zip(keys, texts) if k not in cached}
    if todo:
        logger.info("Encoding %d/%d documents (%d served from cache)",
                    len(todo), len(texts), len(texts) - len(todo))
        fresh = BaselineEncoder(model_name).encode(
            list(todo.values()), prompt_type="document",
            batch_size=config.BATCH_SIZE,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        )
        for k, v in zip(todo.keys(), np.asarray(fresh, dtype=np.float32)):
            cached[k] = v
        if config.ENABLE_EMBEDDING_CACHE:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **cached)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not write %s (%s)", path, exc)
    else:
        logger.info("All %d document vectors served from cache", len(texts))

    return np.stack([cached[k] for k in keys])


def _use_exact_numpy() -> bool:
    """True when the configured index is exact and we should skip faiss.

    WHY THIS EXISTS - faiss + torch SEGFAULTS this build.
    ------------------------------------------------------
    Importing faiss and then running a torch forward pass crashes the
    interpreter with SIGSEGV (exit 139) on this machine: faiss-cpu and torch
    each ship their own OpenMP runtime, and loading both is the classic
    duplicate-libomp crash on macOS. It killed two full test evaluations after
    the corpus index had already been built, with no Python traceback - the
    process simply vanished mid-search.

    Measured, three ways:

        default                          -> SIGSEGV during the torch encode
        faiss.omp_set_num_threads(1)     -> SIGSEGV (too late; the runtime is
                                            already initialised by then)
        OMP_NUM_THREADS=1 in the env     -> works

    The env-var fix works but pins torch to a single thread, and the encoder
    is the dominant cost of every run on this CPU-only box - an 8x slowdown on
    the expensive half to satisfy a library we do not need.

    ``FAISS_INDEX_FACTORY = "Flat"`` is exact brute-force inner product. A
    numpy matmul over unit vectors is not an approximation of that, it is the
    same computation, and it is the code path every number in experiments.md
    was measured with. So for "Flat" we skip faiss entirely and keep the
    encoder multi-threaded.

    Any other factory (IVF, HNSW, PQ) is a real approximate structure that
    numpy cannot stand in for, so those still go through faiss - and will need
    OMP_NUM_THREADS=1 set before torch is imported.
    """
    return config.FAISS_INDEX_FACTORY.strip().lower() == "flat"


class _ExactIndex:
    """Exact inner-product search with the subset of the faiss API we use.

    Deliberately minimal and faiss-shaped (``ntotal``, ``search`` returning
    ``(scores, positions)``) so DenseRetriever does not care which backend it
    got, and so swapping back to faiss is a one-line change if the OpenMP
    conflict is ever fixed upstream.
    """

    def __init__(self, vectors) -> None:
        self._vectors = vectors

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def search(self, queries, k: int):
        """Return ``(scores, positions)``, best first, like faiss does."""
        import numpy as np

        k = min(k, self.ntotal)
        n = queries.shape[0]
        scores = np.empty((n, k), dtype=np.float32)
        positions = np.empty((n, k), dtype=np.int64)
        # Chunked: a full queries x corpus score matrix is hundreds of MB on a
        # box that is already tight, and this buys nothing by being one call.
        for start in range(0, n, 256):
            sims = queries[start : start + 256] @ self._vectors.T
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            order = np.take_along_axis(sims, part, axis=1).argsort(axis=1)[:, ::-1]
            idx = np.take_along_axis(part, order, axis=1)
            positions[start : start + len(sims)] = idx
            scores[start : start + len(sims)] = np.take_along_axis(sims, idx, axis=1)
        return scores, positions
