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

from src import config
from src.interfaces import CorpusId, IndexBuilder, ProcessedSnippet


class DenseIndexBuilder(IndexBuilder):
    """FAISS index over snippet embeddings.

    TODO(corpus): implement.

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
        """Encode and index the corpus. See class docstring for the outline."""
        # TODO(corpus): implement.
        raise NotImplementedError("TODO(corpus): DenseIndexBuilder.build")

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
