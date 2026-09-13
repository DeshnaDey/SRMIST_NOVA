"""Stage contracts for the PRISM retrieval pipeline.

THIS FILE IS THE TEAM'S API. Everything else is an implementation detail.

Read this before writing code, and treat it as frozen unless the whole team
agrees to a change. Four people are building against these signatures at the
same time; if you change one, you break someone else's half-finished branch.

Pipeline shape
--------------
    raw query ─► QueryProcessor ─────► ProcessedQuery ─┐
                                                       ├─► Retriever ─► ranked
    raw snippets ─► SnippetProcessor ─► ProcessedSnippet┘        │
                                                                 ▼
                                                    Fusion ─► Reranker ─► top-k

Rules that hold at every stage
------------------------------
1. **IDs are sacred.** A snippet's ``id`` must survive preprocessing, indexing,
   retrieval, fusion and reranking byte-for-byte. MTEB joins our output back to
   its qrels by that string; mangle it and the score silently drops to zero.
2. **Stages are pure.** Given the same input and config, return the same output.
   No hidden global state - the embedding cache is the one sanctioned exception.
3. **Never raise on bad input.** A malformed query or an empty snippet should
   degrade (return the input unchanged, return an empty list) rather than kill
   a two-hour evaluation run at query 9,000.
4. **Scores are "higher is better"** everywhere, and ranked lists are always
   returned sorted descending by score.

Implementing a stage
--------------------
Subclass the relevant ABC, implement the abstract method, and leave the
existing docstring in place. A no-op passthrough implementation is a perfectly
good first commit - it keeps the pipeline green while you iterate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

# =============================================================================
# Type aliases - use these instead of bare `str` / `float` so signatures read
# unambiguously and mypy can catch an id/text mix-up.
# =============================================================================

QueryId = str
CorpusId = str
Score = float

#: A single scored candidate. ALWAYS ``(corpus_id, score)`` in that order.
ScoredDoc = tuple[CorpusId, Score]

#: A ranked candidate list, sorted descending by score. Index 0 is rank 1.
RankedList = list[ScoredDoc]

#: What MTEB v2's ``SearchProtocol.search()`` must return:
#: ``{query_id: {corpus_id: score}}``. Note this is a *mapping*, not a list -
#: MTEB sorts it internally, but we sort before emitting anyway so that
#: debugging output and the scored output agree.
RetrievalOutput = dict[QueryId, dict[CorpusId, Score]]


# =============================================================================
# Data carried between stages
# =============================================================================


class Snippet(TypedDict):
    """A raw corpus document exactly as it arrives from the dataset.

    ``id``   - the corpus id assigned by MTEB/CoIR. Opaque; never parse it.
    ``text`` - the code snippet body.
    """

    id: CorpusId
    text: str


class ProcessedSnippet(TypedDict):
    """A snippet after preprocessing.

    ``id`` MUST equal the ``id`` of the :class:`Snippet` it came from. The
    contract is a *transform of text*, not a filter: preprocessing may never
    drop, merge, split or reorder corpus entries, because the index positions
    are matched back to ids by offset.
    """

    id: CorpusId
    processed_text: str


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    """A natural-language query after cleaning and (optional) classification.

    Attributes
    ----------
    text:
        The cleaned query handed to the retrievers. Downstream code uses this
        and never touches ``raw``.
    raw:
        The original, untouched query string. Kept for logging and for
        reranking, which sometimes scores better against the raw phrasing.
    category:
        Optional label from ``config.QUERY_CATEGORIES``, or ``None`` if
        classification is disabled or the classifier abstained. Consumers MUST
        tolerate ``None`` and MUST tolerate a label they don't recognise.
    metadata:
        Free-form bag for anything a stage wants to pass along (extracted
        keywords, detected language, expansion terms...). Nothing in the
        pipeline may *require* a key here - it is strictly additive, so that
        adding a key never breaks another stage.
    """

    text: str
    raw: str
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Stage 1 - query preprocessing                    (owner: query)
# =============================================================================


class QueryProcessor(ABC):
    """Cleans a raw natural-language query and optionally labels it.

    Implementations live in ``src/query/preprocess.py``.
    """

    @abstractmethod
    def process(self, raw_query: str) -> ProcessedQuery:
        """Clean one query.

        Parameters
        ----------
        raw_query:
            The query string as it comes from the dataset. May be empty,
            may contain markdown, code fences, boilerplate ("Write a function
            that..."), or unicode punctuation.

        Returns
        -------
        ProcessedQuery
            ``.raw`` set to ``raw_query`` verbatim and ``.text`` set to the
            cleaned form. If cleaning would empty the string, fall back to
            ``raw_query`` - an unclean query retrieves far better than an
            empty one.

        Notes
        -----
        Must not raise. Must be deterministic.
        """

    def process_batch(self, raw_queries: list[str]) -> list[ProcessedQuery]:
        """Clean many queries, preserving order and length.

        Override only if you can batch more efficiently than the default loop
        (e.g. a batched classifier forward pass).
        """
        return [self.process(q) for q in raw_queries]


# =============================================================================
# Stage 2 - snippet preprocessing + indexing       (owner: corpus)
# =============================================================================


class SnippetProcessor(ABC):
    """Normalizes a code snippet before it is indexed.

    Implementations live in ``src/corpus/preprocess.py``.
    """

    @abstractmethod
    def process(self, snippet: Snippet) -> ProcessedSnippet:
        """Transform one snippet's text, preserving its id.

        Parameters
        ----------
        snippet:
            ``{"id": ..., "text": ...}``. ``text`` may be empty or enormous
            (APPS solutions run to thousands of lines).

        Returns
        -------
        ProcessedSnippet
            ``processed_text`` may be truncated, comment-stripped,
            whitespace-normalized, or augmented with a natural-language
            summary - but ``id`` MUST be carried through unchanged.

        Notes
        -----
        Must not raise: return the original text on any parse failure.
        Never returns ``None`` and never drops the snippet.
        """

    def process_batch(self, snippets: list[Snippet]) -> list[ProcessedSnippet]:
        """Transform many snippets. Output length and order MUST match input."""
        return [self.process(s) for s in snippets]


class IndexBuilder(ABC):
    """Builds a searchable structure over the processed corpus.

    Implementations live in ``src/corpus/index.py``. One builder per retrieval
    modality: the FAISS vector index and the BM25 index are separate builders
    sharing this interface.
    """

    @abstractmethod
    def build(self, snippets: list[ProcessedSnippet]) -> None:
        """Index the corpus. Called exactly once before any search.

        Implementations must retain the id ordering so that an internal
        integer position can be mapped back to a :data:`CorpusId`.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the built index so a rerun can skip the build."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a previously saved index. Must be equivalent to ``build``."""


# =============================================================================
# Stage 3 - retrieval, fusion, reranking           (owner: retrieval)
# =============================================================================


class Retriever(ABC):
    """Scores the corpus against one query and returns the best candidates.

    Implementations live in ``src/retrieval/`` - ``dense.py`` (bi-encoder over
    FAISS) and ``bm25.py`` (lexical) both satisfy this interface, which is what
    lets the fusion stage treat them interchangeably.
    """

    @abstractmethod
    def retrieve(self, query: ProcessedQuery, top_k: int) -> RankedList:
        """Return the ``top_k`` best candidates for ``query``.

        Parameters
        ----------
        query:
            The output of :meth:`QueryProcessor.process`. Use ``query.text``;
            ``query.category`` and ``query.metadata`` are optional signals.
        top_k:
            Maximum candidates to return. Returning fewer is allowed (a small
            corpus, or an aggressive filter); returning more is a contract
            violation and will distort fusion.

        Returns
        -------
        RankedList
            ``[(corpus_id, score), ...]`` sorted descending by score, length
            ``<= top_k``. Scores are "higher is better" but are NOT comparable
            across retrievers - BM25 scores and cosine similarities live on
            different scales, which is exactly why RRF (rank-based) is the
            default fusion strategy.

        Notes
        -----
        Return ``[]`` rather than raising if the index is empty or the query
        degenerates to nothing.
        """

    def retrieve_batch(
        self, queries: list[ProcessedQuery], top_k: int
    ) -> list[RankedList]:
        """Retrieve for many queries. Override to exploit batched encoding."""
        return [self.retrieve(q, top_k) for q in queries]


@runtime_checkable
class FusionStrategy(Protocol):
    """Merges several ranked lists of the same corpus into one.

    A Protocol rather than an ABC so a plain function can be used directly.
    Implementations live in ``src/retrieval/fusion.py``.
    """

    def __call__(self, rankings: list[RankedList], top_k: int) -> RankedList:
        """Fuse ``rankings`` into a single ranked list.

        Parameters
        ----------
        rankings:
            One :data:`RankedList` per retriever, in a fixed, caller-defined
            order (the pipeline passes ``[dense, bm25]``). Any list may be
            empty. Lists overlap partially - a document present in one and
            absent from another is normal and must be handled, not skipped.
        top_k:
            Length cap on the fused output.

        Returns
        -------
        RankedList
            Sorted descending, length ``<= top_k``, with NO duplicate
            ``corpus_id``. Deduplication is the fusion stage's job.
        """
        ...


class Reranker(ABC):
    """Reorders a candidate list using a stronger, slower model.

    Implementations live in ``src/retrieval/rerank.py``. Typically a
    cross-encoder that scores (query, snippet) jointly - far more accurate than
    the bi-encoder and far too slow to run over the whole corpus, which is why
    it only ever sees the fused top-N.
    """

    @abstractmethod
    def rerank(
        self,
        query: ProcessedQuery,
        candidates: RankedList,
        top_k: int,
    ) -> RankedList:
        """Reorder ``candidates`` by relevance to ``query``.

        Parameters
        ----------
        query:
            The processed query. ``query.raw`` is often the better input to a
            cross-encoder than the aggressively cleaned ``query.text``.
        candidates:
            Output of the fusion stage: ``[(corpus_id, score), ...]``. The
            incoming scores come from different retrievers and are generally
            NOT comparable - treat this as a *set* with a suggested order.
        top_k:
            Length cap on the returned list.

        Returns
        -------
        RankedList
            A permutation of a subset of ``candidates``, sorted descending by
            the reranker's own scores, length ``<= top_k``. MUST NOT invent
            corpus ids that were not in ``candidates``.

        Notes
        -----
        On model failure, return ``candidates[:top_k]`` unchanged. A degraded
        ranking beats a crashed evaluation.
        """


# =============================================================================
# Stage 4 - embedding cache                        (owner: eval)
# =============================================================================


@runtime_checkable
class EmbeddingCache(Protocol):
    """Content-addressed store for embeddings, keyed by hash of text + config.

    Implementations live in ``src/versioning/cache.py``. The point is that
    re-running an evaluation after changing only the reranker should not
    re-encode 100k snippets on CPU.
    """

    def get(self, key: str) -> Any | None:
        """Return the cached array for ``key``, or ``None`` on a miss.

        Must never raise on a corrupt cache entry - treat it as a miss.
        """
        ...

    def put(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``. Overwrites silently."""
        ...

    def make_key(self, text: str, model_name: str, **extra: Any) -> str:
        """Derive a stable cache key.

        Must incorporate ``config.CACHE_VERSION``, the model name, and anything
        else that changes the embedding - otherwise a preprocessing change
        silently serves stale vectors and the experiment log becomes fiction.
        """
        ...
