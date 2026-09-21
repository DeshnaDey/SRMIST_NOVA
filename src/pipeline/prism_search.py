"""The full PRISM pipeline as an MTEB v2 ``SearchProtocol`` implementation.

This is the object that produces the submitted numbers. Implementing
``SearchProtocol`` (rather than handing MTEB a bare encoder) is what lets our
own preprocessing, hybrid retrieval, fusion and reranking sit *inside* the
official evaluation, so ``appsretrieval_results.json`` reflects the real system.

THE WIRING BELOW IS COMPLETE. The stages it calls are stubs. That is
deliberate: as each workstream fills in its stub, the full pipeline starts
working with no changes to this file. Nobody needs to edit this module to ship
their own stage - which is the point, since all four of us would be editing it
at once.

    index():   corpus ─► SnippetProcessor ─► DenseIndexBuilder
                                          └► BM25IndexBuilder   (if enabled)

    search():  queries ─► QueryProcessor ─► DenseRetriever ─┐
                                         └► BM25Retriever ──┴► Fusion
                                                                 │
                                                                 ▼
                                                    Reranker ─► {qid: {cid: score}}
"""

from __future__ import annotations

import logging
from typing import Any

from src import config
from src.corpus.index import BM25IndexBuilder, DenseIndexBuilder
from src.corpus.preprocess import get_snippet_processor
from src.interfaces import (
    CorpusId,
    ProcessedQuery,
    RankedList,
    RetrievalOutput,
    Snippet,
)
from src.pipeline.baseline import _build_model_meta
from src.pipeline.mteb_compat import record_id, to_records
from src.query.preprocess import get_query_processor
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import get_fusion_strategy
from src.retrieval.rerank import get_reranker

logger = logging.getLogger(__name__)


class PrismSearch:
    """Hybrid retrieve-fuse-rerank pipeline behind MTEB v2's SearchProtocol.

    MTEB calls ``index()`` once per (split, subset) and then ``search()`` with
    the queries for that split.

    Structural typing
    -----------------
    ``SearchProtocol`` is a ``typing.Protocol``, so matching the method
    signatures is all that is required - there is no base class to inherit.
    The signatures below mirror the documented v2 contract exactly, keyword-only
    arguments included; MTEB passes them by keyword.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Parameters
        ----------
        model_name:
            Bi-encoder checkpoint; defaults to ``config.DENSE_MODEL_NAME``.
        """
        self.model_name = model_name or config.DENSE_MODEL_NAME

        # Stage 1 + 2 processors (no-op passthroughs until their flags flip).
        self.query_processor = get_query_processor()
        self.snippet_processor = get_snippet_processor()

        # Built during index().
        self.dense_index: DenseIndexBuilder | None = None
        self.bm25_index: BM25IndexBuilder | None = None
        self.dense_retriever: DenseRetriever | None = None
        self.bm25_retriever: BM25Retriever | None = None

        #: {corpus_id: processed_text}, needed by the cross-encoder - ranked
        #: lists carry only ids.
        self.snippet_lookup: dict[CorpusId, str] = {}

        self.fuse = get_fusion_strategy()
        self.reranker: Any = None  # built in index(), needs snippet_lookup

        #: MTEB names the result directory from this, so it must NOT be None.
        #: Leaving it None raises deep inside mteb's result cache -
        #: ``TypeError: unsupported operand type(s) for /: 'PosixPath' and
        #: 'NoneType'`` in get_task_result_path - and it raises AFTER the
        #: whole evaluation has run, so you lose the entire pass. The bare
        #: encoder path never hit this because BaselineEncoder builds a
        #: ModelMeta of its own.
        self.mteb_model_meta = _build_model_meta(self.model_name)

    # =========================================================================
    # SearchProtocol.index
    # =========================================================================

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        num_proc: int | None = None,
    ) -> None:
        """Preprocess and index the corpus. Called once before ``search``.

        Parameters
        ----------
        corpus:
            The corpus dataset. Rows carry an id (``id`` or ``_id``) and
            ``text``, optionally ``title``.
        task_metadata, hf_split, hf_subset:
            Identify what is being indexed. Kept for logging and for
            cache-key scoping.
        encode_kwargs:
            Forwarded from ``mteb.evaluate(..., encode_kwargs=...)``.
        num_proc:
            Worker count MTEB requests for dataset-side work. Accepted and
            currently ignored - our indexing is single-process. It MUST stay
            in the signature: mteb 2.20.11 passes it by keyword, so dropping
            it raises TypeError the moment evaluation starts.

        Notes
        -----
        Returns ``None`` - the built indexes live on ``self``.
        """
        encode_kwargs = encode_kwargs or {}
        records = to_records(corpus)
        logger.info("Indexing %d snippets (split=%s)", len(records), hf_split)

        # --- Stage 2a: normalize snippet text, preserving ids ----------------
        snippets: list[Snippet] = [
            {"id": record_id(rec, i), "text": _corpus_text(rec)}
            for i, rec in enumerate(records)
        ]
        processed = self.snippet_processor.process_batch(snippets)

        # The id-preservation contract is load-bearing: every index maps
        # integer positions back to ids by offset, so a processor that dropped
        # or reordered a row would shift every id silently and score ~0.
        # Cheap to check, impossible to debug later.
        if len(processed) != len(snippets):
            raise ValueError(
                f"SnippetProcessor changed corpus size: {len(snippets)} in, "
                f"{len(processed)} out. Preprocessing must be a text transform, "
                "not a filter (see src/interfaces.py)."
            )
        for original, result in zip(snippets, processed):
            if result["id"] != original["id"]:
                raise ValueError(
                    f"SnippetProcessor mangled a corpus id: {original['id']!r} "
                    f"became {result['id']!r}. Ids must survive byte-for-byte."
                )

        self.snippet_lookup = {p["id"]: p["processed_text"] for p in processed}

        # --- Stage 2b: build the indexes -------------------------------------
        self.dense_index = DenseIndexBuilder(model_name=self.model_name)
        self.dense_index.build(processed)
        self.dense_retriever = DenseRetriever(
            self.dense_index, model_name=self.model_name
        )

        if config.ENABLE_BM25:
            self.bm25_index = BM25IndexBuilder()
            self.bm25_index.build(processed)
            self.bm25_retriever = BM25Retriever(self.bm25_index)

        # --- Stage 3d: reranker (needs the text lookup) ----------------------
        self.reranker = get_reranker(self.snippet_lookup)

    # =========================================================================
    # SearchProtocol.search
    # =========================================================================

    def search(
        self,
        queries: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        top_k: int = 10,
        encode_kwargs: dict[str, Any] | None = None,
        top_ranked: dict[str, list[str]] | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutput:
        """Retrieve, fuse and rerank for every query.

        Parameters
        ----------
        queries:
            The query dataset; rows carry an id and ``text``.
        top_k:
            How many results MTEB wants per query. We retrieve much deeper
            internally (``config.TOP_K_DENSE``) and narrow down - retrieving
            only ``top_k`` would leave the reranker nothing to reorder.
        top_ranked:
            Present for reranking-style tasks: a precomputed candidate list per
            query that we must restrict to. ``None`` for AppsRetrieval, but
            honoured so this class also works on rerank tasks.
        num_proc:
            As in ``index`` - accepted, ignored, and required in the signature
            because MTEB passes it by keyword.

        Returns
        -------
        RetrievalOutput
            ``{query_id: {corpus_id: score}}``. Scores are "higher is better";
            MTEB sorts internally, and we emit in sorted order anyway so that
            debug output and scored output agree.
        """
        if self.dense_retriever is None:
            raise RuntimeError("search() called before index()")

        encode_kwargs = encode_kwargs or {}
        records = to_records(queries)

        if config.SMOKE_TEST_QUERY_LIMIT is not None:
            records = records[: config.SMOKE_TEST_QUERY_LIMIT]
            logger.warning(
                "SMOKE_TEST_QUERY_LIMIT=%d - partial run, NOT a reportable score",
                config.SMOKE_TEST_QUERY_LIMIT,
            )

        query_ids = [record_id(rec, i) for i, rec in enumerate(records)]
        raw_texts = [str(rec.get("text", "")) for rec in records]

        # --- Stage 1: clean the queries --------------------------------------
        processed_queries = self.query_processor.process_batch(raw_texts)

        # --- Stage 3: retrieve, fuse, rerank ---------------------------------
        results: RetrievalOutput = {}
        dense_hits = self.dense_retriever.retrieve_batch(
            processed_queries, config.TOP_K_DENSE
        )

        for qid, query, dense_ranked in zip(query_ids, processed_queries, dense_hits):
            ranked = self._fuse_and_rerank(query, dense_ranked, top_k)

            if top_ranked is not None and qid in top_ranked:
                allowed = set(top_ranked[qid])
                ranked = [(cid, s) for cid, s in ranked if cid in allowed]

            results[qid] = {cid: float(score) for cid, score in ranked[:top_k]}

        return results

    # =========================================================================
    # internals
    # =========================================================================

    def _fuse_and_rerank(
        self,
        query: ProcessedQuery,
        dense_ranked: RankedList,
        top_k: int,
    ) -> RankedList:
        """Run one query's candidates through fusion and reranking.

        Split out of ``search`` so the per-query logic can be unit-tested with
        hand-built ranked lists, without building an index (see tests F and G).
        """
        # Fusion is only meaningful with a second ranked list to fuse against.
        if config.ENABLE_BM25 and self.bm25_retriever is not None:
            bm25_ranked = self.bm25_retriever.retrieve(query, config.TOP_K_BM25)
            candidates = self.fuse([dense_ranked, bm25_ranked], config.TOP_K_FUSED)
        else:
            candidates = dense_ranked[: config.TOP_K_FUSED]

        # Ask the reranker for at least what MTEB wants, and at least enough to
        # compute NDCG@10 - a shallower list silently truncates the metric.
        final_k = max(top_k, config.TOP_K_FINAL)
        return self.reranker.rerank(query, candidates, final_k)


def _corpus_text(record: dict[str, Any]) -> str:
    """Join a corpus row's title and body into the text we index.

    Mirrors the title handling in ``mteb_compat.extract_texts`` so the
    SearchProtocol path and the bare-encoder baseline index identical strings -
    otherwise the baseline and the full pipeline aren't comparable, and the
    experiments.md deltas mean nothing.

    DEAD CODE ON THIS DATASET - measured, kept deliberately.
    The `title` column of CoIR-Retrieval/apps is empty for every row: 0/8,765
    corpus entries and 0/5,000 queries are non-empty, one distinct value ("").
    So the title branch never fires here and this always returns the body.

    Not removed, because it is the mechanism that keeps the bare-encoder
    baseline and the SearchProtocol path encoding byte-identical strings. If
    one side drops it the two stop being comparable and every experiments.md
    delta between them becomes meaningless. Delete it only if you delete it
    from ALL of: src/pipeline/mteb_compat.py, src/pipeline/prism_search.py,
    scripts/inspect_data.py.
    """
    body = str(record.get("text", "") or "")
    title = str(record.get("title", "") or "")
    if title.strip():
        return f"{title}\n\n{body}"
    return body
