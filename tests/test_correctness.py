"""Layer-1 correctness assertions for the stages we actually ship.

WHAT BELONGS HERE
-----------------
One fast, offline, deterministic check per KEPT stage, aimed at that stage's
most likely SILENT failure - the kind where the run completes, the results
file is well-formed, and the number is simply worse than it should be. This
project has been bitten by that class repeatedly (a wrong instruction prefix,
a mangled id, an all-NaN reranker that sorted as a no-op), and every one of
them would have been caught by an assertion that costs milliseconds.

The shipped stack is deliberately small, because steps 6, 7 and 9 were all
measured and gated out:

    arctic-embed-m  ->  exact inner-product index  ->  top-10
    with a content-hash embedding cache in front of the encoder

So these tests cover the encoder's prompt contract, id survival, the exact
index, batching equivalence, and the title+body join contract. Nothing here
loads a checkpoint or touches the network - stage behaviour is exercised
through fakes so the file stays in `pytest -m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.interfaces import ProcessedQuery


# =============================================================================
# The encoder's prompt contract
# =============================================================================
#
# arctic-embed-m takes an instruction prefix on the QUERY side only. Getting
# this wrong does not raise - it just scores worse, which is why it is the
# first thing asserted.


def test_query_prefix_is_applied_and_document_prefix_is_not() -> None:
    """The query side gets the instruction; the document side must not."""
    from src.pipeline.baseline import _prefix_for

    assert _prefix_for("PromptType.query") == config.QUERY_PROMPT_PREFIX
    assert _prefix_for("PromptType.document") == config.DOCUMENT_PROMPT_PREFIX
    assert _prefix_for(None) == ""

    # The active checkpoint is query-only. If someone swaps the model and
    # forgets the document prefix, this is the line that should fail.
    if "arctic" in config.DENSE_MODEL_NAME:
        assert config.QUERY_PROMPT_PREFIX, "arctic needs a query instruction"
        assert config.DOCUMENT_PROMPT_PREFIX == "", (
            "arctic is query-only; a document prefix silently costs accuracy"
        )


def test_retriever_does_not_prefix_twice() -> None:
    """DenseRetriever must delegate prefixing to the encoder, not re-apply it.

    Applying the instruction twice is silent and costs accuracy. The retriever
    passes prompt_type="query" and hands over the RAW query text; if it ever
    starts prepending the prefix itself the encoder would see it twice.
    """
    import inspect

    from src.retrieval import dense

    source = inspect.getsource(dense.DenseRetriever.retrieve_batch)
    assert "QUERY_PROMPT_PREFIX" not in source, (
        "DenseRetriever must not prepend the prefix itself; BaselineEncoder "
        "already does it via prompt_type"
    )
    assert 'prompt_type="query"' in source


# =============================================================================
# Ids are sacred
# =============================================================================


class _FakeIndexBuilder:
    """Minimal stand-in carrying an id map and an exact index."""

    def __init__(self, ids: list[str], vectors: np.ndarray) -> None:
        from src.corpus.index import _ExactIndex

        self._ids = ids
        self._index = _ExactIndex(vectors)

    @property
    def ids(self) -> list[str]:
        return self._ids

    @property
    def index(self):
        return self._index


class _FakeEncoder:
    """Returns pre-set unit vectors; records what it was asked to encode."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.seen: list[str] = []
        self.prompt_types: list[object] = []

    def encode(self, texts, prompt_type=None, **kwargs):  # noqa: ANN001
        texts = list(texts)
        self.seen.extend(texts)
        self.prompt_types.append(prompt_type)
        return self.vectors[: len(texts)]


def _unit(rows: list[list[float]]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_retrieved_ids_are_the_index_ids_byte_for_byte() -> None:
    """A snippet id must survive retrieval unchanged.

    MTEB joins results to qrels on this exact string. Mangle it - strip it,
    re-number it, coerce it to int - and the score drops to zero with no
    error anywhere.
    """
    from src.retrieval.dense import DenseRetriever

    docs = _unit([[1, 0], [0, 1], [1, 1]])
    ids = ["d1", "d2", "d3"]
    retriever = DenseRetriever(_FakeIndexBuilder(ids, docs))
    retriever._model = _FakeEncoder(_unit([[1, 0]]))

    ranked = retriever.retrieve_batch([ProcessedQuery(text="q", raw="q")], top_k=3)[0]
    returned = [cid for cid, _ in ranked]
    assert set(returned) <= set(ids), f"invented an id: {returned}"
    assert all(isinstance(c, str) for c in returned), "ids must stay strings"
    assert returned[0] == "d1", "nearest neighbour should rank first"


def test_scores_are_descending_and_no_padding_id_is_emitted() -> None:
    """Asking for more neighbours than exist must not emit a padding id.

    The faiss API returns -1 for padding positions, and ids[-1] is a REAL id
    and a wrong answer. The exact index clamps k instead, and this pins that.
    """
    from src.retrieval.dense import DenseRetriever

    docs = _unit([[1, 0], [0, 1]])
    retriever = DenseRetriever(_FakeIndexBuilder(["d1", "d2"], docs))
    retriever._model = _FakeEncoder(_unit([[1, 0]]))

    ranked = retriever.retrieve_batch([ProcessedQuery(text="q", raw="q")],
                                      top_k=50)[0]
    assert len(ranked) == 2, "cannot return more documents than are indexed"
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True), "results must be best-first"


def test_retrieve_matches_retrieve_batch() -> None:
    """Batching is an optimization, not a behaviour change."""
    from src.retrieval.dense import DenseRetriever

    docs = _unit([[1, 0], [0, 1], [1, 1]])
    queries = [ProcessedQuery(text="a", raw="a"), ProcessedQuery(text="b", raw="b")]
    qvecs = _unit([[1, 0], [0, 1]])

    batched_retriever = DenseRetriever(_FakeIndexBuilder(["d1", "d2", "d3"], docs))
    batched_retriever._model = _FakeEncoder(qvecs)
    batched = batched_retriever.retrieve_batch(queries, top_k=3)

    one_at_a_time = []
    for i, query in enumerate(queries):
        solo = DenseRetriever(_FakeIndexBuilder(["d1", "d2", "d3"], docs))
        solo._model = _FakeEncoder(qvecs[i : i + 1])
        one_at_a_time.append(solo.retrieve(query, top_k=3))

    for got, expected in zip(batched, one_at_a_time):
        assert [c for c, _ in got] == [c for c, _ in expected]
        np.testing.assert_allclose([s for _, s in got],
                                   [s for _, s in expected], atol=1e-6)


# =============================================================================
# The exact index IS brute force
# =============================================================================


def test_exact_index_matches_a_plain_matmul() -> None:
    """The numpy index must equal the definition it replaces.

    It stands in for FAISS "Flat", which is exact inner-product search. If it
    ever stops agreeing with a plain matmul, it is no longer the thing every
    number in experiments.md was measured with.
    """
    from src.corpus.index import _ExactIndex

    rng = np.random.default_rng(config.RANDOM_SEED)
    docs = rng.normal(size=(50, 16)).astype(np.float32)
    queries = rng.normal(size=(7, 16)).astype(np.float32)

    scores, positions = _ExactIndex(docs).search(queries, 5)
    reference = queries @ docs.T
    for i in range(len(queries)):
        expected = np.argsort(-reference[i])[:5]
        np.testing.assert_array_equal(positions[i], expected)
        np.testing.assert_allclose(scores[i], reference[i][expected], atol=1e-5)


# =============================================================================
# The title+body join contract
# =============================================================================


def test_title_join_is_identical_across_its_sites() -> None:
    """All join sites must produce byte-identical strings.

    `title` is empty on every row of this dataset, so the branch is dead - but
    the join is kept so the SearchProtocol path and the bare-encoder baseline
    index the same bytes. If the sites drift, the baseline and the full
    pipeline stop being comparable and every delta in experiments.md becomes
    meaningless. Remove all of them together or none.
    """
    from scripts.corpus_variants import corpus_text as variants_join
    from src.pipeline.prism_search import _corpus_text as pipeline_join

    cases = [
        {"text": "body only", "title": ""},
        {"text": "body", "title": "   "},
        {"text": "def f(): pass", "title": "Some Title"},
        {"text": "", "title": ""},
    ]
    for row in cases:
        assert pipeline_join(row) == variants_join(row), row

    # And the dead branch really is dead on this dataset's shape.
    assert pipeline_join({"text": "b", "title": ""}) == "b"


# =============================================================================
# The partition trap
# =============================================================================


def test_partition_column_is_never_used_to_filter() -> None:
    """Filtering the corpus on `partition` manufactures a large fake gain.

    Every query's gold document sits in its own partition, so filtering
    shrinks the candidate pool from 8,765 to 3,765 and the score jumps for
    entirely the wrong reason. Nothing in the shipped path may read it.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in list((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or '"partition"' not in stripped:
                continue
            offenders.append(f"{path.relative_to(root)}:{number}: {stripped}")
    assert not offenders, "partition referenced in shipped code:\n" + "\n".join(offenders)


# =============================================================================
# The submission artifact says what it is
# =============================================================================


def test_submission_artifact_records_the_full_pipeline() -> None:
    """The locked artifact must come from the full pipeline on the full test split.

    A --limit or --pipeline baseline run produces a file that looks entirely
    plausible; on the current stack it even carries the same NDCG, because
    every optional stage is measured-and-off. These three fields are the only
    way to tell them apart.
    """
    import json

    if not config.RESULTS_JSON.exists():
        pytest.skip("no submission artifact yet")
    payload = json.loads(config.RESULTS_JSON.read_text(encoding="utf-8"))
    assert payload["pipeline"] == "full", "artifact is not from the full pipeline"
    assert payload["split"] == "test", "artifact is not the test split"
    assert payload["smoke_run"] is False, "artifact came from a smoke run"
