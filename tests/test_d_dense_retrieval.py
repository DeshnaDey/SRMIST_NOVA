"""Category D - dense bi-encoder retrieval.  (owner: retrieval)

Tests that need a real checkpoint carry ``@pytest.mark.slow`` so the default
``pytest -m "not slow"`` run stays offline and fast. The logic tests below use
a fake index with hand-written vectors and a fake encoder, so they exercise
the real DenseRetriever without downloading anything.

These were skipped as ``TODO(retrieval): implement…`` until STEP 10 landed
``DenseIndexBuilder`` and ``DenseRetriever``. Leaving them skipped afterwards
is worse than not having written them: the suite reports green while the
shipped retrieval path is untested.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.interfaces import ProcessedQuery
from tests.test_correctness import _FakeEncoder, _FakeIndexBuilder, _unit


def _retriever(ids: list[str], docs: np.ndarray, qvecs: np.ndarray):
    from src.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(_FakeIndexBuilder(ids, docs))
    retriever._model = _FakeEncoder(qvecs)
    return retriever


QUERY = ProcessedQuery(text="q", raw="q")


def test_returns_at_most_top_k() -> None:
    """Returning more than top_k distorts fusion downstream."""
    docs = _unit([[1, 0], [0, 1], [1, 1], [1, 0.5]])
    got = _retriever(["a", "b", "c", "d"], docs, _unit([[1, 0]])).retrieve(QUERY, 2)
    assert len(got) == 2


def test_results_are_sorted_descending() -> None:
    """Ranked lists are always sorted descending by score."""
    docs = _unit([[1, 0], [0, 1], [1, 1]])
    got = _retriever(["a", "b", "c"], docs, _unit([[1, 0]])).retrieve(QUERY, 3)
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True)


def test_returns_known_corpus_ids_only() -> None:
    """Every returned id came from the indexed corpus."""
    ids = ["a", "b", "c"]
    docs = _unit([[1, 0], [0, 1], [1, 1]])
    got = _retriever(ids, docs, _unit([[0.3, 0.9]])).retrieve(QUERY, 3)
    assert {cid for cid, _ in got} <= set(ids)


def test_filters_faiss_padding_positions() -> None:
    """Padding must never be dereferenced into ids[-1].

    FAISS pads with -1 when the index holds fewer than top_k vectors, and
    ids[-1] is a real id and a wrong answer. The exact index clamps k instead;
    either way the contract is that only real documents come back.
    """
    docs = _unit([[1, 0], [0, 1]])
    got = _retriever(["a", "b"], docs, _unit([[1, 0]])).retrieve(QUERY, 10)
    assert len(got) == 2
    assert {cid for cid, _ in got} == {"a", "b"}


def test_empty_index_returns_empty_list() -> None:
    """Degrade, don't raise."""
    empty = np.zeros((0, 2), dtype=np.float32)
    got = _retriever([], empty, _unit([[1, 0]])).retrieve(QUERY, 5)
    assert got == []


def test_no_queries_returns_no_results() -> None:
    """An empty batch is legal and must not raise."""
    docs = _unit([[1, 0]])
    assert _retriever(["a"], docs, _unit([[1, 0]])).retrieve_batch([], 5) == []


def test_batch_matches_sequential_results() -> None:
    """retrieve_batch is an optimization, not a behaviour change."""
    docs = _unit([[1, 0], [0, 1], [1, 1]])
    qvecs = _unit([[1, 0], [0, 1]])
    queries = [ProcessedQuery(text="a", raw="a"), ProcessedQuery(text="b", raw="b")]

    batched = _retriever(["a", "b", "c"], docs, qvecs).retrieve_batch(queries, 3)
    solo = [_retriever(["a", "b", "c"], docs, qvecs[i : i + 1]).retrieve(q, 3)
            for i, q in enumerate(queries)]

    for got, expected in zip(batched, solo):
        assert [c for c, _ in got] == [c for c, _ in expected]
        np.testing.assert_allclose([s for _, s in got],
                                   [s for _, s in expected], atol=1e-6)


def test_query_prompt_prefix_is_applied() -> None:
    """The encoder is asked for the QUERY prompt type, and gets the raw text.

    E5/BGE/GTE/arctic-family models need their instruction and degrade quietly
    without it. The retriever must not prepend it itself - that would apply it
    twice, which is equally silent.
    """
    from src import config

    docs = _unit([[1, 0]])
    encoder = _FakeEncoder(_unit([[1, 0]]))
    retriever = _retriever(["a"], docs, _unit([[1, 0]]))
    retriever._model = encoder
    retriever.retrieve(ProcessedQuery(text="find a heap", raw="find a heap"), 1)

    assert encoder.prompt_types == ["query"], encoder.prompt_types
    assert encoder.seen == ["find a heap"], (
        "the retriever must hand over raw text; the encoder applies the prefix"
    )
    assert not encoder.seen[0].startswith(config.QUERY_PROMPT_PREFIX)


@pytest.mark.slow
@pytest.mark.skip(reason="needs a downloaded checkpoint; covered end-to-end by "
                         "the full test run in experiments.md")
def test_retrieves_semantically_relevant_snippet() -> None:
    """"find an element in a sorted array" ranks doc_1 (binary_search) first."""
