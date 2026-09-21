"""Category C - index construction.  (owner: corpus)

Everything here is a placeholder until the builders exist. The theme is the
position -> corpus id mapping: an index that returns the right *positions* and
the wrong *ids* looks completely healthy and scores zero.
"""

from __future__ import annotations

import pytest

from src.corpus.index import BM25IndexBuilder, DenseIndexBuilder


def test_builders_expose_the_interface() -> None:
    """Both builders satisfy IndexBuilder before any of it is implemented."""
    for builder in (DenseIndexBuilder(), BM25IndexBuilder()):
        assert hasattr(builder, "build")
        assert hasattr(builder, "save")
        assert hasattr(builder, "load")
        assert builder.ids == []


# =============================================================================
# Dense index - TODO(corpus)
# =============================================================================


@pytest.fixture()
def fake_encode(monkeypatch, tmp_path):
    """Build the index without downloading a checkpoint.

    Patches the encoder behind DenseIndexBuilder and points the embedding
    cache at a tmp dir, so these stay offline and in `-m "not slow"`.
    Vectors are deliberately NOT unit length, so the normalisation assertion
    below is testing something.
    """
    import numpy as np

    from src import config
    from src.corpus import index as index_mod

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    def _fake(texts, model_name):  # noqa: ANN001
        # Distinct, non-unit vectors, one per text, stable across calls.
        return np.array([[float(len(t) + 1), 2.0, 3.0] for t in texts],
                        dtype=np.float32)

    monkeypatch.setattr(index_mod, "_encode_cached", _fake)
    return _fake


def _snips(n: int) -> list[dict]:
    return [{"id": f"d{i}", "processed_text": "x" * (i + 1)} for i in range(n)]


def test_dense_index_size_matches_corpus(fake_encode) -> None:
    """index.ntotal == len(snippets) == len(ids)."""
    builder = DenseIndexBuilder()
    builder.build(_snips(5))
    assert builder.index.ntotal == 5
    assert len(builder.ids) == 5


def test_dense_ids_are_in_corpus_order(fake_encode) -> None:
    """ids[i] is the id of the i-th input snippet. This is the whole mapping.

    An index that returns the right positions and the wrong ids looks
    completely healthy and scores zero.
    """
    snippets = _snips(6)
    builder = DenseIndexBuilder()
    builder.build(snippets)
    assert builder.ids == [s["id"] for s in snippets]


def test_dense_embeddings_are_normalized_when_configured(fake_encode) -> None:
    """NORMALIZE_EMBEDDINGS gives unit-norm rows (cosine via inner product)."""
    import numpy as np

    from src import config

    assert config.NORMALIZE_EMBEDDINGS, "this test assumes the shipped setting"
    builder = DenseIndexBuilder()
    builder.build(_snips(4))
    norms = np.linalg.norm(builder.index._vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)


def test_dense_build_rejects_id_map_mismatch(monkeypatch, tmp_path) -> None:
    """len(ids) != index.ntotal must fail loudly - it is a silent scoring bug."""
    import numpy as np

    from src import config
    from src.corpus import index as index_mod

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    # An encoder that returns the wrong number of rows is exactly how the id
    # map and the vectors drift apart.
    monkeypatch.setattr(index_mod, "_encode_cached",
                        lambda texts, model_name: np.ones((len(texts) - 1, 3),
                                                          dtype=np.float32))
    with pytest.raises(ValueError, match="ids"):
        DenseIndexBuilder().build(_snips(4))


@pytest.mark.skip(reason="save/load intentionally unimplemented: the "
                         "content-hash embedding cache supersedes it - a warm "
                         "rebuild is 0.7s (data/cache_rebuild_demo.json), so "
                         "persisting the index buys nothing and adds a second "
                         "artifact that can go stale against the corpus.")
def test_dense_save_load_roundtrip() -> None:
    """A loaded index returns identical results to the freshly built one."""


# =============================================================================
# BM25 index - TODO(corpus)
# =============================================================================


@pytest.mark.skip(reason="TODO(corpus): implement BM25IndexBuilder.build")
def test_bm25_index_size_matches_corpus() -> None:
    """One tokenized document per snippet, ids in corpus order."""


@pytest.mark.skip(reason="TODO(corpus): implement BM25IndexBuilder.build")
def test_bm25_handles_empty_document() -> None:
    """doc_4 is empty; it must index as an empty token list, not crash."""


@pytest.mark.skip(reason="TODO(corpus): implement tokenize_code")
def test_tokenizer_splits_snake_case() -> None:
    """``binary_search`` -> tokens including "binary" and "search".

    This is what lets the query "binary search" match the identifier.
    """


@pytest.mark.skip(reason="TODO(corpus): implement tokenize_code")
def test_tokenizer_splits_camel_case() -> None:
    """``binarySearch`` -> tokens including "binary" and "search"."""


@pytest.mark.skip(reason="TODO(corpus): implement tokenize_code")
def test_tokenizer_is_shared_by_index_and_query() -> None:
    """Index side and query side must call the same tokenize_code.

    A tokenizer mismatch destroys recall silently - nothing errors, the
    numbers are just quietly bad.
    """
