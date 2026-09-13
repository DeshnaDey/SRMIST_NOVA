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


@pytest.mark.skip(reason="TODO(corpus): implement DenseIndexBuilder.build")
def test_dense_index_size_matches_corpus() -> None:
    """index.ntotal == len(snippets) == len(ids)."""


@pytest.mark.skip(reason="TODO(corpus): implement DenseIndexBuilder.build")
def test_dense_ids_are_in_corpus_order() -> None:
    """ids[i] is the id of the i-th input snippet. This is the whole mapping."""


@pytest.mark.skip(reason="TODO(corpus): implement DenseIndexBuilder")
def test_dense_embeddings_are_normalized_when_configured() -> None:
    """config.NORMALIZE_EMBEDDINGS gives unit-norm rows (cosine via inner product)."""


@pytest.mark.skip(reason="TODO(corpus): implement save/load")
def test_dense_save_load_roundtrip() -> None:
    """A loaded index returns identical results to the freshly built one."""


@pytest.mark.skip(reason="TODO(corpus): implement save/load")
def test_dense_load_rejects_id_map_mismatch() -> None:
    """len(ids) != index.ntotal must fail loudly - it is a silent scoring bug."""


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
