"""Category H - end-to-end pipeline and the MTEB contract.  (owner: eval)

Two jobs:

1. Verify our MTEB v2 adapters match the signatures MTEB will actually call.
   If these drift, everything else can be perfect and the evaluation still
   fails at minute one.
2. Run the pipeline end to end on the toy corpus and check output *shape*, not
   retrieval quality - quality is what run_eval.py measures.
"""

from __future__ import annotations

import inspect

import pytest

from src.pipeline.baseline import BaselineEncoder
from src.pipeline.mteb_compat import extract_texts, record_id, to_records
from src.pipeline.prism_search import PrismSearch


# =============================================================================
# MTEB v2 contract - these run offline and must always pass
# =============================================================================


def test_baseline_encoder_has_v2_encode_signature() -> None:
    """encode() accepts the v2 keyword arguments MTEB passes.

    v2 calls encode(inputs, task_metadata=..., hf_split=..., hf_subset=...,
    prompt_type=...). v1 passed a bare list[str]; this project targets v2.
    """
    params = inspect.signature(BaselineEncoder.encode).parameters
    for name in ("inputs", "task_metadata", "hf_split", "hf_subset", "prompt_type"):
        assert name in params, f"encode() is missing the v2 parameter {name!r}"


def test_prism_search_matches_search_protocol() -> None:
    """index() and search() carry the documented v2 SearchProtocol signatures."""
    index_params = inspect.signature(PrismSearch.index).parameters
    for name in ("corpus", "task_metadata", "hf_split", "hf_subset", "encode_kwargs"):
        assert name in index_params, f"index() is missing {name!r}"

    search_params = inspect.signature(PrismSearch.search).parameters
    for name in (
        "queries",
        "task_metadata",
        "hf_split",
        "hf_subset",
        "top_k",
        "encode_kwargs",
        "top_ranked",
    ):
        assert name in search_params, f"search() is missing {name!r}"


def test_search_before_index_raises() -> None:
    """Calling search() first is a programming error and should say so."""
    with pytest.raises(RuntimeError, match="before index"):
        PrismSearch().search([], top_k=10)


# =============================================================================
# MTEB v2 API regression tests
#
# These run only when mteb is installed, and they exist because all three
# assertions below were WRONG in the first draft of this scaffold - caught by
# pinning the version and checking against the real package rather than by
# reading the docs. The failure modes were: a silent fallback to `object`
# (losing the inherited similarity implementation), a silently-None ModelMeta,
# and a TypeError that would only have surfaced once evaluation started.
#
# If an mteb upgrade moves any of this, these fail in ~10 seconds instead of
# 90 minutes into a run.
# =============================================================================


def test_abs_encoder_import_path_is_still_valid() -> None:
    """AbsEncoder lives at mteb.models.abs_encoder - NOT mteb.AbsEncoder.

    Our resolver falls back to `object` silently, so a moved symbol would
    otherwise cost us the inherited similarity defaults with no error.
    """
    pytest.importorskip("mteb")
    from mteb.models.abs_encoder import AbsEncoder

    from src.pipeline.mteb_compat import resolve_abs_encoder

    resolved, is_real = resolve_abs_encoder()
    assert is_real, "resolver fell back to object - AbsEncoder has moved"
    assert resolved is AbsEncoder


def test_baseline_encoder_subclasses_real_abs_encoder() -> None:
    """The baseline genuinely inherits MTEB's similarity implementation."""
    pytest.importorskip("mteb")
    from mteb.models.abs_encoder import AbsEncoder

    assert isinstance(BaselineEncoder(), AbsEncoder)


def test_model_meta_actually_builds() -> None:
    """ModelMeta construction succeeds rather than silently returning None.

    It is a pydantic model with 17 required fields; omitting any one of them
    is a validation error, which _build_model_meta catches and swallows.
    """
    pytest.importorskip("mteb")
    meta = BaselineEncoder().mteb_model_meta
    assert meta is not None, "ModelMeta silently failed to build"
    assert str(meta.similarity_fn_name).lower().endswith("cosine")


def test_prism_search_signatures_match_real_search_protocol() -> None:
    """Our index()/search() accept every parameter MTEB will pass.

    `num_proc` is keyword-only and has no default in the protocol - omitting
    it raises TypeError the moment evaluation starts.
    """
    mteb = pytest.importorskip("mteb")
    protocol = mteb.SearchProtocol

    for method in ("index", "search"):
        expected = set(inspect.signature(getattr(protocol, method)).parameters)
        actual = set(inspect.signature(getattr(PrismSearch, method)).parameters)
        missing = expected - actual - {"self"}
        assert not missing, f"{method}() is missing MTEB parameters: {sorted(missing)}"


def test_apps_retrieval_task_resolves() -> None:
    """The graded task exists under the name we evaluate."""
    mteb = pytest.importorskip("mteb")
    from src import config

    assert mteb.get_task(config.MTEB_TASK_NAME) is not None


def test_mteb_is_v2_not_v1() -> None:
    """`evaluate` is v2; v1 exposed `MTEB(tasks).run(model)` instead."""
    mteb = pytest.importorskip("mteb")
    assert hasattr(mteb, "evaluate"), "installed mteb is v1 - this project needs v2"


# =============================================================================
# mteb_compat helpers - offline, fully implemented, must always pass
# =============================================================================


def test_extract_texts_from_batch_dicts() -> None:
    """The v2 DataLoader shape: batches of {"text": [...]}. """
    batches = [{"text": ["a", "b"]}, {"text": ["c"]}]
    assert extract_texts(batches) == ["a", "b", "c"]


def test_extract_texts_from_plain_strings() -> None:
    """A plain list[str] still works, so tests need not build a DataLoader."""
    assert extract_texts(["a", "b"]) == ["a", "b"]


def test_extract_texts_joins_title_and_body() -> None:
    """A non-empty title is prepended; dropping it would lose signal."""
    assert extract_texts([{"title": "T", "text": "B"}]) == ["T\n\nB"]


def test_extract_texts_skips_empty_title() -> None:
    """An empty title adds no separator."""
    assert extract_texts([{"title": "", "text": "B"}]) == ["B"]


def test_extract_texts_survives_mismatched_title_column() -> None:
    """A short title column must not silently truncate the batch.

    zip() would drop the untitled rows and shift every downstream id.
    """
    assert len(extract_texts([{"text": ["a", "b", "c"], "title": ["T"]}])) == 3


def test_to_records_from_column_dict() -> None:
    """Column-oriented input transposes to row-oriented records, in order."""
    records = to_records({"id": ["1", "2"], "text": ["a", "b"]})
    assert records == [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]


def test_to_records_from_row_list() -> None:
    """Row-oriented input passes through unchanged."""
    rows = [{"id": "1", "text": "a"}]
    assert to_records(rows) == rows


def test_record_id_prefers_id_then_underscore_id() -> None:
    """MTEB has used both `id` and `_id` across dataset versions."""
    assert record_id({"id": "x"}, 0) == "x"
    assert record_id({"_id": "y"}, 0) == "y"


def test_record_id_falls_back_to_position() -> None:
    """A malformed row must not abort the run; it just never matches a qrel."""
    assert record_id({}, 7) == "7"


# =============================================================================
# End-to-end - TODO(eval)
# =============================================================================


@pytest.mark.skip(reason="TODO(eval): needs the stage stubs implemented")
def test_pipeline_indexes_and_searches_toy_corpus() -> None:
    """index() then search() over the conftest corpus returns the right shape.

    Assert {query_id: {corpus_id: float}}, every corpus_id known, every query
    represented. Shape only - quality is run_eval.py's job.
    """


@pytest.mark.skip(reason="TODO(eval): needs the stage stubs implemented")
def test_pipeline_rejects_id_mangling_preprocessor() -> None:
    """A SnippetProcessor that changes ids raises at index() time.

    Guards the invariant that otherwise fails silently with a zero score.
    """


@pytest.mark.skip(reason="TODO(eval): needs the stage stubs implemented")
def test_pipeline_rejects_corpus_size_change() -> None:
    """A SnippetProcessor that drops a snippet raises at index() time."""


@pytest.mark.skip(reason="TODO(eval): needs the stage stubs implemented")
def test_top_ranked_restriction_is_honoured() -> None:
    """When MTEB passes top_ranked, results are restricted to those candidates."""


@pytest.mark.skip(reason="TODO(eval): implement DiskEmbeddingCache")
def test_cache_key_changes_with_text_model_and_version() -> None:
    """Any of text / model / CACHE_VERSION changing produces a different key.

    A key that misses one of these serves stale vectors and turns the
    experiment log into fiction.
    """


@pytest.mark.slow
@pytest.mark.skip(reason="TODO(eval): full MTEB run, minutes on CPU")
def test_run_eval_writes_results_json() -> None:
    """scripts/run_eval.py --pipeline baseline --limit 5 produces valid JSON."""
