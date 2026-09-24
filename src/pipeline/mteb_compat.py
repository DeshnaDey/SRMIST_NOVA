"""Thin compatibility helpers for the MTEB v2 API.

WHY THIS FILE EXISTS
--------------------
MTEB v2 moved several symbols between modules across its 2.x releases, and its
``encode()`` contract changed substantially from v1: v1 passed a ``list[str]``,
v2 passes a ``DataLoader`` yielding batch dicts keyed by modality ("text",
"images", ...). Rather than scatter defensive imports and isinstance checks
through the pipeline, they are quarantined here.

The import paths below were confirmed against the pinned mteb 2.20.11. The
fallbacks are kept deliberately: they cost nothing, and each one is a path a
past 2.x release actually used, so they are cheap insurance against a future
move rather than untested speculation.
"""

from __future__ import annotations

from typing import Any, Iterable

# =============================================================================
# Locating AbsEncoder
# =============================================================================

#: Import paths tried, in order, when resolving the v2 encoder base class.
#: VERIFIED against mteb 2.20.11: the real location is
#: ``mteb.models.abs_encoder.AbsEncoder``. It is NOT re-exported as
#: ``mteb.AbsEncoder`` or ``mteb.models.AbsEncoder``, so those two paths alone
#: silently fall back to ``object`` and lose the inherited similarity defaults.
#: The extra paths stay as cheap insurance against a future move.
_ABS_ENCODER_PATHS: tuple[tuple[str, str], ...] = (
    ("mteb.models.abs_encoder", "AbsEncoder"),
    ("mteb.models", "AbsEncoder"),
    ("mteb", "AbsEncoder"),
)


def resolve_abs_encoder() -> tuple[type, bool]:
    """Return ``(base_class, is_real)`` for the MTEB v2 encoder base class.

    Subclassing the real ``AbsEncoder`` is worth doing: it supplies default
    ``similarity`` / ``similarity_pairwise`` implementations driven by
    ``ModelMeta.similarity_fn_name``, so the retrieval task scores pairs the
    way the checkpoint expects.

    Returns
    -------
    (type, bool)
        The resolved class and whether it is genuinely MTEB's. Falls back to
        ``object`` when MTEB is absent or has moved the symbol - our
        ``encode()`` satisfies the structural ``EncoderProtocol`` either way,
        so the pipeline still runs, just without the inherited similarity
        defaults.
    """
    import importlib

    for module_name, attr in _ABS_ENCODER_PATHS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        base = getattr(module, attr, None)
        if isinstance(base, type):
            return base, True
    return object, False


# =============================================================================
# Unpacking v2 encode() inputs
# =============================================================================


def extract_texts(inputs: Any) -> list[str]:
    """Flatten MTEB v2 ``encode()`` inputs into a plain list of strings.

    The v2 signature is ``encode(inputs: DataLoader[BatchedInput], ...)``,
    where each batch is a dict keyed by modality. This helper accepts that and
    the other shapes that turn up in tests and older code paths, so callers
    never have to care:

    * a ``DataLoader`` / any iterable of batch dicts -> ``{"text": [str, ...]}``
    * an iterable of record dicts -> ``{"text": str, "title": str}``
    * a plain ``list[str]``
    * a HuggingFace ``Dataset``

    Title handling
    --------------
    Retrieval corpora often carry a ``title`` alongside ``text``. When both are
    present and the title is non-empty they are joined with a blank line -
    dropping the title silently loses signal on datasets that use it.

    Returns
    -------
    list[str]
        One string per input item, in input order. Order is load-bearing: the
        caller maps embedding row ``i`` back to corpus position ``i``.
    """
    texts: list[str] = []

    for item in _iter_items(inputs):
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.extend(_texts_from_mapping(item))
        else:  # pragma: no cover - defensive
            texts.append(str(item))

    return texts


def _iter_items(inputs: Any) -> Iterable[Any]:
    """Yield the top-level items of ``inputs`` without loading a Dataset twice."""
    if isinstance(inputs, str):
        return [inputs]
    return inputs


def _texts_from_mapping(item: dict[str, Any]) -> list[str]:
    """Extract the text(s) from one batch dict or one record dict."""
    body = item.get("text", item.get("body", ""))
    title = item.get("title", "")

    # Batched form: {"text": [...], "title": [...]}
    if isinstance(body, (list, tuple)):
        titles = title if isinstance(title, (list, tuple)) else [""] * len(body)
        # Guard against a title column shorter than the text column rather
        # than letting zip() silently truncate the batch.
        if len(titles) != len(body):
            titles = [""] * len(body)
        return [_join_title_body(t, b) for t, b in zip(titles, body)]

    # Single-record form: {"text": "...", "title": "..."}
    return [_join_title_body(title if isinstance(title, str) else "", body)]


def _join_title_body(title: str, body: Any) -> str:
    """Combine a title and body into one encodable string.

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
    body_str = "" if body is None else str(body)
    title_str = "" if title is None else str(title)
    if title_str.strip():
        return f"{title_str}\n\n{body_str}"
    return body_str


# =============================================================================
# Unpacking corpus / query datasets for SearchProtocol
# =============================================================================


def to_records(dataset: Any) -> list[dict[str, Any]]:
    """Normalize a corpus or query dataset into a list of record dicts.

    MTEB hands ``index()`` and ``search()`` a HuggingFace ``Dataset``, but
    tests hand them plain lists, and older paths hand them a column dict. All
    three collapse to ``[{"id": ..., "text": ..., ...}, ...]`` here.

    Returns
    -------
    list[dict]
        Records in dataset order. Order matters: it defines the position ->
        corpus id mapping that every index depends on.
    """
    # Column-oriented dict: {"id": [...], "text": [...]}
    if isinstance(dataset, dict):
        keys = list(dataset.keys())
        if not keys:
            return []
        length = len(dataset[keys[0]])
        return [{k: dataset[k][i] for k in keys} for i in range(length)]

    records: list[dict[str, Any]] = []
    for row in dataset:
        if isinstance(row, dict):
            records.append(dict(row))
        else:  # pragma: no cover - defensive
            records.append({"id": str(len(records)), "text": str(row)})
    return records


def record_id(record: dict[str, Any], position: int) -> str:
    """Return a record's corpus/query id as a string.

    MTEB has used ``id`` and ``_id`` across dataset versions. Falling back to
    the positional index keeps a malformed row from aborting the run - it will
    simply never match a qrel.
    """
    for key in ("id", "_id", "doc_id", "query_id"):
        if key in record and record[key] is not None:
            return str(record[key])
    return str(position)
