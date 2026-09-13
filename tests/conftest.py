"""Shared pytest fixtures.

Keep fixtures here rather than duplicating them per file - the tiny corpus
below is the common vocabulary for categories B through H, and tests read much
better when they all refer to the same five snippets.

Nothing here may download a model or hit the network. Tests that need a real
checkpoint belong behind the ``slow`` marker.
"""

from __future__ import annotations

import pytest

from src.interfaces import ProcessedQuery, ProcessedSnippet, Snippet


@pytest.fixture
def raw_snippets() -> list[Snippet]:
    """A five-document toy corpus, shaped like APPS entries.

    Deliberately includes the awkward cases: an empty document, an id that is
    not an integer, and two documents that share vocabulary so ranking order is
    actually meaningful.
    """
    return [
        {
            "id": "doc_1",
            "text": (
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        if arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    return -1\n"
            ),
        },
        {
            "id": "doc_2",
            "text": (
                "# Sort a list using bubble sort\n"
                "def bubble_sort(items):\n"
                "    for i in range(len(items)):\n"
                "        for j in range(len(items) - i - 1):\n"
                "            if items[j] > items[j + 1]:\n"
                "                items[j], items[j + 1] = items[j + 1], items[j]\n"
                "    return items\n"
            ),
        },
        {
            "id": "doc_3",
            "text": (
                "def fib(n):\n"
                '    """Return the nth Fibonacci number."""\n'
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            ),
        },
        # Empty document: preprocessing must not drop it, and retrieval must
        # not crash on it.
        {"id": "doc_4", "text": ""},
        # Non-numeric id with punctuation: ids are opaque strings, never parsed.
        {"id": "apps/train/0042", "text": "print(sum(int(x) for x in input().split()))"},
    ]


@pytest.fixture
def processed_snippets(raw_snippets: list[Snippet]) -> list[ProcessedSnippet]:
    """The toy corpus in post-preprocessing form (identity transform)."""
    return [{"id": s["id"], "processed_text": s["text"]} for s in raw_snippets]


@pytest.fixture
def corpus_ids(raw_snippets: list[Snippet]) -> list[str]:
    """Corpus ids in corpus order - the mapping every index must preserve."""
    return [s["id"] for s in raw_snippets]


@pytest.fixture
def sample_query() -> ProcessedQuery:
    """A processed query matching ``doc_1``."""
    text = "find an element in a sorted array efficiently"
    return ProcessedQuery(text=text, raw=text, category=None)


@pytest.fixture
def raw_queries() -> list[str]:
    """Raw query strings, including the degenerate ones that must not crash."""
    return [
        "find an element in a sorted array efficiently",
        "Write a Python function that sorts a list without using sorted().",
        "compute fibonacci numbers",
        "",  # empty query
        "   \n\t  ",  # whitespace only
        "read two integers from stdin and print their sum",
    ]


@pytest.fixture
def dense_ranking() -> list[tuple[str, float]]:
    """A dense ranked list: cosine-like scores in [0, 1], sorted descending."""
    return [("doc_1", 0.91), ("doc_3", 0.74), ("doc_2", 0.58), ("doc_4", 0.12)]


@pytest.fixture
def bm25_ranking() -> list[tuple[str, float]]:
    """A BM25 ranked list over the same corpus.

    Note the scale (unbounded, ~0-20) and the different ordering - that
    mismatch is exactly what fusion has to reconcile, and why naive score
    addition hands the ranking to BM25.
    """
    return [("doc_2", 18.4), ("doc_1", 11.2), ("apps/train/0042", 4.7)]
