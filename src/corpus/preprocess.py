"""Snippet normalization.  (owner: corpus)

APPS solutions are competition-style Python programs, and the bi-encoder's
context window is short, so what we keep here does affect NDCG.

But calibrate the effort - MEASURED, see data/inspection_report.md:

    snippets over the 254-token window   23.5%   (median  132 tokens)
    QUERIES  over the 254-token window   61.9%   (median  314 tokens)

Most snippets already fit. The truncation problem is overwhelmingly on the
query side, roughly 2.6x as often, and that is stage 1's territory
(src/query/preprocess.py), not this file's. An earlier version of this
docstring had it the other way round.

Ideas worth measuring (log each in experiments.md):
  - truncate to config.MAX_SNIPPET_TOKENS (head? tail? head+signature?)
  - strip comments/docstrings (config.STRIP_COMMENTS) - may remove noise, may
    remove the only natural-language bridge to the query. Test, don't assume.
  - keep function/class signatures and hoist them to the front
  - collapse whitespace and normalize indentation
  - chunk long files with overlap and pool the chunk embeddings
    (config.ENABLE_CHUNKING) - note this needs an id->chunks mapping, and the
    id-preservation rule in interfaces.py still applies at the snippet level
"""

from __future__ import annotations

from src import config
from src.interfaces import ProcessedSnippet, Snippet, SnippetProcessor


class NoOpSnippetProcessor(SnippetProcessor):
    """Passthrough baseline: ``processed_text`` is the original ``text``."""

    def process(self, snippet: Snippet) -> ProcessedSnippet:
        """Return the snippet unchanged, id preserved."""
        return {"id": snippet["id"], "processed_text": snippet["text"]}


class PrismSnippetProcessor(SnippetProcessor):
    """The real snippet preprocessor.

    TODO(corpus): implement. Replace the passthrough body below.

    Expected behaviour
    ------------------
    Input : {"id": str, "text": str} - text may be empty or many KB of code.
    Output: {"id": <same id, byte-for-byte>, "processed_text": str}

    Hard requirements
    -----------------
    * ``id`` MUST be preserved exactly. MTEB joins our results to its qrels by
      this string; a mangled id scores zero with no error message.
    * Never drop, merge, split or reorder snippets. ``process_batch`` output
      length and order must match its input - index positions are mapped back
      to ids by offset.
    * Never raise. Malformed code that fails to parse must fall back to the
      original text.
    * Deterministic.
    """

    def __init__(self) -> None:
        # TODO(corpus): set up whatever the transform needs (an AST-based
        # comment stripper, a tokenizer for length-aware truncation, ...).
        # Truncate in TOKENS, against config.MAX_SNIPPET_TOKENS, using the
        # tokenizer of config.DENSE_MODEL_NAME. The old character cap was a
        # proxy so crude it never fired: 4,000 chars against a limit the
        # tokenizer enforces at ~1,000 chars' worth of Python.
        pass

    def process(self, snippet: Snippet) -> ProcessedSnippet:
        """Normalize one snippet. See class docstring for the contract."""
        # TODO(corpus): replace this passthrough with the real implementation.
        return {"id": snippet["id"], "processed_text": snippet["text"]}


def get_snippet_processor() -> SnippetProcessor:
    """Return the processor selected by ``config.ENABLE_SNIPPET_PREPROCESSING``."""
    if config.ENABLE_SNIPPET_PREPROCESSING:
        return PrismSnippetProcessor()
    return NoOpSnippetProcessor()
