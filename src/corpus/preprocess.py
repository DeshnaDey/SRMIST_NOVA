"""Snippet normalization.  (owner: corpus)

APPS solutions are long, competition-style Python programs. The bi-encoder's
context window is short, so what we keep and what we throw away here has a
large effect on NDCG - larger, usually, than the choice of fusion strategy.

Ideas worth measuring (log each in experiments.md):
  - truncate to config.MAX_SNIPPET_CHARS (head? tail? head+signature?)
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
        # Prefer tokenizer-aware truncation over character counts if you can
        # afford it - config.MAX_SNIPPET_CHARS is a crude proxy.
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
