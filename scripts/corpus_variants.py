#!/usr/bin/env python3
"""Measure corpus-side preprocessing variants on recall@100, one lever at a time.

STEP 7 asks three separate questions about the document side, and they must be
measured separately or a combined number tells you nothing about which change
earned it:

  A. SIGNATURE/COMMENT AUGMENTATION - surface the little natural language a
     snippet has (def/class lines, # comments) at the front, where it is not
     competing with 500 tokens of loop body.
  B. starter_code - index meta_information.starter_code, which the pipeline
     has never fed the encoder.
  C. CHUNKING - split the snippets that exceed the 510-token budget and score
     a document by its best-matching chunk, so the tail stops being invisible.

WHAT THIS DATASET DOES TO THE BRIEF'S PREMISE
---------------------------------------------
Two of the brief's assumptions do not survive contact with the corpus, and the
variants below are shaped around what is actually there:

* **Extraction would destroy the corpus.** These are competitive-programming
  solutions, not library code: only 20.7% carry a `#` comment, 5.8% any
  triple-quote, and 32.6% have NO def/class line or comment at all. Extracting
  "signatures, docstrings and comments" leaves a median of 8 tokens against a
  raw median of 132, and leaves a third of the corpus empty. So variant A
  AUGMENTS - it prepends the distilled line and keeps the body - rather than
  replacing. Replacing was measured as a non-starter before it was built.

* **Whitespace normalisation is an exact no-op.** Verified on 1,500 documents:
  the wordpiece tokenizer already discards indentation, so normalising
  whitespace yields BYTE-IDENTICAL token ids, hence identical embeddings and
  identical recall. It is not a weak lever, it is not a lever - and it is not
  given an arm here, because an arm would burn a 15-minute encode to
  reproduce the control to 5 decimal places.

WHY EACH ARM IS CHEAP
---------------------
Embeddings are cached per DOCUMENT by content hash, not per run. A variant
that rewrites 3,401 of 8,765 documents therefore encodes 3,401 documents and
reuses the rest, instead of re-encoding the whole corpus. The control's
vectors are imported straight from the BM25 diagnostic's cache, so the control
arm costs nothing at all and still has to reproduce the logged 0.6870.

Query embeddings are untouched by every variant in this file - STEP 7 changes
only the document side - so they are read from cache once and shared.

Usage
-----
    python scripts/corpus_variants.py --split train
    python scripts/corpus_variants.py --split train --variants base,starter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from scripts.bm25_diagnostic import load_block  # noqa: E402

logger = logging.getLogger("corpus_variants")

#: Lines that carry what little natural language a solution has.
_SIG_RE = re.compile(r"(?m)^[ \t]*(?:def|class)[ \t]+.*$")
_COMMENT_RE = re.compile(r"(?m)#.*$")


def corpus_text(row: dict[str, Any], body: str | None = None) -> str:
    """Join a row into the string the encoder sees.

    DEAD-JOIN CONTRACT SITE (see guide.md). Remove every site together or
    none. The title+body join is reproduced EXACTLY as the baseline and the
    BM25 diagnostic do it, including the dead branch. `title` is empty on all 8,765
    rows, so the join always takes the body path - but it is kept byte-for-byte
    because the pipeline's three join sites must stay identical to each other
    (see guide.md). Variants substitute a rewritten ``body``; none of them
    touches the join itself.
    """
    text = row.get("text", "") if body is None else body
    body_s = str(text or "")
    title = str(row.get("title", "") or "")
    return f"{title}\n\n{body_s}" if title.strip() else body_s


# -- variants -----------------------------------------------------------------
# Each returns the list of texts representing one document. One element is the
# normal case; chunking is the reason the contract is a list.


def v_base(row: dict[str, Any]) -> list[str]:
    """Control: exactly what the pipeline indexes today."""
    return [corpus_text(row)]


def v_starter(row: dict[str, Any]) -> list[str]:
    """Append ``meta_information.starter_code`` to the body.

    Appended rather than prepended: the signature is a few tokens and the body
    is the substance, and putting it last means a document that fits the budget
    is unchanged in its first 500 tokens, so any measured difference is the
    starter code itself rather than a reshuffle of what got truncated.
    """
    sc = str(((row.get("meta_information") or {}).get("starter_code") or "")).strip()
    body = str(row.get("text", "") or "")
    return [corpus_text(row, f"{body}\n\n{sc}" if sc else body)]


#: Hard cap on the augmentation header, in characters.
#:
#: MEASURED THE HARD WAY. An uncapped header duplicates every def line and
#: comment in the document, which roughly doubles its length and pushes a
#: large minority of documents up against ENCODER_WINDOW_CAP (2,048 tokens).
#: Attention is quadratic, so a document at the cap costs ~240x a median
#: 132-token snippet: the uncapped arm spent over three hours of wall clock
#: without finishing 5,905 documents, against 247 seconds for 3,401 in the
#: starter arm. The cap keeps the arm testing its actual hypothesis - that a
#: little natural language helps MORE at offset 0 than buried mid-body - at a
#: cost proportional to the signal rather than to the noise around it.
_AUGMENT_HEADER_CHARS: int = 300


def v_augment(row: dict[str, Any]) -> list[str]:
    """Prepend the document's own signatures and comments as a short header.

    NON-DESTRUCTIVE on purpose - see the module docstring. The body is kept
    whole; the distilled lines are merely repeated at the front, capped at
    ``_AUGMENT_HEADER_CHARS``. The bet is positional: a def line 400 tokens
    deep competes with the loop body around it, and the same line at offset 0
    does not.
    """
    body = str(row.get("text", "") or "")
    parts = _SIG_RE.findall(body) + _COMMENT_RE.findall(body)
    header = " ".join(p.strip() for p in parts if p.strip())[:_AUGMENT_HEADER_CHARS]
    return [corpus_text(row, f"{header}\n\n{body}" if header else body)]


def _chunk_tokens(text: str, tok: Any, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of ``size`` content tokens.

    Cut on the tokenizer's own character offsets so each chunk is a genuine
    substring of the source - the same reason the query truncation probe does
    it this way, and it keeps a document that fits the budget byte-identical
    to the control instead of round-tripping it through a lossy decode.
    """
    enc = tok(text, add_special_tokens=False, truncation=False,
              return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if len(offs) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(offs), step):
        window = offs[start : start + size]
        if not window:
            break
        out.append(text[window[0][0] : window[-1][1]])
        if start + size >= len(offs):
            break
    return out


def make_v_chunk(tok: Any, size: int, overlap: int) -> Callable[[dict], list[str]]:
    """Chunk only the documents that exceed the budget; others are untouched.

    A document is scored by its BEST chunk (max cosine), which is the standard
    reduction and the only one that makes sense here: the gold snippet is
    relevant because some part of it answers the query, not because its average
    window does.
    """
    def v_chunk(row: dict[str, Any]) -> list[str]:
        return _chunk_tokens(corpus_text(row), tok, size, overlap)
    return v_chunk


# -- embedding cache ----------------------------------------------------------


class HashEmbeddingCache:
    """Content-hash keyed embedding store, so variants only encode what changed.

    Without this, every arm re-encodes all 8,765 documents (~14 min each) even
    though `starter` rewrites 3,401 of them and `chunk` rewrites 587. With it,
    an arm costs only its own deltas and the whole file runs in a fraction of
    one baseline encode.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as z:
                self.vecs = {k: z[k] for k in z.files}
            logger.info("Loaded %d cached document vectors", len(self.vecs))

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def seed(self, texts: list[str], vecs: np.ndarray) -> None:
        """Adopt an existing aligned (texts, vectors) pair, e.g. the control."""
        added = 0
        for t, v in zip(texts, vecs):
            k = self.key(t)
            if k not in self.vecs:
                self.vecs[k] = v
                added += 1
        logger.info("Seeded %d vectors from an existing cache", added)

    def encode_missing(self, texts: list[str], encoder: Any) -> float:
        todo = sorted({self.key(t): t for t in texts if self.key(t) not in self.vecs}.items())
        if not todo:
            logger.info("All %d texts already cached", len(texts))
            return 0.0
        logger.info("Encoding %d new texts (of %d)", len(todo), len(texts))
        started = time.perf_counter()
        vecs = encoder.encode([t for _, t in todo], prompt_type="document")
        for (k, _), v in zip(todo, np.asarray(vecs, dtype=np.float32)):
            self.vecs[k] = v
        return time.perf_counter() - started

    def matrix(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vecs[self.key(t)] for t in texts])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.vecs)
        logger.info("Saved %d vectors -> %s", len(self.vecs), self.path.name)


def gold_rank_chunked(
    qry: np.ndarray, chunks: np.ndarray, owner: np.ndarray, n_docs: int,
    gold: np.ndarray,
) -> np.ndarray:
    """0-based rank of each query's gold document, scoring a doc by its best chunk.

    ``owner[i]`` is the document index that chunk ``i`` belongs to, and chunks
    are laid out in document order so the per-document max is a single
    ``reduceat`` over segment boundaries.
    """
    bounds = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    ranks = np.empty(len(qry), dtype=np.int32)
    for s in range(0, len(qry), 256):
        sims = qry[s : s + 256] @ chunks.T
        doc_scores = np.maximum.reduceat(sims, bounds, axis=1)
        g = gold[s : s + 256]
        gs = doc_scores[np.arange(len(doc_scores)), g][:, None]
        ranks[s : s + len(sims)] = (doc_scores > gs).sum(axis=1)
    return ranks


VARIANTS: dict[str, str] = {
    "base": "control - what the pipeline indexes today",
    "starter": "append meta_information.starter_code",
    "augment": "prepend def/class lines and # comments, keep the body",
    "chunk": "split >budget documents into overlapping windows, score by best chunk",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--variants", default="base,starter,chunk,augment")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk length in tokens; default = the snippet budget.")
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--out", type=Path,
                        default=config.PROJECT_ROOT / "data" / "corpus_variants.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    from src.pipeline.baseline import BaselineEncoder, _load_sentence_transformer

    corpus, queries, qrels = load_block(args.split)
    pos = {str(r["id"]): i for i, r in enumerate(corpus)}
    keep, gold = [], []
    for i, row in enumerate(queries):
        docs = qrels.get(str(row["id"])) or {}
        if not docs:
            continue
        did = next(iter(docs))
        if did in pos:
            keep.append(i)
            gold.append(pos[did])
    queries = [queries[i] for i in keep]
    gold_idx = np.array(gold, dtype=np.int32)
    logger.info("%s: %d corpus, %d queries with a gold doc",
                args.split, len(corpus), len(queries))

    cache_dir = config.DATA_DIR / "embed_cache_diag"
    tag = config.DENSE_MODEL_NAME.replace("/", "__") + f"_w{config.ENCODER_WINDOW_CAP}"
    qry_path = cache_dir / f"{tag}_{args.split}_queries.npy"
    legacy_q = cache_dir / f"{tag}_queries.npy"
    encoder = BaselineEncoder()

    if qry_path.exists():
        qvecs = np.load(qry_path)
    elif legacy_q.exists() and len(np.load(legacy_q, mmap_mode="r")) == len(queries):
        qvecs = np.load(legacy_q)
        logger.info("Reusing %s", legacy_q.name)
    else:
        logger.info("Encoding %d queries (once; every variant shares them)", len(queries))
        qvecs = np.asarray(encoder.encode([str(r["text"]) for r in queries],
                                          prompt_type="query"), dtype=np.float32)
        np.save(qry_path, qvecs)
    logger.info("Query vectors %s", qvecs.shape)

    cache = HashEmbeddingCache(cache_dir / f"{tag}_doc_hash_cache.npz")
    base_texts = [corpus_text(r) for r in corpus]
    doc_path = cache_dir / f"{tag}_corpus.npy"
    if doc_path.exists():
        cache.seed(base_texts, np.load(doc_path))

    tok = _load_sentence_transformer(config.DENSE_MODEL_NAME).tokenizer
    size = args.chunk_size or config.max_snippet_tokens()
    builders: dict[str, Callable[[dict], list[str]]] = {
        "base": v_base,
        "starter": v_starter,
        "augment": v_augment,
        "chunk": make_v_chunk(tok, size, args.chunk_overlap),
    }

    results: list[dict[str, Any]] = []
    control_hit: np.ndarray | None = None
    k = args.top_k

    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        build = builders[name]
        per_doc = [build(r) for r in corpus]
        texts = [t for chunks in per_doc for t in chunks]
        owner = np.repeat(np.arange(len(corpus)),
                          [len(c) for c in per_doc]).astype(np.int32)
        changed = sum(1 for a, b in zip(per_doc, base_texts)
                      if not (len(a) == 1 and a[0] == b))
        enc_s = cache.encode_missing(texts, encoder)
        cache.save()

        ranks = gold_rank_chunked(qvecs, cache.matrix(texts), owner, len(corpus), gold_idx)
        hit = ranks < k
        row: dict[str, Any] = {
            "variant": name,
            "description": VARIANTS[name],
            "documents_changed": changed,
            "n_chunks": int(len(texts)),
            "recall_at_100": round(float(hit.mean()), 5),
            "recall_at_10": round(float((ranks < 10).mean()), 5),
            "median_gold_rank": int(np.median(ranks)),
            "encode_seconds": round(enc_s, 1),
        }
        if control_hit is None:
            control_hit = hit
        else:
            row["delta_vs_base"] = round(float(hit.mean() - control_hit.mean()), 5)
            row["queries_lost"] = int((control_hit & ~hit).sum())
            row["queries_gained"] = int((~control_hit & hit).sum())
            # Restrict to the documents this variant actually rewrote: a lever
            # that touches 6.7% of the corpus is invisible in a pooled mean.
            touched = np.array([not (len(per_doc[g]) == 1 and per_doc[g][0] == base_texts[g])
                                for g in gold_idx])
            if touched.any():
                row["on_affected_queries"] = {
                    "n": int(touched.sum()),
                    "base_recall_at_100": round(float(control_hit[touched].mean()), 5),
                    "variant_recall_at_100": round(float(hit[touched].mean()), 5),
                }
        results.append(row)
        logger.info("%-8s recall@100 %.5f  (changed %d docs, %d chunks, %.0fs)",
                    name, row["recall_at_100"], changed, len(texts), enc_s)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "split": args.split, "model": config.DENSE_MODEL_NAME, "top_k": k,
            "seed": config.RANDOM_SEED, "snippet_budget": config.max_snippet_tokens(),
            "chunk_size": size, "chunk_overlap": args.chunk_overlap,
            "n_queries": len(queries), "n_documents": len(corpus),
            "variants": results,
        }, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"  CORPUS VARIANTS — {args.split}, {len(queries)} queries, top-{k}")
    print("=" * 78)
    print(f"  {'variant':9s} {'docs chg':>8s} {'chunks':>7s} {'recall@100':>11s} {'delta':>9s}")
    for r in results:
        d = f"{r['delta_vs_base']:+.5f}" if "delta_vs_base" in r else "control"
        print(f"  {r['variant']:9s} {r['documents_changed']:>8d} {r['n_chunks']:>7d} "
              f"{r['recall_at_100']:>11.5f} {d:>9s}")
    print("=" * 78)
    print(f"  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
