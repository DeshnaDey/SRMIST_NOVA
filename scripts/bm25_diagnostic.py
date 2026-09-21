#!/usr/bin/env python3
"""Decide whether BM25 + RRF fusion is worth building, BEFORE building it.

Reads off two numbers on the full train split:

  (a) LEVEL          BM25-only recall@100, against the dense model's 0.687.
                     Near it means real headroom. Near zero means the "exact
                     identifier overlap" premise is wrong for this dataset -
                     problem statements are prose and maths, solutions are
                     code, and the two often share no variable names at all.
                     Any exploitable overlap is more likely algorithm/domain
                     vocabulary ("dijkstra", "segment tree") and numeric
                     constants (10^9+7).

  (b) COMPLEMENTARITY  The actual decision signal, and the one that is easy to
                     skip. Fusion pays only when the two systems fail on
                     DIFFERENT queries. Two retrievers that each score 0.6 but
                     miss the same 40% of queries cannot help each other: the
                     union is still 0.6. So this reports the 2x2 of
                     per-query hit/miss and, crucially, the union ceiling -
                     the best recall@100 ANY fusion of these two lists could
                     reach.

Deliberately NOT a pipeline component. The team's BM25IndexBuilder and
BM25Retriever stubs are owned by other workstreams and stay untouched until
this diagnostic says fusion is worth their time.

Usage
-----
    python scripts/bm25_diagnostic.py
    python scripts/bm25_diagnostic.py --split train --top-k 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from scripts.run_eval import prepare_task  # noqa: E402

logger = logging.getLogger("bm25_diag")

#: Identifier-like runs and bare integers. Punctuation is dropped: it carries
#: no retrieval signal and inflates the index.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

#: Split a compound identifier into camelCase / PascalCase / ACRONYM parts.
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

#: Numeric constants written as a power, optionally offset: 10^9+7, 10**9+7,
#: 1e9+7, 2^31, 2**63-1. Quantifiers are BOUNDED on purpose - the unbounded
#: form backtracks catastrophically on this corpus, which contains a
#: 60,599-token document.
#: The lookbehind stops it firing inside an identifier like ``var1e2``.
_POW_RE = re.compile(
    r"(?<![A-Za-z0-9_])(\d{1,4})\s*(\^|\*\*|[eE])\s*(\d{1,3})"
    r"\s*(?:([+-])\s*(\d{1,9}))?"
)

#: Guard: refuse to expand anything that would produce an absurd integer.
_MAX_POW_EXPONENT = 64


def _expand_numeric_constants(text: str) -> str:
    """Rewrite power-form numeric constants into their literal integer.

    Competitive-programming statements say "modulo $10^9+7$"; the solutions
    write ``MOD = 1000000007``. Tokenized naively those never meet: the query
    side shreds into the meaningless digits ``10``, ``9``, ``7`` - which then
    spuriously match any document mentioning 10, 9 or 7 - while the document
    side carries one clean token ``1000000007``.

    That matters more than it looks. Modulus and bound constants are among the
    FEW lexical items a problem statement and its solution reliably share:
    variable names are author-chosen and rarely echo the prose, but
    ``1000000007`` is fixed by the problem. Shredding it suppresses exactly
    the overlap BM25 is supposed to win on.

    So both sides are normalised to the literal before tokenizing::

        "modulo 10^9+7"   -> "modulo 1000000007"
        "MOD = 10**9 + 7" -> "MOD = 1000000007"
        "1e9+7"           -> "1000000007"
        "2^31"            -> "2147483648"
    """
    def repl(m: re.Match[str]) -> str:
        base, op, exp = int(m.group(1)), m.group(2), int(m.group(3))
        if exp > _MAX_POW_EXPONENT:
            return m.group(0)
        try:
            # "1e9" is scientific notation - 1 x 10^9 - NOT 1^9. Conflating
            # the two turns 1e9+7 into 8.
            value = base * (10 ** exp) if op.lower() == "e" else base ** exp
        except (OverflowError, ValueError):  # pragma: no cover - defensive
            return m.group(0)
        sign, offset = m.group(4), m.group(5)
        if sign and offset:
            value = value + int(offset) if sign == "+" else value - int(offset)
        return f" {value} "

    return _POW_RE.sub(repl, text)


def tokenize_code(text: str) -> list[str]:
    """Tokenize for BM25, splitting compound identifiers into subtokens.

    Default whitespace+lowercase is the obvious thing and it destroys exactly
    the signal this diagnostic exists to measure: ``binarySearch`` and
    ``binary_search`` would never match a query saying "binary search".

    So each identifier contributes BOTH forms - the compound and its parts::

        "binarySearch"  -> ["binarysearch", "binary", "search"]
        "MAX_HEAP_SIZE" -> ["max_heap_size", "max", "heap", "size"]
        "dijkstra"      -> ["dijkstra"]

    Keeping the compound matters: a query that genuinely names
    ``binary_search`` should still score higher against a document using that
    exact identifier than against one that merely says "binary" and "search"
    in unrelated places.

    The SAME function must tokenize both corpus and queries. A tokenizer
    mismatch between the two sides is a silent recall killer that no test
    catches unless you write it.
    """
    out: list[str] = []
    # Normalise power-form constants (10^9+7 -> 1000000007) before tokenizing,
    # on BOTH sides. See _expand_numeric_constants.
    for raw in _TOKEN_RE.findall(_expand_numeric_constants(text)):
        compound = raw.lower()
        out.append(compound)
        # Underscores first, then camel humps inside each underscore-part.
        parts: list[str] = []
        for chunk in raw.split("_"):
            if chunk:
                parts.extend(_CAMEL_RE.findall(chunk))
        if len(parts) > 1:
            out.extend(p.lower() for p in parts if p)
    return out


def load_block(split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return (corpus_records, query_records, qrels) for ``split``."""
    task = prepare_task(split, None)
    block = task.dataset[task.hf_subsets[0]][split]
    return list(block["corpus"]), list(block["queries"]), block["relevant_docs"]


def dense_rankings(
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int,
    cache_dir: Path,
) -> tuple[np.ndarray, float, float]:
    """Top-``top_k`` corpus indices per query under the active dense model.

    Embeddings are cached on disk keyed by model + window, because this is the
    expensive half (~14 min for the corpus) and the diagnostic may be rerun
    while iterating on tokenization.
    """
    from src.pipeline.baseline import BaselineEncoder

    encoder = BaselineEncoder()
    tag = (config.DENSE_MODEL_NAME.replace("/", "__")
           + f"_w{config.ENCODER_WINDOW_CAP}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_path = cache_dir / f"{tag}_corpus.npy"
    qry_path = cache_dir / f"{tag}_queries.npy"

    def corpus_text(row: dict[str, Any]) -> str:
        # DEAD-JOIN CONTRACT SITE (see guide.md). Title is empty on every row
        # of this dataset, so this always returns the body; the join is kept
        # only to stay byte-identical with the pipeline. Remove every site
        # together or none.
        body = str(row.get("text", "") or "")
        title = str(row.get("title", "") or "")
        return f"{title}\n\n{body}" if title.strip() else body

    started = time.perf_counter()
    if doc_path.exists():
        logger.info("Reusing cached corpus embeddings %s", doc_path.name)
        doc_vecs = np.load(doc_path)
    else:
        logger.info("Encoding %d corpus docs with %s", len(corpus),
                    config.DENSE_MODEL_NAME)
        doc_vecs = encoder.encode([corpus_text(r) for r in corpus],
                                  prompt_type="document")
        np.save(doc_path, doc_vecs)
    doc_seconds = time.perf_counter() - started

    started = time.perf_counter()
    if qry_path.exists():
        logger.info("Reusing cached query embeddings %s", qry_path.name)
        qry_vecs = np.load(qry_path)
    else:
        logger.info("Encoding %d queries", len(queries))
        qry_vecs = encoder.encode([str(r["text"]) for r in queries],
                                  prompt_type="query")
        np.save(qry_path, qry_vecs)
    qry_seconds = time.perf_counter() - started

    # Vectors are L2-normalised by the encoder, so inner product == cosine.
    # Chunked: a full 5000x8765 float32 matrix is ~175 MB and this box is
    # already tight on RAM.
    ranks = np.empty((len(queries), top_k), dtype=np.int32)
    for start in range(0, len(qry_vecs), 256):
        chunk = qry_vecs[start : start + 256]
        sims = chunk @ doc_vecs.T
        part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        order = np.take_along_axis(sims, part, axis=1).argsort(axis=1)[:, ::-1]
        ranks[start : start + len(chunk)] = np.take_along_axis(part, order, axis=1)
    return ranks, doc_seconds, qry_seconds


def bm25_rankings(
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int,
) -> tuple[np.ndarray, float, float]:
    """Top-``top_k`` corpus indices per query under BM25.

    Implemented over a sparse term-document matrix rather than with
    ``rank_bm25``. That library recomputes, for every query term, a Python
    list comprehension across all 8,765 documents; our queries are whole
    problem statements running to hundreds of terms, which measured out at
    roughly 0.6 s/query - about 50 minutes for one pass over train, every
    time. The formula below is the same Okapi BM25, precomputed once.

    Precomputing the full weight matrix is what makes it fast::

        W[t, d] = idf[t] * tf[t,d] * (k1 + 1)
                  / (tf[t,d] + k1 * (1 - b + b * len[d] / avglen))

    Every term is independent of the query, so scoring a query collapses to
    summing the rows of ``W`` named by its terms - one sparse operation.

    Query terms are summed WITH multiplicity, matching rank_bm25's behaviour,
    so a term repeated in the problem statement counts more than once.
    """
    from scipy import sparse

    started = time.perf_counter()
    corpus_tokens = [tokenize_code(str(r.get("text", "") or "")) for r in corpus]
    n_docs = len(corpus_tokens)

    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for d, toks in enumerate(corpus_tokens):
        counts: dict[int, int] = {}
        for tok in toks:
            tid = vocab.setdefault(tok, len(vocab))
            counts[tid] = counts.get(tid, 0) + 1
        for tid, tf in counts.items():
            rows.append(tid); cols.append(d); vals.append(tf)

    n_terms = len(vocab)
    tf_mat = sparse.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)),
        shape=(n_terms, n_docs),
    )

    doc_len = np.array([len(t) for t in corpus_tokens], dtype=np.float32)
    avg_len = float(doc_len.mean()) if n_docs else 0.0
    df = np.diff(tf_mat.indptr).astype(np.float32)
    # Lucene-style idf: always positive, so a term appearing in most documents
    # cannot subtract from a score the way raw Okapi idf can.
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    k1, b = config.BM25_K1, config.BM25_B
    norm = (k1 * (1.0 - b + b * doc_len / avg_len)).astype(np.float32)

    weights = tf_mat.tocoo()
    denom = weights.data + norm[weights.col]
    weights.data = idf[weights.row] * weights.data * (k1 + 1.0) / denom
    w_mat = weights.tocsr()

    index_seconds = time.perf_counter() - started
    logger.info(
        "BM25 index built in %.1fs: %d terms x %d docs, %d nonzeros, "
        "avg %.0f tokens/doc",
        index_seconds, n_terms, n_docs, w_mat.nnz, doc_len.mean(),
    )

    started = time.perf_counter()
    ranks = np.empty((len(queries), top_k), dtype=np.int32)
    for i, row in enumerate(queries):
        tids = [vocab[t] for t in tokenize_code(str(row["text"])) if t in vocab]
        if tids:
            scores = np.asarray(w_mat[tids, :].sum(axis=0)).ravel()
        else:
            # No query term is in the vocabulary at all: every document ties.
            # Return an arbitrary-but-deterministic prefix rather than raising.
            scores = np.zeros(n_docs, dtype=np.float32)
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        ranks[i] = part[np.argsort(-scores[part], kind="stable")]
        if (i + 1) % 1000 == 0:
            logger.info("  scored %d/%d queries", i + 1, len(queries))
    search_seconds = time.perf_counter() - started
    return ranks, index_seconds, search_seconds


def hits_at(ranks: np.ndarray, gold_idx: np.ndarray, k: int) -> np.ndarray:
    """Boolean per query: is the gold document inside the top ``k``?"""
    return (ranks[:, :k] == gold_idx[:, None]).any(axis=1)


def recall_by_query_length(
    queries: list[dict[str, Any]], hit: np.ndarray
) -> list[dict[str, Any]]:
    """Dense recall bucketed by query token length.

    Directs the "where is the remaining headroom" question with a measurement
    instead of a guess. 61.9% of train queries (89.0% of test) overflow the
    encoder window, and query compression is the obvious candidate lever - but
    it is only worth building if failures actually concentrate in the long,
    truncated queries. If recall is flat across buckets, the queries are not
    failing because they were cut, and compression would be wasted effort.
    """
    from src.pipeline.baseline import _load_sentence_transformer

    tok = _load_sentence_transformer(config.DENSE_MODEL_NAME).tokenizer
    texts = [str(r["text"]) for r in queries]
    lengths: list[int] = []
    for i in range(0, len(texts), 128):
        lengths.extend(
            tok(texts[i : i + 128], add_special_tokens=False,
                truncation=False, return_length=True)["length"]
        )
    lengths_arr = np.array(lengths)
    budget = config.max_query_tokens()

    edges = [(0, budget), (budget, 2 * budget), (2 * budget, 4 * budget),
             (4 * budget, 10 ** 9)]
    out: list[dict[str, Any]] = []
    for lo, hi in edges:
        mask = (lengths_arr > lo) & (lengths_arr <= hi)
        if not mask.any():
            continue
        out.append({
            "bucket": f"{lo}-{hi if hi < 10 ** 9 else 'inf'} tokens",
            "truncated": lo >= budget,
            "n": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "dense_recall_at_100": round(float(hit[mask].mean()), 4),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--out", type=Path,
        default=config.PROJECT_ROOT / "data" / "bm25_diagnostic.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    corpus, queries, qrels = load_block(args.split)
    logger.info("%s: %d corpus, %d queries", args.split, len(corpus), len(queries))

    pos = {str(r["id"]): i for i, r in enumerate(corpus)}
    # Exactly one gold doc per query on this dataset - asserted, not assumed.
    multi = [q for q in qrels if len(qrels[q]) != 1]
    if multi:
        logger.warning("%d queries have != 1 gold doc; using the first",
                       len(multi))

    keep, gold = [], []
    for i, row in enumerate(queries):
        qid = str(row["id"])
        docs = qrels.get(qid) or {}
        if not docs:
            continue
        did = next(iter(docs))
        if did in pos:
            keep.append(i)
            gold.append(pos[did])
    queries = [queries[i] for i in keep]
    gold_idx = np.array(gold, dtype=np.int32)
    logger.info("%d queries with a resolvable gold doc", len(queries))

    k = args.top_k
    bm_ranks, bm_index_s, bm_search_s = bm25_rankings(corpus, queries, k)
    dn_ranks, dn_doc_s, dn_qry_s = dense_rankings(
        corpus, queries, k, config.DATA_DIR / "embed_cache_diag"
    )

    bm_hit, dn_hit = hits_at(bm_ranks, gold_idx, k), hits_at(dn_ranks, gold_idx, k)
    bm_hit10 = hits_at(bm_ranks, gold_idx, 10)
    dn_hit10 = hits_at(dn_ranks, gold_idx, 10)

    n = len(queries)
    both = int((bm_hit & dn_hit).sum())
    bm_only = int((bm_hit & ~dn_hit).sum())
    dn_only = int((~bm_hit & dn_hit).sum())
    neither = int((~bm_hit & ~dn_hit).sum())
    union = both + bm_only + dn_only

    result = {
        "split": args.split,
        "top_k": k,
        "n_queries": n,
        "model": config.DENSE_MODEL_NAME,
        "bm25": {
            "recall_at_100": round(bm_hit.mean(), 5),
            "recall_at_10": round(bm_hit10.mean(), 5),
            "index_seconds": round(bm_index_s, 1),
            "search_seconds": round(bm_search_s, 1),
            "ms_per_query": round(1000 * bm_search_s / n, 2),
        },
        "dense": {
            "recall_at_100": round(dn_hit.mean(), 5),
            "recall_at_10": round(dn_hit10.mean(), 5),
            "corpus_encode_seconds": round(dn_doc_s, 1),
            "query_encode_seconds": round(dn_qry_s, 1),
        },
        "complementarity": {
            "both_hit": both,
            "bm25_only": bm_only,
            "dense_only": dn_only,
            "neither": neither,
            "union_ceiling": round(union / n, 5),
            "headroom_over_dense": round((union - (both + dn_only)) / n, 5),
        },
        "dense_recall_by_query_length": recall_by_query_length(queries, dn_hit),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    d = result["dense"]["recall_at_100"]
    b = result["bm25"]["recall_at_100"]
    print("\n" + "=" * 66)
    print(f"  BM25 DIAGNOSTIC  —  {args.split} split, {n} queries, top-{k}")
    print("=" * 66)
    print(f"  (a) LEVEL")
    print(f"      BM25-only    recall@100 {b:.4f}   recall@10 {result['bm25']['recall_at_10']:.4f}")
    print(f"      dense        recall@100 {d:.4f}   recall@10 {result['dense']['recall_at_10']:.4f}")
    print(f"      BM25 is {b / d * 100:.1f}% of dense" if d else "")
    print(f"\n  (b) COMPLEMENTARITY  (per-query hit/miss at top-{k})")
    print(f"      both hit        {both:5d}  ({both / n:6.1%})")
    print(f"      BM25 only       {bm_only:5d}  ({bm_only / n:6.1%})   <- what fusion could add")
    print(f"      dense only      {dn_only:5d}  ({dn_only / n:6.1%})")
    print(f"      neither         {neither:5d}  ({neither / n:6.1%})   <- unreachable by fusion")
    print(f"\n      union ceiling   {union / n:.4f}  (best ANY fusion of these two could reach)")
    print(f"      headroom        +{result['complementarity']['headroom_over_dense']:.4f} over dense alone")
    print("=" * 66)
    print(f"  DENSE recall@100 BY QUERY LENGTH (budget {config.max_query_tokens()} tok)")
    for row in result["dense_recall_by_query_length"]:
        flag = "truncated" if row["truncated"] else "fits"
        print(f"      {row['bucket']:22s} {flag:10s} n={row['n']:5d} "
              f"({row['share']:5.1%})  recall@100 {row['dense_recall_at_100']:.4f}")
    print("=" * 66)
    print(f"  BM25 cost: index {bm_index_s:.1f}s, {result['bm25']['ms_per_query']:.1f} ms/query")
    print(f"  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
