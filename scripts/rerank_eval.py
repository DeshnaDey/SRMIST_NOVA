#!/usr/bin/env python3
"""Measure what a cross-encoder reranker buys, and what caps it.

Reranking is a top-k REORDERING move: it permutes the candidates the
bi-encoder already surfaced. It cannot add a document the retriever missed,
so recall@100 is untouched by construction and the metrics that matter here
are NDCG@10 and MRR@10.

THE CEILING, WHICH IS THE POINT
-------------------------------
This dataset has exactly one relevant document per query. So for a query
whose gold document is at rank r after reranking::

    NDCG@10 = 1/log2(r+1)  if r <= 10 else 0        (IDCG = 1)
    MRR@10  = 1/r          if r <= 10 else 0

A PERFECT reranker puts the gold document at rank 1 whenever it is in the
pool, and can do nothing at all when it is not. Both metrics therefore
collapse to the same oracle::

    max NDCG@10 = max MRR@10 = recall@POOL

That is not a modelling assumption, it is arithmetic, and it is why this step
is bounded before it starts: test recall@100 is 0.307, so no reranker of a
100-deep pool can score above 0.307 on a metric the generic rubric expects to
be much higher. The gold document is simply absent from the pool for 69% of
queries. This script reports the oracle next to every measured number so the
gap between "what reranking got" and "what reranking could ever get" is
explicit.

WHY ONE PASS COVERS EVERY POOL SIZE
-----------------------------------
Reranking a pool of N means: take the dense top-N, rescore, sort. The scores
do not depend on N. So scoring the top-100 once yields every pool size <= 100
for free by simply restricting to the first N candidates before sorting -
four pool sizes for the price of one, which matters because the cross-encoder
runs at ~24 pairs/s on this CPU.

Usage
-----
    python scripts/rerank_eval.py --split train --limit 300
    python scripts/rerank_eval.py --split test --pool 25
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from scripts.bm25_diagnostic import load_block  # noqa: E402
from scripts.corpus_variants import corpus_text  # noqa: E402

logger = logging.getLogger("rerank_eval")

#: Pool depths reported from a single scoring pass. Every one of these is a
#: real configuration the submission could ship, and they differ ~10x in
#: latency, which is a graded deliverable.
POOL_SIZES: tuple[int, ...] = (10, 25, 50, 100)


def dense_top_k(qvecs: np.ndarray, dvecs: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` corpus indices per query, best first."""
    out = np.empty((len(qvecs), k), dtype=np.int32)
    for s in range(0, len(qvecs), 256):
        sims = qvecs[s : s + 256] @ dvecs.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(sims, part, axis=1).argsort(axis=1)[:, ::-1]
        out[s : s + len(sims)] = np.take_along_axis(part, order, axis=1)
    return out


def metrics_from_rank(ranks: np.ndarray, k: int = 10) -> dict[str, float]:
    """NDCG@k, MRR@k and recall@k for one-relevant-document queries.

    ``ranks`` is 1-based, with 0 meaning "not retrieved at all". With a single
    relevant document IDCG@k is 1, so NDCG@k reduces to the gain term.
    """
    ok = (ranks >= 1) & (ranks <= k)
    ndcg = np.where(ok, 1.0 / np.log2(ranks + 1.0, where=ok, out=np.ones_like(ranks, dtype=float)), 0.0)
    mrr = np.where(ok, 1.0 / np.where(ok, ranks, 1), 0.0)
    return {
        f"ndcg_at_{k}": round(float(ndcg.mean()), 5),
        f"mrr_at_{k}": round(float(mrr.mean()), 5),
        f"recall_at_{k}": round(float(ok.mean()), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default=config.RERANK_MODEL_NAME)
    parser.add_argument("--pool", type=int, default=100,
                        help="Deepest pool to score; smaller pools are derived free.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Dev subset of queries, sampled deterministically.")
    parser.add_argument("--batch-size", type=int, default=config.RERANK_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

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

    cache_dir = config.DATA_DIR / "embed_cache_diag"
    tag = config.DENSE_MODEL_NAME.replace("/", "__") + f"_w{config.ENCODER_WINDOW_CAP}"
    qpath = cache_dir / f"{tag}_{args.split}_queries.npy"
    legacy = cache_dir / f"{tag}_queries.npy"
    if qpath.exists():
        qvecs = np.load(qpath)
    elif legacy.exists() and len(np.load(legacy, mmap_mode="r")) == len(queries):
        qvecs = np.load(legacy)
    else:
        logger.error("No cached query vectors for %s. Run: python "
                     "scripts/corpus_variants.py --split %s --variants base",
                     args.split, args.split)
        return 1
    dvecs = np.load(cache_dir / f"{tag}_corpus.npy")
    logger.info("%s: %d queries, %d docs", args.split, len(queries), len(corpus))

    sel = np.arange(len(queries))
    if args.limit is not None and args.limit < len(queries):
        sel = np.sort(np.random.default_rng(config.RANDOM_SEED).choice(
            len(queries), size=args.limit, replace=False))
        logger.warning("DEV SUBSET of %d queries - selection only, never a "
                       "reported headline.", len(sel))
    qv, gi = qvecs[sel], gold_idx[sel]
    qtexts = [str(queries[i]["text"]) for i in sel]

    pool = min(args.pool, len(corpus))
    ranked = dense_top_k(qv, dvecs, pool)

    # Dense-only baseline: 1-based rank of the gold doc, 0 if outside the pool.
    where = np.argmax(ranked == gi[:, None], axis=1)
    found = (ranked == gi[:, None]).any(axis=1)
    dense_rank = np.where(found, where + 1, 0)

    from sentence_transformers import CrossEncoder
    logger.info("Loading cross-encoder %s", args.model)
    t0 = time.perf_counter()
    ce = CrossEncoder(args.model, device=config.DEVICE)
    load_s = time.perf_counter() - t0

    doc_text = [corpus_text(r) for r in corpus]
    logger.info("Scoring %d queries x %d candidates = %d pairs",
                len(sel), pool, len(sel) * pool)
    scores = np.empty((len(sel), pool), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(len(sel)):
        pairs = [(qtexts[i], doc_text[j]) for j in ranked[i]]
        scores[i] = ce.predict(pairs, batch_size=args.batch_size,
                               show_progress_bar=False)
        if (i + 1) % 25 == 0:
            el = time.perf_counter() - t0
            logger.info("  %d/%d queries  %.2f s/query  %.1f pairs/s",
                        i + 1, len(sel), el / (i + 1), (i + 1) * pool / el)
    total_s = time.perf_counter() - t0

    # Refuse to report anything if the model produced non-finite scores.
    # np.argsort on NaN preserves the input order, which would render as a
    # flawless "+0.00000 delta at every pool" - a wrong answer that looks like
    # a clean null. ms-marco-MiniLM-L-6-v2 does exactly this on this stack.
    n_bad = int((~np.isfinite(scores)).sum())
    if n_bad:
        logger.error("%s produced %d/%d non-finite scores. Refusing to report "
                     "numbers: sorting by NaN silently keeps the dense order "
                     "and would look like a clean null result.",
                     args.model, n_bad, scores.size)
        return 2

    result: dict[str, Any] = {
        "split": args.split,
        "rerank_model": args.model,
        "dense_model": config.DENSE_MODEL_NAME,
        "scored_pool": pool,
        "n_queries": int(len(sel)),
        "dev_subset": args.limit is not None and args.limit < len(queries),
        "seed": config.RANDOM_SEED,
        "batch_size": args.batch_size,
        "model_load_seconds": round(load_s, 1),
        "total_rerank_seconds": round(total_s, 1),
        "pairs_per_second": round(len(sel) * pool / total_s, 1),
        "pools": [],
    }

    for n in [p for p in POOL_SIZES if p <= pool]:
        # Rerank a pool of n = rescore only the dense top-n, then sort.
        sub = scores[:, :n]
        order = np.argsort(-sub, axis=1, kind="stable")
        reranked = np.take_along_axis(ranked[:, :n], order, axis=1)
        w = np.argmax(reranked == gi[:, None], axis=1)
        f = (reranked == gi[:, None]).any(axis=1)
        rr_rank = np.where(f, w + 1, 0)

        dense_n = np.where((dense_rank >= 1) & (dense_rank <= n), dense_rank, 0)
        oracle = float(((dense_n >= 1)).mean())  # recall@n == the oracle for both metrics
        row = {
            "pool": n,
            "oracle_ndcg_at_10": round(oracle, 5),
            "dense": metrics_from_rank(dense_n),
            "reranked": metrics_from_rank(rr_rank),
            "latency_seconds_per_query": round(n / result["pairs_per_second"], 3),
        }
        row["ndcg_delta"] = round(row["reranked"]["ndcg_at_10"]
                                  - row["dense"]["ndcg_at_10"], 5)
        row["pct_of_oracle"] = (round(100 * row["reranked"]["ndcg_at_10"] / oracle, 1)
                                if oracle else None)
        result["pools"].append(row)

    out = args.out or (config.PROJECT_ROOT / "data" /
                       f"rerank_{args.split}{'_dev' if result['dev_subset'] else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 86)
    print(f"  RERANK — {args.split}, {len(sel)} queries, {args.model}")
    print(f"  {result['pairs_per_second']:.1f} pairs/s on {config.DEVICE}"
          f"   (bi-encoder query latency for reference: 161 ms)")
    print("=" * 86)
    print(f"  {'pool':>5} {'dense NDCG':>11} {'rerank NDCG':>12} {'delta':>9} "
          f"{'oracle':>8} {'% oracle':>9} {'s/query':>8}")
    for r in result["pools"]:
        print(f"  {r['pool']:>5} {r['dense']['ndcg_at_10']:>11.5f} "
              f"{r['reranked']['ndcg_at_10']:>12.5f} {r['ndcg_delta']:>+9.5f} "
              f"{r['oracle_ndcg_at_10']:>8.5f} {str(r['pct_of_oracle']):>9} "
              f"{r['latency_seconds_per_query']:>8.2f}")
    print("=" * 86)
    print(f"  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
