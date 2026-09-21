#!/usr/bin/env python3
"""Demonstrate the P1 bar: rebuild cost tracks the CHANGE, not the corpus.

THE REQUIREMENT
---------------
A new version of the codebase must not pay to re-embed a corpus that mostly
did not change. Concretely: edit 100 of 8,765 snippets and the rebuild should
cost about 100 encodes, not 8,765.

WHAT THIS MEASURES
------------------
Three rebuilds of the same corpus, timed end to end:

  1. COLD    - empty cache for this version. Every snippet is encoded. This is
               the number every later rebuild is compared against, and it is
               the cost the cache exists to avoid paying twice.
  2. WARM    - nothing changed. Every snippet must hit cache; the correct
               result is ~0 encodes and a rebuild time near zero.
  3. CHANGED - a seeded random subset of snippets is edited. EXACTLY those
               must re-embed and nothing else.

The pass conditions are asserted, not eyeballed:

  * warm encodes == 0
  * changed encodes == the number of snippets actually edited
  * changed rebuild time is far below cold (the proportionality claim)

WHY A SEPARATE SCRIPT AND NOT A UNIT TEST
-----------------------------------------
The cold arm is a real ~10-minute corpus encode on this CPU box, which does
not belong in `pytest -m "not slow"`. The correctness half - that only
changed snippets miss - is cheap and IS unit-tested; this script is the
timing evidence for the deliverable.

Usage
-----
    python scripts/cache_rebuild_demo.py --changed 100
    python scripts/cache_rebuild_demo.py --changed 100 --limit 1500   # quicker
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from scripts.bm25_diagnostic import load_block  # noqa: E402

logger = logging.getLogger("cache_demo")


def unique_keys(snippets: list[dict[str, str]]) -> set[str]:
    """The distinct cache keys this corpus maps to.

    NOT the same as len(snippets): this corpus contains **11 exact duplicate
    snippets**, which share a key and are encoded once. Counting hits per
    snippet and expectations per key is an off-by-11 waiting to happen - it
    failed exactly that way the first time this ran.
    """
    from src.corpus.index import doc_cache_extras
    from src.versioning.cache import get_cache

    cache = get_cache()
    extras = doc_cache_extras()
    return {cache.make_key(s["processed_text"], config.DENSE_MODEL_NAME, **extras)
            for s in snippets}


def build(snippets: list[dict[str, str]]) -> tuple[float, int, int]:
    """Run one index build; return (seconds, cache_hits, encodes).

    Hits and encodes are counted in UNIQUE KEYS, not snippets.
    """
    from src.corpus.index import DenseIndexBuilder
    from src.versioning.cache import get_cache

    cache = get_cache()
    keys = unique_keys(snippets)
    before = sum(1 for k in keys if cache.get(k) is not None)

    started = time.perf_counter()
    builder = DenseIndexBuilder()
    builder.build(snippets)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started
    return elapsed, before, len(keys) - before


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--changed", type=int, default=100,
                        help="How many snippets to edit for the third rebuild.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N snippets (faster demo).")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Do NOT clear the cache first; skips the cold arm.")
    parser.add_argument("--out", type=Path,
                        default=config.PROJECT_ROOT / "data" / "cache_rebuild_demo.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    np.random.seed(config.RANDOM_SEED)

    from src.pipeline.prism_search import _corpus_text
    from src.pipeline.mteb_compat import to_records
    from src.versioning.cache import DiskEmbeddingCache

    corpus, _, _ = load_block(args.split)
    records = to_records(corpus)
    if args.limit:
        records = records[: args.limit]
    snippets = [{"id": str(r.get("id") or r.get("_id") or i),
                 "processed_text": _corpus_text(r)}
                for i, r in enumerate(records)]
    logger.info("Corpus for demo: %d snippets", len(snippets))

    root = DiskEmbeddingCache().root
    if not args.keep_cache and root.exists():
        logger.warning("Clearing cache tree %s for a true cold arm", root)
        shutil.rmtree(root)

    arms: list[dict[str, Any]] = []

    if not args.keep_cache:
        secs, hits, misses = build(snippets)
        arms.append({"arm": "cold", "seconds": round(secs, 1),
                     "cache_hits": hits, "encoded": misses,
                     "snippets": len(snippets),
                     "unique_keys": len(unique_keys(snippets))})
        logger.info("COLD    %.1fs  hits=%d encoded=%d", secs, hits, misses)

    secs, hits, misses = build(snippets)
    arms.append({"arm": "warm", "seconds": round(secs, 1),
                 "cache_hits": hits, "encoded": misses,
                 "snippets": len(snippets)})
    logger.info("WARM    %.1fs  hits=%d encoded=%d", secs, hits, misses)

    # --- third arm: edit a seeded subset, as a new code version would -------
    n_changed = min(args.changed, len(snippets))
    idx = np.random.default_rng(config.RANDOM_SEED).choice(
        len(snippets), size=n_changed, replace=False)
    edited = [dict(s) for s in snippets]
    for i in idx:
        # A real edit to the indexed text - this is what a new snippet version
        # looks like to the cache.
        edited[i]["processed_text"] = edited[i]["processed_text"] + "\n# v2 edit\n"

    secs, hits, misses = build(edited)
    arms.append({"arm": "changed", "seconds": round(secs, 1),
                 "cache_hits": hits, "encoded": misses,
                 "snippets": len(snippets), "snippets_edited": n_changed})
    logger.info("CHANGED %.1fs  hits=%d encoded=%d (edited %d)",
                secs, hits, misses, n_changed)

    by = {a["arm"]: a for a in arms}
    # Counted in unique keys, because 11 snippets in this corpus are exact
    # duplicates of another and share one cache entry.
    n_keys_edited = len(unique_keys(edited))
    expected_hits = n_keys_edited - n_changed
    checks: list[dict[str, Any]] = [
        {"check": "warm re-embeds nothing",
         "expected": 0, "actual": by["warm"]["encoded"],
         "pass": by["warm"]["encoded"] == 0},
        {"check": "changed re-embeds exactly the edited snippets",
         "expected": n_changed, "actual": by["changed"]["encoded"],
         "pass": by["changed"]["encoded"] == n_changed},
        {"check": "changed leaves every other distinct snippet cached",
         "expected": expected_hits,
         "actual": by["changed"]["cache_hits"],
         "pass": by["changed"]["cache_hits"] == expected_hits},
    ]
    if "cold" in by:
        ratio = by["changed"]["seconds"] / by["cold"]["seconds"] if by["cold"]["seconds"] else 0
        share = n_changed / len(snippets)
        checks.append({
            "check": "rebuild time tracks change size, not corpus size",
            "expected": f"changed/cold time ratio near the {share:.1%} changed share",
            "actual": f"{ratio:.1%}", "pass": ratio < 0.5,
        })

    result = {"split": args.split, "model": config.DENSE_MODEL_NAME,
              "cache_version": config.CACHE_VERSION, "cache_root": str(root),
              "seed": config.RANDOM_SEED, "arms": arms, "checks": checks,
              "all_passed": all(c["pass"] for c in checks)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"  CACHE REBUILD — {len(snippets)} snippets, {n_changed} edited, "
          f"version {config.CACHE_VERSION}")
    print("=" * 78)
    print(f"  {'arm':9} {'seconds':>9} {'cache hits':>11} {'encoded':>9}")
    for a in arms:
        print(f"  {a['arm']:9} {a['seconds']:>9.1f} {a['cache_hits']:>11d} "
              f"{a['encoded']:>9d}")
    print("-" * 78)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}  "
              f"(expected {c['expected']}, got {c['actual']})")
    print("=" * 78)
    print(f"  wrote {args.out}\n")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
