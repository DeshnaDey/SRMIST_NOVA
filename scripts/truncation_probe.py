#!/usr/bin/env python3
"""Is truncation CAUSING the recall loss, or just correlated with it?

THE QUESTION
------------
Dense recall@100 on train, stratified by query length against arctic's
510-token budget (``data/bm25_diagnostic.json``)::

    0-510 tokens    fits        n=3,770   recall@100  0.7493
    510-1020 tokens truncated   n=1,098   recall@100  0.4827

A 27-point gap that looks like an open-and-shut case for query compression.
It is not. The comparison is CORRELATIONAL: the two buckets contain different
queries, and a problem statement that runs to 800 tokens is plausibly just a
harder problem - more constraints, more cases, more subtle - than one that
fits in 400. Under that reading the queries in the second bucket would score
0.48 even with an infinite context window, and compressing them would buy
nothing.

Nothing in an observational split can separate those two stories, because
length is exactly what defines the buckets.

THE INTERVENTION
----------------
So intervene instead of observing. Take the queries that currently FIT -
whose recall (0.7493) is measured with nothing cut - and cut them anyway.
Same queries, same gold documents, same corpus, same model: the ONLY thing
that changes between arms is how much of the query the encoder sees. Any
recall drop is therefore caused by truncation, because it is the only free
variable. That is the whole design.

Queries are cut to a FRACTION of their own token length rather than to a
fixed token count, because that is the shape of the real intervention: a
700-token query meeting a 510-token budget keeps 73% of itself, and a
1,000-token one keeps 51%. Fixed-count truncation would instead confound the
dose with the original length. The observed retention ratio of the genuinely
truncated bucket is computed below and printed next to the arms, so the
probe can be read against the number it is trying to explain.

READING THE RESULT
------------------
* Recall falls steeply as the retention ratio drops -> truncation is causal,
  the tail of a query carries retrieval signal, and query compression (keep
  the ask and the constraints, drop the worked examples) is worth building.
* Recall is flat -> the tail carries nothing the first N tokens did not
  already say. Compression cannot help, the 27-point gap is intrinsic
  difficulty, and the remaining headroom is inside the encoder (fine-tuning,
  hard-negative mining) rather than in what we feed it.

WHY IT IS CHEAP
---------------
Corpus embeddings are cached from the BM25 diagnostic, so each arm costs one
query encode over the fitting subset and a matrix multiply.

THREE THINGS THIS IS CAREFUL ABOUT
----------------------------------
1. **No decode round-trip.** Truncating by decoding token ids back to text
   would lowercase everything and normalise whitespace under this uncased
   wordpiece tokenizer - a confound applied to the treatment arms and not the
   control. Cutting on CHARACTER OFFSETS instead makes every arm a genuine
   prefix of the original raw string, so the 100% arm is the original text
   byte-for-byte.
2. **The control is re-encoded, not reused.** The 100% arm goes through the
   same code path as every other arm rather than reading the cached query
   vectors, so an arm-to-arm difference cannot be an artifact of the path.
   The cached vectors are still scored, and printed as a harness check: it
   must reproduce the 0.6870 / 0.7493 already in the log.
3. **Within-query, not between-bucket.** Every arm is scored over the same
   query set, so the comparison never reintroduces the composition problem
   that made the original stratification correlational.

Usage
-----
    python scripts/truncation_probe.py
    python scripts/truncation_probe.py --split train --limit 300
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

logger = logging.getLogger("trunc_probe")

#: recall@100 already in experiments.md for each split, so the harness check
#: can assert against a number nobody has to look up. TRAIN is the stratified
#: row in the BM25 decision log; TEST is the logged full-split result, and has
#: no published fitting-only figure, hence the None.
KNOWN_RECALL_AT_100: dict[str, dict[str, float | None]] = {
    "train": {"all": 0.6870, "fitting": 0.7493},
    "test": {"all": 0.30677, "fitting": None},
}

#: Fraction of its own tokens each query keeps. 1.0 is the control arm: it is
#: re-encoded like the others rather than read from cache, so that an arm-to-
#: arm delta cannot be an artifact of the encode path.
#:
#: The ladder is geometric rather than linear because the hypothesis under
#: test is about the SHAPE of the dose-response, not a single point. A flat
#: 0.75 arm alone would be ambiguous - it could mean the tail is worthless, or
#: that 25% is too small a dose to register. Carrying the ladder down to 0.10
#: (a ~40-token stub of a competitive-programming statement) forces the issue:
#: if recall is still flat THERE, the encoder was never using the body of the
#: query, and no amount of clever compression changes that.
RETENTION_RATIOS: tuple[float, ...] = (1.0, 0.75, 0.50, 0.25, 0.10)


def token_lengths(texts: list[str], tok: Any) -> np.ndarray:
    """Content-token length of each text, with no truncation."""
    lengths: list[int] = []
    for i in range(0, len(texts), 128):
        lengths.extend(
            tok(texts[i : i + 128], add_special_tokens=False,
                truncation=False, return_length=True)["length"]
        )
    return np.array(lengths, dtype=np.int32)


def truncate_to_ratio(texts: list[str], ratio: float, tok: Any) -> list[str]:
    """Keep the first ``ratio`` of each text's tokens, cutting on characters.

    The cut point is the END OFFSET of the last kept token, so the result is a
    genuine prefix of the input string. This is the whole reason the probe
    does not decode: ``tok.decode(ids[:k])`` under an uncased wordpiece
    tokenizer returns lowercased, whitespace-normalised, ``##``-rejoined text,
    which would apply a systematic distortion to every treatment arm that the
    control never sees. A character prefix has no such artifact - at
    ``ratio=1.0`` it returns the original text unchanged but for trailing
    whitespace.

    ``ceil`` on the token count keeps a one-token query from becoming empty.
    """
    out: list[str] = []
    for i in range(0, len(texts), 128):
        batch = texts[i : i + 128]
        enc = tok(batch, add_special_tokens=False, truncation=False,
                  return_offsets_mapping=True)
        for text, offsets in zip(batch, enc["offset_mapping"]):
            if not offsets:
                out.append(text)
                continue
            keep = max(1, min(len(offsets), int(np.ceil(len(offsets) * ratio))))
            out.append(text[: offsets[keep - 1][1]])
    return out


def gold_rank(query_vecs: np.ndarray, doc_vecs: np.ndarray,
              gold_idx: np.ndarray) -> np.ndarray:
    """0-based rank of each query's gold document under cosine similarity.

    Counting how many documents outscore the gold one is exact and avoids
    materialising a top-k list per arm. Vectors are L2-normalised by the
    encoder, so the inner product IS cosine. Chunked over queries because a
    full 5,000 x 8,765 float32 score matrix is ~175 MB on an 8 GB box.
    """
    ranks = np.empty(len(query_vecs), dtype=np.int32)
    for start in range(0, len(query_vecs), 256):
        sims = query_vecs[start : start + 256] @ doc_vecs.T
        gold = gold_idx[start : start + 256]
        gold_scores = sims[np.arange(len(sims)), gold][:, None]
        ranks[start : start + len(sims)] = (sims > gold_scores).sum(axis=1)
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke run over N fitting queries. Never a logged number.")
    parser.add_argument(
        "--out", type=Path,
        default=config.PROJECT_ROOT / "data" / "truncation_probe.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    from src.pipeline.baseline import BaselineEncoder, _load_sentence_transformer

    corpus, queries, qrels = load_block(args.split)

    # Resolve gold documents exactly as the BM25 diagnostic does, so the query
    # ordering matches the cached query-embedding file row for row.
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
    gold_all = np.array(gold, dtype=np.int32)
    logger.info("%s: %d corpus, %d queries with a resolvable gold doc",
                args.split, len(corpus), len(queries))

    tok = _load_sentence_transformer(config.DENSE_MODEL_NAME).tokenizer
    texts_all = [str(r["text"]) for r in queries]
    lengths_all = token_lengths(texts_all, tok)
    budget = config.max_query_tokens()

    cache_dir = config.DATA_DIR / "embed_cache_diag"
    tag = (config.DENSE_MODEL_NAME.replace("/", "__")
           + f"_w{config.ENCODER_WINDOW_CAP}")
    doc_path = cache_dir / f"{tag}_corpus.npy"
    if not doc_path.exists():
        logger.error("No cached corpus embeddings at %s. Run "
                     "scripts/bm25_diagnostic.py --split %s first.",
                     doc_path, args.split)
        return 1
    doc_vecs = np.load(doc_path)
    logger.info("Loaded cached corpus embeddings %s %s", doc_path.name, doc_vecs.shape)

    k = args.top_k

    # -- harness check --------------------------------------------------------
    # Score the cached query vectors before touching anything. These must
    # reproduce the numbers already in experiments.md; if they do not, the
    # gold-resolution order or the cache is wrong and every arm below is junk.
    baseline_check: dict[str, Any] = {}
    qry_path = cache_dir / f"{tag}_queries.npy"
    if qry_path.exists():
        cached = np.load(qry_path)
        if len(cached) == len(queries):
            r = gold_rank(cached, doc_vecs, gold_all)
            fits_mask_all = lengths_all <= budget
            known = KNOWN_RECALL_AT_100.get(args.split, {})
            baseline_check = {
                "source": qry_path.name,
                "recall_at_100_all": round(float((r < k).mean()), 5),
                "recall_at_10_all": round(float((r < 10).mean()), 5),
                "recall_at_100_fitting": round(float((r < k)[fits_mask_all].mean()), 5),
                "recall_at_100_truncated": round(
                    float((r < k)[~fits_mask_all].mean()), 5),
                "expected_recall_at_100_all": known.get("all"),
                "expected_recall_at_100_fitting": known.get("fitting"),
            }
            logger.info("Harness check on cached vectors: recall@100 all %.4f "
                        "(log says %s), fitting %.4f (log says %s)",
                        baseline_check["recall_at_100_all"],
                        known.get("all", "n/a"),
                        baseline_check["recall_at_100_fitting"],
                        known.get("fitting") or "n/a")
        else:
            logger.warning("Cached query vectors are %d rows against %d queries; "
                           "skipping the harness check.", len(cached), len(queries))

    # -- the observed retention ratio the probe is trying to explain ----------
    # A truncated query keeps budget/len of itself. Reporting the median for
    # the 510-1020 bucket puts the arms on the same axis as the 0.4827 they
    # are being compared against.
    over = lengths_all > budget
    just_over = over & (lengths_all <= 2 * budget)
    observed = {
        "n_truncated": int(over.sum()),
        "median_retention_ratio_all_truncated": (
            round(float(np.median(budget / lengths_all[over])), 4)
            if over.any() else None),
        "median_retention_ratio_510_1020_bucket": (
            round(float(np.median(budget / lengths_all[just_over])), 4)
            if just_over.any() else None),
    }

    # -- the fitting subset: the only queries the probe intervenes on ---------
    fit_idx = np.flatnonzero(lengths_all <= budget)
    if args.limit is not None:
        rng = np.random.default_rng(config.RANDOM_SEED)
        fit_idx = np.sort(rng.choice(fit_idx, size=min(args.limit, len(fit_idx)),
                                     replace=False))
        logger.warning("SMOKE RUN over %d fitting queries - not a logged number.",
                       len(fit_idx))

    texts = [texts_all[i] for i in fit_idx]
    lengths = lengths_all[fit_idx]
    gold_idx = gold_all[fit_idx]
    logger.info("Intervening on %d queries that FIT the %d-token budget "
                "(median %d tokens)", len(texts), budget, int(np.median(lengths)))

    # Bands within the fitting subset. A 200-token query has far less tail to
    # lose than a 480-token one, so a single pooled number would average a
    # strong effect together with queries the intervention barely touches.
    bands = [(0, budget // 4), (budget // 4, budget // 2),
             (budget // 2, 3 * budget // 4), (3 * budget // 4, budget)]

    header_fields: dict[str, Any] = {
        "probe": "causal truncation",
        "split": args.split,
        "top_k": k,
        "model": config.DENSE_MODEL_NAME,
        "query_token_budget": budget,
        "smoke_run": args.limit is not None,
        "seed": config.RANDOM_SEED,
        "n_fitting_queries": int(len(texts)),
        "harness_check": baseline_check,
        "observed_truncation": observed,
    }

    encoder = BaselineEncoder()
    arms: list[dict[str, Any]] = []
    hit_by_ratio: dict[float, np.ndarray] = {}

    # Each arm's query vectors are cached, and the result file is rewritten
    # after every arm. The encode is ~29 minutes of CPU across the ladder and
    # an earlier run of this probe was killed after the last arm but before it
    # wrote anything - losing all of it. Neither is acceptable for a number
    # that gates two weeks of work, so a rerun now costs seconds and a kill
    # costs at most the arm in flight.
    arm_cache = cache_dir / "truncation_arms"
    arm_cache.mkdir(parents=True, exist_ok=True)

    def write_partial() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**header_fields, "arms": arms}, indent=2),
                            encoding="utf-8")

    for ratio in RETENTION_RATIOS:
        cut = truncate_to_ratio(texts, ratio, tok) if ratio < 1.0 else list(texts)
        kept = token_lengths(cut, tok)
        # Keyed by everything that changes the vectors: model, window, split,
        # ratio, and the query count (so a --limit run can never be mistaken
        # for a full one).
        vec_path = arm_cache / f"{tag}_{args.split}_n{len(cut)}_r{ratio:.2f}.npy"
        started = time.perf_counter()
        if vec_path.exists():
            logger.info("Reusing cached arm vectors %s", vec_path.name)
            vecs = np.load(vec_path)
        else:
            vecs = encoder.encode(cut, prompt_type="query")
            np.save(vec_path, np.asarray(vecs, dtype=np.float32))
        encode_s = time.perf_counter() - started
        ranks = gold_rank(np.asarray(vecs, dtype=np.float32), doc_vecs, gold_idx)
        hit = ranks < k
        hit_by_ratio[ratio] = hit

        arm: dict[str, Any] = {
            "retention_ratio": ratio,
            "n": int(len(cut)),
            "median_tokens_kept": int(np.median(kept)),
            "recall_at_100": round(float(hit.mean()), 5),
            "recall_at_10": round(float((ranks < 10).mean()), 5),
            "median_gold_rank": int(np.median(ranks)),
            "encode_seconds": round(encode_s, 1),
            "by_original_length": [],
        }
        for lo, hi in bands:
            m = (lengths > lo) & (lengths <= hi)
            if not m.any():
                continue
            arm["by_original_length"].append({
                "bucket": f"{lo}-{hi} tokens",
                "n": int(m.sum()),
                "recall_at_100": round(float(hit[m].mean()), 5),
            })
        arms.append(arm)
        logger.info("ratio %.2f: median %4d tokens kept, recall@100 %.4f "
                    "(encode %.0fs)", ratio, arm["median_tokens_kept"],
                    arm["recall_at_100"], encode_s)
        write_partial()

    control = arms[0]["recall_at_100"]
    for arm in arms:
        arm["delta_vs_control"] = round(arm["recall_at_100"] - control, 5)
        # Queries the control found and this arm lost. The pooled recall delta
        # is a net figure; this is the gross damage, and the two differ when
        # truncation also luckily promotes something.
        lost = int((hit_by_ratio[1.0] & ~hit_by_ratio[arm["retention_ratio"]]).sum())
        gained = int((~hit_by_ratio[1.0] & hit_by_ratio[arm["retention_ratio"]]).sum())
        arm["queries_lost_vs_control"] = lost
        arm["queries_gained_vs_control"] = gained

    result = {**header_fields, "arms": arms}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  CAUSAL TRUNCATION PROBE — {args.split} split, "
          f"{len(texts)} queries that FIT the {budget}-token budget")
    print("=" * 74)
    if baseline_check:
        print(f"  harness check (cached vectors): recall@100 all "
              f"{baseline_check['recall_at_100_all']:.4f} (logged 0.6870), "
              f"fitting {baseline_check['recall_at_100_fitting']:.4f} "
              f"(logged 0.7493)")
    if observed["median_retention_ratio_510_1020_bucket"] is not None:
        print(f"  real truncated queries keep a median "
              f"{observed['median_retention_ratio_510_1020_bucket']:.0%} of "
              f"themselves (510-1020 bucket), and score 0.4827")
    print("-" * 74)
    print(f"  {'keep':>6}  {'med tok':>7}  {'recall@100':>10}  {'delta':>8}  "
          f"{'lost':>5}  {'gained':>6}")
    for arm in arms:
        print(f"  {arm['retention_ratio']:>5.0%}  {arm['median_tokens_kept']:>7d}  "
              f"{arm['recall_at_100']:>10.4f}  {arm['delta_vs_control']:>+8.4f}  "
              f"{arm['queries_lost_vs_control']:>5d}  "
              f"{arm['queries_gained_vs_control']:>6d}")
    print("-" * 74)
    print("  recall@100 by ORIGINAL query length (rows = retention ratio)")
    header = [b["bucket"] for b in arms[0]["by_original_length"]]
    print(f"  {'keep':>6}  " + "  ".join(f"{h:>16s}" for h in header))
    for arm in arms:
        cells = "  ".join(f"{b['recall_at_100']:>16.4f}"
                          for b in arm["by_original_length"])
        print(f"  {arm['retention_ratio']:>5.0%}  " + cells)
    print("=" * 74)
    print(f"  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
