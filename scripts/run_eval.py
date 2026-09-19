#!/usr/bin/env python3
"""Run the MTEB AppsRetrieval evaluation and write appsretrieval_results.json.

This is the single entry point for producing a score. Everything it needs
comes from ``src/config.py``, so a result is reproducible from a git SHA.

Usage
-----
    # day-one baseline: bare bi-encoder, no pipeline stages
    python scripts/run_eval.py --pipeline baseline

    # the full hybrid pipeline (needs the stage stubs implemented)
    python scripts/run_eval.py --pipeline full

    # fast smoke run over 50 queries - NOT a reportable number
    python scripts/run_eval.py --pipeline baseline --limit 50

After a run, copy the printed NDCG@10 / MRR into ``experiments.md`` along with
what you changed. An unlogged experiment is an experiment we will repeat.
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

# Make `src` importable when this script is run directly (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.pipeline.baseline import BaselineEncoder  # noqa: E402
from src.pipeline.prism_search import PrismSearch  # noqa: E402

logger = logging.getLogger("run_eval")


def build_model(pipeline: str) -> Any:
    """Instantiate the object handed to ``mteb.evaluate``.

    Parameters
    ----------
    pipeline:
        ``"baseline"`` for the bare bi-encoder (MTEB wraps it into a search
        model itself), or ``"full"`` for our ``SearchProtocol`` implementation.
    """
    if pipeline == "baseline":
        logger.info("Baseline: bare bi-encoder %s", config.DENSE_MODEL_NAME)
        return BaselineEncoder()
    if pipeline == "full":
        logger.info("Full pipeline: hybrid retrieve -> fuse -> rerank")
        return PrismSearch()
    raise ValueError(f"Unknown pipeline {pipeline!r}; expected 'baseline' or 'full'")


def prepare_task(split: str, limit: int | None) -> Any:
    """Load the graded task, optionally on a non-default split and subsampled.

    Why this exists
    ---------------
    ``mteb.evaluate`` has no ``limit`` argument, and the baseline path never
    reaches ``PrismSearch.search`` (MTEB wraps a bare encoder into its own
    search model), so ``config.SMOKE_TEST_QUERY_LIMIT`` is invisible to it.
    A smoke run therefore has to be produced by shrinking the task's own query
    set before evaluation starts.

    ``AppsRetrieval`` also declares ``eval_splits == ["test"]`` only, so any
    other split has to be requested explicitly or ``load_data()`` simply will
    not materialise it. Verified against mteb 2.20.11.

    Parameters
    ----------
    split:
        Split to evaluate. ``"test"`` for official numbers; ``"train"`` for
        experimentation.
    limit:
        Keep only this many queries, sampled deterministically. ``None``
        evaluates the whole split.

    Returns
    -------
    AbsTask
        A loaded task whose query set and qrels have been narrowed together.
        The corpus is deliberately left at full size: shrinking it would make
        retrieval artificially easy and the number meaningless.
    """
    import mteb

    task = mteb.get_task(config.MTEB_TASK_NAME)
    try:
        task._eval_splits = [split]
    except Exception:  # pragma: no cover - defensive
        task.metadata.eval_splits = [split]

    task.load_data()

    if limit is None:
        return task

    subset = task.hf_subsets[0]
    block = task.dataset[subset][split]
    queries, qrels = block["queries"], block["relevant_docs"]

    # Deterministic by construction: sort first (HuggingFace row order is not
    # guaranteed stable across dataset revisions), then sample with a fixed
    # seed. Two runs on the same split and limit see the identical subset, so
    # a model-vs-model delta is a real delta and not a resampling artifact.
    ordered = sorted(str(row["id"]) for row in queries)
    rng = random.Random(config.RANDOM_SEED)
    keep = set(rng.sample(ordered, min(limit, len(ordered))))

    block["queries"] = queries.filter(lambda row: str(row["id"]) in keep)
    # Narrow the qrels to match. Leaving them wide would have MTEB scoring
    # against relevance judgements for queries we never ran.
    block["relevant_docs"] = {q: v for q, v in qrels.items() if q in keep}

    logger.info(
        "Smoke subset: %d/%d queries (seed=%d), corpus left at full size",
        block["queries"].num_rows, len(ordered), config.RANDOM_SEED,
    )
    return task


def extract_scores(results: Any) -> dict[str, float]:
    """Pull the headline metrics out of whatever ``mteb.evaluate`` returned.

    MTEB's result objects have been reshaped across 2.x releases, so this walks
    the structure defensively rather than assuming a fixed path. The full
    result is written to disk regardless; this is only for the console summary
    and the experiments.md row.

    Returns
    -------
    dict[str, float]
        Metric name -> value. Empty if nothing recognisable was found, which is
        a reporting failure, not an evaluation failure.
    """
    wanted = (config.PRIMARY_METRIC, config.SECONDARY_METRIC, "mrr_at_10", "ndcg_at_10")
    found: dict[str, float] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in wanted and isinstance(value, (int, float)):
                    found.setdefault(key, float(value))
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif hasattr(node, "scores"):
            walk(node.scores)
        elif hasattr(node, "__dict__"):
            walk(vars(node))

    walk(results)
    return found


def to_serializable(results: Any) -> Any:
    """Convert MTEB's result objects into something ``json.dump`` accepts."""
    for attr in ("to_dict", "model_dump", "dict"):
        method = getattr(results, attr, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                pass
    if isinstance(results, (list, tuple)):
        return [to_serializable(r) for r in results]
    if isinstance(results, dict):
        return {k: to_serializable(v) for k, v in results.items()}
    if hasattr(results, "__dict__"):
        return {k: to_serializable(v) for k, v in vars(results).items()}
    return results


def main() -> int:
    """Run the evaluation. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Keep the usage examples in the module docstring readable; the default
        # formatter collapses their line breaks into one paragraph.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        choices=("baseline", "full"),
        default="baseline",
        help="Which model to evaluate (default: baseline).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to evaluate. 'train' for experiments, 'test' for official "
             "numbers only (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap queries for a smoke run, sampled deterministically from "
             "config.RANDOM_SEED. NOT a reportable score.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_JSON,
        help=f"Results JSON path (default: {config.RESULTS_JSON.name}).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.limit is not None:
        config.SMOKE_TEST_QUERY_LIMIT = args.limit
        logger.warning("Smoke run: %d queries. Do NOT log this to experiments.md.",
                       args.limit)

    config.ensure_dirs()

    # Imported here, after arg parsing, so that `--help` works without mteb
    # installed - useful when a teammate is still setting up their venv.
    try:
        import mteb
    except ImportError:
        logger.error(
            "mteb is not installed. Run: pip install -r requirements.txt\n"
            "Note this project requires MTEB v2 (mteb.evaluate + SearchProtocol), "
            "not v1."
        )
        return 1

    if not hasattr(mteb, "evaluate"):
        logger.error(
            "Installed mteb has no `evaluate` - that is the v1 API. "
            "This project targets MTEB v2. Upgrade and pin it in requirements.txt."
        )
        return 1

    model = build_model(args.pipeline)

    logger.info("Loading task %s (split=%s)", config.MTEB_TASK_NAME, args.split)
    task = prepare_task(args.split, args.limit)

    logger.info("Running evaluation (CPU-only; this takes a while)")
    started = time.perf_counter()
    results = mteb.evaluate(
        model,
        tasks=[task],
        encode_kwargs={"batch_size": config.BATCH_SIZE},
        # MTEB defaults to a results cache at ~/.cache/mteb with
        # overwrite_strategy="only-missing", which will happily hand back a
        # previous run's scores instead of re-evaluating. That is exactly how
        # a model comparison silently compares one model against itself, so
        # every run here is forced to actually run.
        overwrite_strategy="always",
    )
    wall_clock = time.perf_counter() - started
    logger.info("Evaluation finished in %.1fs (%.1f min)", wall_clock, wall_clock / 60)

    payload = {
        "task": config.MTEB_TASK_NAME,
        "pipeline": args.pipeline,
        "split": args.split,
        "release_tag": config.RELEASE_TAG,
        "config": config.describe(),
        "smoke_run": args.limit is not None,
        "query_limit": args.limit,
        "wall_clock_seconds": round(wall_clock, 1),
        "results": to_serializable(results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote %s", args.output)

    scores = extract_scores(results)
    if scores:
        print("\n" + "=" * 56)
        print(f"  {config.MTEB_TASK_NAME}  ({args.pipeline}, split={args.split})")
        print("=" * 56)
        for name in (config.PRIMARY_METRIC, config.SECONDARY_METRIC):
            if name in scores:
                print(f"  {name:<24} {scores[name]:.4f}")
        print(f"  {'wall_clock':<24} {wall_clock:.1f}s ({wall_clock / 60:.1f} min)")
        print("=" * 56)
        if args.limit is not None:
            print("  -> SMOKE RUN, not reportable\n")
        elif args.split != "test":
            print(f"  -> split={args.split}: experiment only, not an official number\n")
        else:
            print("  -> add a row to experiments.md\n")
    else:
        logger.warning(
            "Could not locate %s in the results object; inspect %s by hand "
            "and fix extract_scores().",
            config.PRIMARY_METRIC,
            args.output,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
