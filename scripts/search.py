#!/usr/bin/env python3
"""Ask a coding question, get matching code snippets back.

Runs the SAME pipeline that ``scripts/run_eval.py --pipeline full`` scores -
``PrismSearch`` with config defaults, so query preprocessing, BM25 + fusion
and reranking are applied exactly when ``src/config.py`` turns them on. The
corpus vectors are read from the content-hash embedding cache under
``config.CACHE_DIR``; nothing is re-encoded when that cache is warm.

Good questions are competitive-programming style problem descriptions: the
corpus is Python solutions to APPS problems.

Usage
-----
    # interactive: load once, then ask as many questions as you like
    python scripts/search.py

    # one question, then exit
    python scripts/search.py --query "Given an array, find the longest increasing subsequence" --k 5

    # run real test question N (1-based) and check the correct snippet's rank
    python scripts/search.py --from-test 12
"""

from __future__ import annotations

import os

# Silence the libraries BEFORE they are imported: this output is meant to be
# read (and recorded), so progress bars and advisory warnings must not land
# in the middle of the results. --verbose turns logging back on.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# Start offline. Once the model and dataset are cached, every Hub request at
# startup is only a freshness check, and on a slow link those checks cost tens
# of seconds or hang outright. If something is not cached yet, main() restarts
# the script once with the Hub enabled (see _restart_online).
_FORCED_OFFLINE = "HF_HUB_OFFLINE" not in os.environ
if _FORCED_OFFLINE:
    for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# Make `src` and `scripts` importable when run directly (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

SNIPPET_LINES = 25
QUESTION_LINES = 12
SPLIT = "test"


def rule(char: str = "─") -> str:
    """A full-width separator line, capped so it stays readable when wide."""
    return char * min(shutil.get_terminal_size((88, 20)).columns, 88)


def step(message: str) -> float:
    """Print a loading step without a newline; returns the start time."""
    print(f"  {message} ...", end="", flush=True)
    return time.perf_counter()


def done(started: float) -> None:
    print(f" done ({time.perf_counter() - started:.1f}s)", flush=True)


def quiet_libraries() -> None:
    """Keep library logging and warnings out of the results."""
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("sentence_transformers", "transformers", "huggingface_hub",
                 "datasets", "mteb", "src"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # These libraries configure their own loggers, so the calls above do not
    # reach them; each has its own switch.
    try:
        import datasets
        import huggingface_hub.utils
        import transformers

        datasets.disable_progress_bars()
        datasets.logging.set_verbosity_error()
        huggingface_hub.utils.logging.set_verbosity_error()
        huggingface_hub.utils.disable_progress_bars()
        transformers.logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


class SearchSession:
    """Everything loaded once: the task data, the indexed pipeline, the model."""

    def __init__(self) -> None:
        startup = time.perf_counter()
        print(rule("═"))
        print("  PRISM code search")
        print(rule("═"))

        # torch, sentence-transformers and mteb take a while to import.
        t = step("Starting up")
        from scripts.run_eval import prepare_task
        from src.pipeline.mteb_compat import record_id, to_records
        from src.pipeline.prism_search import PrismSearch
        from src.versioning.cache import get_cache
        done(t)

        config.ensure_dirs()

        t = step(f"Loading the {config.MTEB_TASK_NAME} dataset ({SPLIT} split)")
        task = prepare_task(SPLIT, None)
        block = task.dataset[task.hf_subsets[0]][SPLIT]
        corpus = block["corpus"]
        self.qrels: dict[str, dict[str, int]] = block["relevant_docs"]
        queries = to_records(block["queries"])
        # Numeric id order (q5001, q5002, ...) so --from-test N is stable.
        queries.sort(key=lambda r: (len(record_id(r, 0)), record_id(r, 0)))
        self.test_queries = [(record_id(r, i), str(r.get("text", "")))
                             for i, r in enumerate(queries)]
        done(t)
        print(f"    {len(corpus):,} code snippets, {len(self.test_queries):,} test questions")

        cache = get_cache()
        stats = cache.stats() if hasattr(cache, "stats") else {"entries": 0}
        if stats["entries"] == 0:
            print(f"  Embedding cache at {config.CACHE_DIR} is empty: building it "
                  "now. This is a one-time step that takes roughly 10-15 min on "
                  "a laptop CPU (up to ~40 min in Docker); later starts take seconds.")
        else:
            print(f"    embedding cache: {stats['entries']:,} saved vectors in "
                  f"{config.CACHE_DIR}")

        t = step("Indexing the corpus from the saved embeddings")
        self.pipeline = PrismSearch()
        self.pipeline.index(corpus, hf_split=SPLIT)
        done(t)

        t = step(f"Loading the query encoder ({config.DENSE_MODEL_NAME})")
        # DenseRetriever loads its encoder lazily; one throwaway query here
        # moves that cost out of the first real question.
        self.pipeline.dense_retriever.retrieve_batch(
            self.pipeline.query_processor.process_batch(["warm up"]), 1)
        done(t)

        if config.ENABLE_RERANK:
            t = step(f"Loading the reranker ({config.RERANK_MODEL_NAME})")
            _ = self.pipeline.reranker.model
            done(t)

        # Time the rerank stage by wrapping whatever reranker the pipeline
        # built; retrieval time is then total minus rerank.
        self.rerank_seconds = 0.0
        inner = self.pipeline.reranker.rerank

        def timed_rerank(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                self.rerank_seconds += time.perf_counter() - started

        self.pipeline.reranker.rerank = timed_rerank

        stages = [
            "query preprocessing" if config.ENABLE_QUERY_PREPROCESSING else None,
            "BM25 + dense, RRF fusion" if config.ENABLE_BM25 else "dense",
            "rerank" if config.ENABLE_RERANK else None,
        ]
        self.startup_seconds = time.perf_counter() - startup
        print(f"  Ready in {self.startup_seconds:.1f}s. Pipeline: "
              + " -> ".join(s for s in stages if s)
              + "  (stages that config turns off are skipped)")

    def search(self, question: str, depth: int) -> tuple[list[tuple[str, float]], dict[str, float]]:
        """Run one question through the pipeline; return hits and timings."""
        self.rerank_seconds = 0.0
        started = time.perf_counter()
        results = self.pipeline.search([{"id": "query", "text": question}],
                                       hf_split=SPLIT, top_k=depth)
        total = time.perf_counter() - started
        ranked = sorted(results["query"].items(), key=lambda kv: kv[1], reverse=True)
        return ranked, {"retrieval": total - self.rerank_seconds,
                        "rerank": self.rerank_seconds, "total": total}

    def show(self, question: str, k: int, relevant: set[str] | None = None) -> None:
        # With a qrel to check, ask for the full candidate pool so we can say
        # where the correct snippet landed even when it misses the top k. The
        # top k is unchanged: the pipeline ranks the same pool either way.
        depth = max(k, config.TOP_K_FUSED) if relevant else k
        ranked, timings = self.search(question, depth)

        for rank, (cid, score) in enumerate(ranked[:k], start=1):
            mark = "  ✔ correct" if relevant and cid in relevant else ""
            print(rule())
            print(f"  #{rank}   {cid}   score {score:.4f}{mark}")
            print(rule())
            lines = self.pipeline.snippet_lookup.get(cid, "").rstrip().splitlines()
            for line in lines[:SNIPPET_LINES]:
                print(f"    {line}")
            if len(lines) > SNIPPET_LINES:
                print(f"    ... ({len(lines) - SNIPPET_LINES} more lines)")
        print(rule())

        if relevant:
            positions = [i for i, (cid, _) in enumerate(ranked, start=1) if cid in relevant]
            target = ", ".join(sorted(relevant))
            if positions and positions[0] <= k:
                print(f"  Correct snippet {target}: IN the top {k}, at rank {positions[0]}")
            elif positions:
                print(f"  Correct snippet {target}: NOT in the top {k} (rank {positions[0]})")
            else:
                print(f"  Correct snippet {target}: NOT in the top {k} "
                      f"(not in the top {len(ranked)} either)")

        rerank = (f"{timings['rerank'] * 1000:.1f} ms" if config.ENABLE_RERANK
                  else "off in config")
        print(f"  Time: retrieval {timings['retrieval'] * 1000:.1f} ms | "
              f"rerank {rerank} | total {timings['total'] * 1000:.1f} ms")
        print(rule("═"))

    def run_test_question(self, n: int, k: int) -> None:
        if not 1 <= n <= len(self.test_queries):
            print(f"  --from-test must be between 1 and {len(self.test_queries)}")
            return
        qid, text = self.test_queries[n - 1]
        print(rule("═"))
        print(f"  Test question {n} of {len(self.test_queries)}  (id {qid})")
        print(rule())
        lines = text.strip().splitlines()
        for line in lines[:QUESTION_LINES]:
            print(f"    {line}")
        if len(lines) > QUESTION_LINES:
            print(f"    ... ({len(lines) - QUESTION_LINES} more lines)")
        self.show(text, k, relevant=set(self.qrels.get(qid, {})))

    def interactive(self, k: int) -> None:
        while True:
            print()
            try:
                question = input("Ask a coding question (or 'q' to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if question.lower() in ("q", "quit", "exit"):
                return
            if question:
                print(rule("═"))
                self.show(question, k)


def _restart_online() -> None:
    """Re-run this script with the Hub enabled, to download what is missing.

    A restart, not a flag flip: huggingface_hub reads HF_HUB_OFFLINE once, at
    import time, so it cannot be switched on inside the running process.
    """
    print("\n  The model or dataset is not cached yet: restarting with network "
          "access to download it (one-time).", flush=True)
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[var] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--query", help="Ask one question and exit.")
    mode.add_argument("--from-test", type=int, metavar="N",
                      help="Run real test question N (1-based) and report the "
                           "correct snippet's rank.")
    parser.add_argument("--k", type=int, default=5,
                        help="How many results to show (default: 5).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show library logging and warnings.")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    else:
        quiet_libraries()

    try:
        session = SearchSession()
    except Exception:
        if not _FORCED_OFFLINE:
            raise
        _restart_online()
    if args.query is not None:
        print(rule("═"))
        session.show(args.query, args.k)
    elif args.from_test is not None:
        session.run_test_question(args.from_test, args.k)
    else:
        session.interactive(args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
