#!/usr/bin/env python3
"""Validate appsretrieval_results.json against what MTEB v2 actually expects.

A results file that scores fine locally but is malformed for submission is the
expensive failure: nothing errors, and you find out after the deadline. This
script checks the file three ways.

  1. STRUCTURE  - our wrapper carries the fields a human grader needs (task,
                  pipeline, split, config, the MTEB payload).
  2. ROUND-TRIP - the embedded MTEB payload parses back through MTEB's own
                  ``TaskResult`` model. This is the real test: if MTEB can
                  rehydrate it, it is a well-formed v2 result.
  3. AGREEMENT  - the scores in our file match, to the last decimal, the
                  canonical result MTEB wrote into its own cache during the
                  run. A mismatch means our serialisation lost or reshaped
                  something.

Usage
-----
    python scripts/validate_results.py
    python scripts/validate_results.py --results appsretrieval_results.json

Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

#: Fields our wrapper must carry for a run to be traceable.
REQUIRED_WRAPPER_FIELDS = (
    "task", "pipeline", "split", "release_tag", "config", "smoke_run", "results",
)

#: Fields MTEB v2 puts on every TaskResult.
REQUIRED_TASK_RESULT_FIELDS = (
    "task_name", "mteb_version", "dataset_revision", "scores", "evaluation_time",
)

#: Metrics the submission is graded on.
REQUIRED_METRICS = ("ndcg_at_10", "mrr_at_10", "main_score")


class Report:
    """Collects pass/fail lines so every check runs before we exit."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        suffix = f"  — {detail}" if detail else ""
        print(f"  [{mark}] {label}{suffix}")
        return ok

    def info(self, label: str, detail: str) -> None:
        print(f"  [ .. ] {label}  — {detail}")


def find_task_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the single MTEB TaskResult dict out of our wrapper."""
    results = payload.get("results")
    if isinstance(results, dict):
        task_results = results.get("task_results")
        if isinstance(task_results, list) and task_results:
            return task_results[0]
        if "task_name" in results:  # already a bare TaskResult
            return results
    return None


def canonical_cache_result(task_name: str) -> Path | None:
    """Locate the result MTEB wrote into its own cache for ``task_name``."""
    import os

    root = Path(os.environ.get("MTEB_CACHE", Path.home() / ".cache" / "mteb"))
    if not root.exists():
        return None
    matches = sorted(
        root.rglob(f"{task_name}.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=config.RESULTS_JSON)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"FATAL: {args.results} does not exist. Run scripts/run_eval.py first.")
        return 1

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rep = Report()

    # -- 1. structure ---------------------------------------------------------
    print(f"\n1. STRUCTURE  ({args.results.name})")
    for field in REQUIRED_WRAPPER_FIELDS:
        rep.check(field in payload, f"wrapper has '{field}'")
    rep.check(
        payload.get("task") == config.MTEB_TASK_NAME,
        "wrapper task name",
        f"{payload.get('task')!r} (expected {config.MTEB_TASK_NAME!r})",
    )
    if payload.get("smoke_run"):
        rep.info("smoke_run is true",
                 f"query_limit={payload.get('query_limit')} — NOT submittable")

    task_result = find_task_result(payload)
    if not rep.check(task_result is not None, "embedded MTEB TaskResult found"):
        print("\nCannot continue without a TaskResult.")
        return 1
    assert task_result is not None

    # -- 2. MTEB's own fields + round-trip ------------------------------------
    print("\n2. MTEB v2 CONTRACT")
    for field in REQUIRED_TASK_RESULT_FIELDS:
        rep.check(field in task_result, f"TaskResult has '{field}'")

    rep.check(
        task_result.get("task_name") == config.MTEB_TASK_NAME,
        "task_name",
        repr(task_result.get("task_name")),
    )
    rep.info("mteb_version", str(task_result.get("mteb_version")))
    rep.info("dataset_revision", str(task_result.get("dataset_revision")))

    scores = task_result.get("scores", {})
    rep.check(isinstance(scores, dict) and bool(scores), "scores is a non-empty dict")
    split = payload.get("split", "test")
    rep.check(split in scores, f"scores contains the '{split}' split",
              f"present: {list(scores)}")

    if split in scores:
        entries = scores[split]
        rep.check(isinstance(entries, list) and bool(entries),
                  f"scores['{split}'] is a non-empty list")
        entry = entries[0]
        rep.check("hf_subset" in entry, "subset entry has 'hf_subset'",
                  repr(entry.get("hf_subset")))
        rep.check("languages" in entry, "subset entry has 'languages'",
                  repr(entry.get("languages")))
        for metric in REQUIRED_METRICS:
            rep.check(metric in entry, f"metric '{metric}' present",
                      f"{entry.get(metric)}" if metric in entry else "MISSING")
        if "main_score" in entry and config.PRIMARY_METRIC in entry:
            rep.check(
                abs(entry["main_score"] - entry[config.PRIMARY_METRIC]) < 1e-9,
                f"main_score == {config.PRIMARY_METRIC}",
                f"{entry['main_score']} vs {entry[config.PRIMARY_METRIC]}",
            )

    try:
        # VERIFIED against mteb 2.20.11: TaskResult is re-exported at the top
        # level from mteb.results.task_result. It is NOT under
        # mteb.load_results, which is where the v1-era path pointed.
        from mteb import TaskResult

        TaskResult.model_validate(task_result)
        rep.check(True, "round-trips through mteb TaskResult.model_validate")
    except ImportError as exc:
        rep.check(False, "TaskResult importable from mteb", str(exc)[:160])
    except Exception as exc:
        rep.check(False, "round-trips through mteb TaskResult.model_validate",
                  f"{type(exc).__name__}: {str(exc)[:200]}")

    # -- 3. agreement with MTEB's own cached copy -----------------------------
    print("\n3. AGREEMENT WITH MTEB'S OWN OUTPUT")
    cached = canonical_cache_result(config.MTEB_TASK_NAME)
    if cached is None:
        rep.info("canonical copy", "none found under MTEB_CACHE; skipped")
    else:
        rep.info("canonical copy", str(cached))
        canon = json.loads(cached.read_text(encoding="utf-8"))
        canon_entry = (canon.get("scores", {}).get(split) or [{}])[0]
        ours = (scores.get(split) or [{}])[0]
        # MTEB rounds every score to 6 decimal places on write
        # (TaskResult._round_scores(..., 6)), while we serialise the in-memory
        # object before that happens. Our file therefore carries MORE
        # precision, not less - so compare at MTEB's own granularity.
        for metric in (config.PRIMARY_METRIC, config.SECONDARY_METRIC):
            if metric in canon_entry and metric in ours:
                rep.check(
                    abs(canon_entry[metric] - ours[metric]) <= 5e-7,
                    f"{metric} matches MTEB's cached value (6dp)",
                    f"ours={ours[metric]} cached={canon_entry[metric]}",
                )

    print("\n" + "=" * 58)
    if rep.failures:
        print(f"  {rep.failures} CHECK(S) FAILED — do not submit this file")
        print("=" * 58 + "\n")
        return 1
    print("  ALL CHECKS PASSED")
    if payload.get("smoke_run"):
        print("  (but smoke_run=true — rerun without --limit for the real file)")
    print("=" * 58 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
