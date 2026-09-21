#!/usr/bin/env python3
"""Bake the model and dataset into the image so the graded run needs no network.

WHY THE ID AND REVISION ARE NOT WRITTEN DOWN HERE
-------------------------------------------------
They are read from the MTEB task itself. A prefetch that hardcodes
"CoIR-Retrieval/apps" is correct until mteb pins a different revision, and
then it is a NEAR-MISS: the build succeeds, the cache looks populated, and
the graded run quietly reaches for the Hub anyway because it wants a
revision that was never downloaded. Deriving both from
``task.metadata.dataset`` makes that class of drift impossible.

The same applies to the model: it is loaded through ``SentenceTransformer``
rather than ``snapshot_download`` so that exactly the files the runtime
loader asks for are the files that land in the cache - config,
sentence_bert_config, the pooling module, the tokenizer and the weights.

VERIFY, DO NOT ASSUME
---------------------
``--verify`` re-loads both with ``HF_HUB_OFFLINE=1`` set, which is the only
honest check that the prefetch covers the runtime load. A prefetch that is
merely *plausible* is worth nothing at grade time.

Usage
-----
    python scripts/prefetch_assets.py            # download into HF_HOME
    python scripts/prefetch_assets.py --verify   # then prove it works offline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

logger = logging.getLogger("prefetch")


def dataset_coordinates() -> tuple[str, str | None]:
    """Return (repo_id, revision) exactly as MTEB will request them."""
    import mteb

    task = mteb.get_task(config.MTEB_TASK_NAME)
    spec = dict(getattr(task.metadata, "dataset", {}) or {})
    repo_id = spec.get("path")
    if not repo_id:
        raise RuntimeError(
            f"{config.MTEB_TASK_NAME} metadata has no dataset path; cannot "
            "prefetch. Inspect task.metadata.dataset."
        )
    return str(repo_id), spec.get("revision")


def prefetch(verify_only: bool = False) -> int:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    repo_id, revision = dataset_coordinates()
    logger.info("Dataset : %s @ %s", repo_id, revision or "(no pinned revision)")
    logger.info("Model   : %s", config.DENSE_MODEL_NAME)

    if not verify_only:
        snapshot_download(repo_id=repo_id, revision=revision,
                          repo_type="dataset")
        logger.info("Dataset cached")
        SentenceTransformer(config.DENSE_MODEL_NAME, device="cpu")
        logger.info("Model cached")
    return 0


def verify() -> int:
    """Load both with the Hub switched off. Fails loudly if anything is missing."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    import mteb
    from sentence_transformers import SentenceTransformer

    logger.info("Offline check: loading the model with HF_HUB_OFFLINE=1")
    SentenceTransformer(config.DENSE_MODEL_NAME, device="cpu")

    logger.info("Offline check: materialising the task's data")
    task = mteb.get_task(config.MTEB_TASK_NAME)
    try:
        task._eval_splits = ["test"]
    except Exception:  # pragma: no cover - defensive
        task.metadata.eval_splits = ["test"]
    task.load_data()
    block = task.dataset[task.hf_subsets[0]]["test"]
    n_corpus = len(list(block["corpus"]))
    n_queries = len(list(block["queries"]))
    logger.info("Offline check OK: %d corpus rows, %d queries", n_corpus, n_queries)
    if n_corpus == 0 or n_queries == 0:
        logger.error("Offline load produced an empty split")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="Re-load model and dataset with the Hub disabled.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip downloading; only run the offline check.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    if not args.verify_only:
        rc = prefetch()
        if rc:
            return rc
    if args.verify or args.verify_only:
        return verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
