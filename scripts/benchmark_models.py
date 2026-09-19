#!/usr/bin/env python3
"""Benchmark one candidate bi-encoder on a split and append the row to JSON.

ONE MODEL PER INVOCATION, on purpose. Each run is a fresh process, so a 1.5B
checkpoint cannot leave its weights resident and skew the next model's timing
on an 8 GB box. Results accumulate in ``data/model_benchmark.json``.

Usage
-----
    python scripts/benchmark_models.py --model jinaai/jina-embeddings-v2-base-code
    python scripts/benchmark_models.py --model all-MiniLM --split train
    python scripts/benchmark_models.py --list

WHY recall@100 IS THE HEADLINE HERE
-----------------------------------
There is exactly one gold document per query, and the baseline finds it inside
the top 100 only 25.3% of the time. Everything downstream - fusion, and above
all the cross-encoder reranker - can only reorder what retrieval already
surfaced, so recall@100 is a hard ceiling on the whole pipeline. A model that
wins on NDCG@10 but loses on recall@100 is the worse choice here.

THE TWO WAYS TO GET A SILENTLY WRONG NUMBER
-------------------------------------------
1. PREFIXES. e5, bge and arctic each expect a specific query (and sometimes
   passage) prefix. Omit it and nothing errors - the score is just quietly
   bad. Every entry in ``CANDIDATES`` records its scheme and the exact place
   in the model card it came from, so a suspicious result can be checked
   against the source rather than re-guessed.
2. OUTLIER SNIPPETS. The corpus max is 60,599 tokens. Under a 254-token model
   that is truncated away harmlessly; under an 8192-token model it is not, and
   attention is quadratic in sequence length, so a handful of documents can
   dominate the encode. Every model is therefore capped at
   ``SNIPPET_TOKEN_CAP`` regardless of how much context it advertises.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from scripts.run_eval import prepare_task, to_serializable  # noqa: E402

logger = logging.getLogger("benchmark")

#: Hard ceiling on the encoder window, whatever the checkpoint advertises.
#: p99 of the corpus is 1,023 tokens, so this binds on very few documents while
#: capping the quadratic-attention blowup from the long tail.
SNIPPET_TOKEN_CAP = 2048

#: Each candidate's prefix scheme, with the provenance of that decision.
CANDIDATES: dict[str, dict[str, Any]] = {
    "all-MiniLM": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "document_prefix": "",
        "scheme": "none",
        "prefix_source": "Incumbent baseline. Card documents no prefix scheme.",
        "published_coir_apps": None,
    },
    "jina-code": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "query_prefix": "",
        "document_prefix": "",
        "scheme": "none",
        "prefix_source": (
            "Model card usage examples embed raw strings directly; no prefix "
            "documented anywhere on the card. 161M params, 8192 ctx via ALiBi, "
            "mean pooling + L2 norm."
        ),
        "published_coir_apps": None,
    },
    "e5-base": {
        "name": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "scheme": "query: / passage:",
        "prefix_source": (
            "Model card FAQ, question 1: 'Each input text should start with "
            "\"query: \" or \"passage: \".' Asymmetric retrieval uses both."
        ),
        "published_coir_apps": 11.52,
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
        "scheme": "query-only instruction",
        "prefix_source": (
            "Model card 'Usage for Retrieval': instruction on the query only, "
            "'no instruction needs to be added to passages'. v1.5 notes the "
            "instruction is now optional with only slight degradation."
        ),
        "published_coir_apps": 4.05,
    },
    "arctic-m": {
        "name": "Snowflake/snowflake-arctic-embed-m",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
        "scheme": "query-only instruction",
        "prefix_source": (
            "Model card usage: 'use the query prefix below (just on the "
            "query)'. Initialised from intfloat/e5-base-unsupervised."
        ),
        "published_coir_apps": None,
    },
    "qodo": {
        "name": "Qodo/Qodo-Embed-1-1.5B",
        "query_prefix": "",
        "document_prefix": "",
        "scheme": "none",
        "prefix_source": "Card shows plain SentenceTransformer.encode with no prefix.",
        "published_coir_apps": None,
    },
}


class TimedEncoder:
    """Wraps BaselineEncoder to split encode time into corpus vs query.

    MTEB does not report the two separately, but they answer different
    questions: corpus encode is a one-off indexing cost we pay once and can
    cache, while query latency is what the demo actually shows a judge.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.doc_seconds = 0.0
        self.query_seconds = 0.0
        self.doc_items = 0
        self.query_items = 0

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def encode(self, inputs: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        out = self._inner.encode(inputs, **kwargs)
        elapsed = time.perf_counter() - started
        n = len(out) if hasattr(out, "__len__") else 0

        name = str(kwargs.get("prompt_type", "")).lower()
        if "query" in name:
            self.query_seconds += elapsed
            self.query_items += n
        else:
            self.doc_seconds += elapsed
            self.doc_items += n
        return out


def truncation_stats(model_name: str, split: str, window: int) -> dict[str, Any]:
    """What fraction of queries and snippets overflow ``window`` tokens."""
    # Go through the shared loader, not SentenceTransformer directly: it is
    # what applies the remote-code allowlist and the attention-backend
    # fallback, and it guarantees we measure with the SAME tokenizer that will
    # do the encoding.
    from src.pipeline.baseline import _load_sentence_transformer

    tok = _load_sentence_transformer(model_name).tokenizer
    task = prepare_task(split, None)
    block = task.dataset[task.hf_subsets[0]][split]

    def over(texts: list[str]) -> tuple[float, int]:
        lengths: list[int] = []
        for i in range(0, len(texts), 128):
            lengths.extend(
                tok(texts[i : i + 128], add_special_tokens=False,
                    truncation=False, return_length=True)["length"]
            )
        pct = 100.0 * sum(1 for x in lengths if x > window) / len(lengths)
        return round(pct, 1), max(lengths)

    q_pct, q_max = over([str(r["text"]) for r in block["queries"]])
    d_pct, d_max = over([str(r["text"]) for r in block["corpus"]])
    return {
        "queries_truncated_pct": q_pct,
        "query_max_tokens": q_max,
        "snippets_truncated_pct": d_pct,
        "snippet_max_tokens": d_max,
    }


def run_one(key: str, split: str, results_path: Path) -> dict[str, Any]:
    """Benchmark one candidate and append its row to ``results_path``."""
    import mteb

    from src.pipeline.baseline import BaselineEncoder

    spec = CANDIDATES[key]
    model_name = spec["name"]

    # Point config at this candidate BEFORE building anything: BaselineEncoder
    # and the budget resolver both read these at construction time.
    config.DENSE_MODEL_NAME = model_name
    config.QUERY_PROMPT_PREFIX = spec["query_prefix"]
    config.DOCUMENT_PROMPT_PREFIX = spec["document_prefix"]

    advertised = config.model_context_tokens(model_name)
    window = min(advertised, SNIPPET_TOKEN_CAP)
    config.MAX_SEQ_LENGTH = window

    logger.info("=" * 68)
    logger.info("MODEL   %s", model_name)
    logger.info("prefix  %s  (%s)", spec["scheme"] or "none", spec["prefix_source"][:60])
    logger.info("window  advertised %d -> capped to %d tokens", advertised, window)
    logger.info("=" * 68)

    trunc = truncation_stats(model_name, split, window)
    logger.info("truncation at %d: queries %.1f%%, snippets %.1f%%",
                window, trunc["queries_truncated_pct"], trunc["snippets_truncated_pct"])

    encoder = TimedEncoder(BaselineEncoder(model_name=model_name))
    task = prepare_task(split, None)

    started = time.perf_counter()
    results = mteb.evaluate(
        encoder,
        tasks=[task],
        encode_kwargs={"batch_size": config.BATCH_SIZE},
        overwrite_strategy="always",
    )
    wall = time.perf_counter() - started

    payload = to_serializable(results)
    scores: dict[str, Any] = {}
    try:
        scores = payload["task_results"][0]["scores"][split][0]
    except Exception:
        logger.error("Could not locate scores; dumping payload shape %s", type(payload))

    n_queries = encoder.query_items or 1
    row = {
        "key": key,
        "model": model_name,
        "split": split,
        "advertised_tokens": advertised,
        "window_used": window,
        **trunc,
        "prefix_scheme": spec["scheme"],
        "prefix_source": spec["prefix_source"],
        "published_coir_apps": spec["published_coir_apps"],
        "recall_at_100": scores.get("recall_at_100"),
        "recall_at_10": scores.get("recall_at_10"),
        "ndcg_at_10": scores.get("ndcg_at_10"),
        "mrr_at_10": scores.get("mrr_at_10"),
        "corpus_encode_seconds": round(encoder.doc_seconds, 1),
        "corpus_items": encoder.doc_items,
        "query_encode_seconds": round(encoder.query_seconds, 1),
        "query_items": encoder.query_items,
        "per_query_latency_ms": round(1000 * encoder.query_seconds / n_queries, 2),
        "wall_clock_seconds": round(wall, 1),
    }

    existing = []
    if results_path.exists():
        existing = json.loads(results_path.read_text(encoding="utf-8"))
    existing = [r for r in existing if not (r["key"] == key and r["split"] == split)]
    existing.append(row)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print("\n" + "-" * 68)
    print(f"  {model_name}  ({split})")
    print("-" * 68)
    print(f"  recall@100   {row['recall_at_100']}        <- PRIMARY")
    print(f"  recall@10    {row['recall_at_10']}")
    print(f"  NDCG@10      {row['ndcg_at_10']}")
    print(f"  MRR@10       {row['mrr_at_10']}")
    print(f"  corpus encode {row['corpus_encode_seconds']}s "
          f"({row['corpus_items']} docs)")
    print(f"  query latency {row['per_query_latency_ms']} ms/query")
    print(f"  wall clock    {row['wall_clock_seconds']}s")
    print("-" * 68 + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help=f"one of: {', '.join(CANDIDATES)}")
    parser.add_argument("--split", default="train",
                        help="Model selection happens on train (default: train).")
    parser.add_argument("--results", type=Path,
                        default=config.PROJECT_ROOT / "data" / "model_benchmark.json")
    parser.add_argument("--list", action="store_true", help="List candidates and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    if args.list:
        for key, spec in CANDIDATES.items():
            print(f"  {key:12s} {spec['name']:45s} prefix={spec['scheme'] or 'none'}")
        return 0

    if args.model not in CANDIDATES:
        print(f"Unknown model {args.model!r}. Options: {', '.join(CANDIDATES)}")
        return 1

    run_one(args.model, args.split, args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
