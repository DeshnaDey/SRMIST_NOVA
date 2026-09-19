#!/usr/bin/env python3
"""Inspect the AppsRetrieval dataset and write data/inspection_report.md.

Answers the six questions we need settled before tuning anything:

  1. corpus / query / qrels counts, per split
  2. three full example queries with their relevant snippets, verbatim
  3. the exact shape of a corpus id and a query id
  4. token-length distribution of corpus snippets under the tokenizer of
     ``config.DENSE_MODEL_NAME``, and how much silent truncation we are eating
  5. relevant-docs-per-query distribution
  6. whether queries share a boilerplate prefix

Usage
-----
    python scripts/inspect_data.py
    python scripts/inspect_data.py --output data/inspection_report.md

SPLIT DISCIPLINE
----------------
Counts are reported for BOTH splits - a count is not an example, and we need
to know how big test is to estimate wall-clock.

Everything qualitative - the worked examples, the boilerplate-prefix analysis,
the query-length stats - is computed on TRAIN ONLY. The test split exists to
produce one official number at the end, and a heuristic designed by reading
test examples is a heuristic that has already leaked.

The corpus is the one shared object: both splits retrieve against the same
8,765 snippets, and we have to index all of them regardless of which split we
are scoring. Token stats are therefore reported over the full corpus (that is
the truncation we will actually eat) AND over the train-relevant subset alone,
so the two can be compared without taking anything on trust.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

logger = logging.getLogger("inspect_data")

#: How many worked examples to print in full.
N_EXAMPLES = 3

#: Queries sampled when hunting for a shared boilerplate prefix.
N_PREFIX_SAMPLE = 5_000


# =============================================================================
# loading
# =============================================================================


def load_task(splits: list[str]) -> Any:
    """Load AppsRetrieval with ``splits`` materialised.

    ``AbsTask.eval_splits`` is a read-only property backed by
    ``metadata.eval_splits`` (with a private ``_eval_splits`` override taking
    precedence). AppsRetrieval declares ``eval_splits == ["test"]`` only, so
    the train split has to be requested explicitly or ``load_data()`` will
    simply not materialise it. Verified against mteb 2.20.11.
    """
    import mteb

    task = mteb.get_task(config.MTEB_TASK_NAME)
    try:
        task._eval_splits = list(splits)
    except Exception:  # pragma: no cover - defensive
        task.metadata.eval_splits = list(splits)
    task.load_data()
    return task


def split_block(task: Any, split: str) -> dict[str, Any]:
    """Return the ``{corpus, queries, relevant_docs, top_ranked}`` block."""
    subset = task.hf_subsets[0]
    return task.dataset[subset][split]


# =============================================================================
# helpers
# =============================================================================


def corpus_text(row: dict[str, Any]) -> str:
    """Join title and body exactly as the pipeline does.

    Mirrors ``src.pipeline.prism_search._corpus_text`` and
    ``mteb_compat.extract_texts``. If this drifts, the token stats below stop
    describing the strings we actually encode.
    """
    body = str(row.get("text", "") or "")
    title = str(row.get("title", "") or "")
    return f"{title}\n\n{body}" if title.strip() else body


def percentiles(values: list[int]) -> dict[str, float]:
    """p50/p90/p99/max/mean over ``values``."""
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> int:
        # Nearest-rank: index of the smallest value >= p% of the data.
        idx = min(len(ordered) - 1, int(round(p / 100.0 * len(ordered) + 0.5)) - 1)
        return ordered[max(0, idx)]

    return {
        "min": ordered[0],
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 1),
    }


def longest_common_prefix(strings: list[str]) -> str:
    """Longest string that every entry of ``strings`` starts with."""
    if not strings:
        return ""
    first, shortest = strings[0], min(strings, key=len)
    limit = len(shortest)
    i = 0
    while i < limit and all(s[i] == first[i] for s in strings):
        i += 1
    return first[:i]


def leading_ngram_counts(queries: list[str], n_words: int) -> list[tuple[str, int]]:
    """Most common openings of length ``n_words``, as (phrase, count)."""
    counter: Counter[str] = Counter()
    for q in queries:
        words = q.split()
        if len(words) >= n_words:
            counter[" ".join(words[:n_words])] += 1
    return counter.most_common(5)


def fence(text: str, lang: str = "") -> str:
    """Wrap ``text`` in a fence long enough to survive backticks inside it."""
    longest = 0
    run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    bar = "`" * max(3, longest + 1)
    return f"{bar}{lang}\n{text}\n{bar}"


# =============================================================================
# the six findings
# =============================================================================


def analyse_counts(task: Any, splits: list[str]) -> tuple[str, dict[str, Any]]:
    """Finding 1 - counts per split."""
    lines = [
        "| Split | Corpus | Queries | Qrels (query→doc pairs) |",
        "|---|---:|---:|---:|",
    ]
    data: dict[str, Any] = {}
    for split in splits:
        blk = split_block(task, split)
        n_corpus = blk["corpus"].num_rows
        n_queries = blk["queries"].num_rows
        n_qrels = sum(len(v) for v in blk["relevant_docs"].values())
        data[split] = {
            "corpus": n_corpus,
            "queries": n_queries,
            "qrels_pairs": n_qrels,
            "qrels_queries": len(blk["relevant_docs"]),
        }
        lines.append(f"| `{split}` | {n_corpus:,} | {n_queries:,} | {n_qrels:,} |")

    corpora = {d["corpus"] for d in data.values()}
    note = (
        "\nThe corpus is **identical across splits** "
        f"({corpora.pop():,} snippets) — both splits retrieve against the same "
        "pool, only the query set and qrels differ."
        if len(corpora) == 1
        else "\nCorpus size differs between splits."
    )
    return "\n".join(lines) + "\n" + note, data


def analyse_examples(task: Any, split: str) -> tuple[str, list[dict[str, Any]]]:
    """Finding 2 - N worked examples, verbatim. TRAIN ONLY."""
    blk = split_block(task, split)
    queries = {r["id"]: r for r in blk["queries"]}
    corpus = {r["id"]: r for r in blk["corpus"]}
    qrels = blk["relevant_docs"]

    out: list[str] = []
    captured: list[dict[str, Any]] = []
    # Sorted for determinism - a different example every run makes the report
    # impossible to diff.
    for qid in sorted(qrels)[:N_EXAMPLES]:
        qrow = queries.get(qid, {})
        qtext = str(qrow.get("text", "<MISSING>"))
        out.append(f"#### Example — query `{qid}`\n")
        out.append(f"*query length: {len(qtext):,} chars*\n")
        out.append(fence(qtext))
        for cid, rel in sorted(qrels[qid].items()):
            crow = corpus.get(cid, {})
            ctext = corpus_text(crow) if crow else "<MISSING FROM CORPUS>"
            out.append(f"\n**Relevant snippet `{cid}` (relevance={rel})** — "
                       f"{len(ctext):,} chars\n")
            out.append(fence(ctext, "python"))
            captured.append({"query_id": qid, "doc_id": cid, "relevance": rel})
        out.append("\n---\n")
    return "\n".join(out), captured


def analyse_id_format(task: Any, splits: list[str]) -> tuple[str, dict[str, Any]]:
    """Finding 3 - exact id shapes."""
    lines: list[str] = []
    info: dict[str, Any] = {}
    blk = split_block(task, splits[0])

    for label, ds in (("corpus id", blk["corpus"]), ("query id", blk["queries"])):
        sample = [str(r["id"]) for r in ds.select(range(min(2000, ds.num_rows)))]
        lengths = sorted({len(s) for s in sample})
        prefixes = Counter(s[:1] for s in sample)
        all_numeric_tail = all(s[1:].isdigit() for s in sample)
        key = label.replace(" ", "_")
        info[key] = {
            "python_type": type(ds[0]["id"]).__name__,
            "length_range": [lengths[0], lengths[-1]],
            "prefix_chars": dict(prefixes),
            "numeric_after_prefix": all_numeric_tail,
            "samples": sample[:5],
        }
        lines.append(
            f"**{label}** — type `{type(ds[0]['id']).__name__}`, "
            f"length {lengths[0]}–{lengths[-1]} chars, "
            f"prefix `{'`/`'.join(prefixes)}`, "
            f"numeric after prefix: {all_numeric_tail}  \n"
            f"samples: {', '.join('`' + s + '`' for s in sample[:5])}"
        )

    # Do the q<N>/d<N> numbers line up? If so that is a trap worth naming.
    qrels = blk["relevant_docs"]
    paired = [
        (q, d)
        for q, docs in list(qrels.items())[:2000]
        for d in docs
    ]
    identity = sum(1 for q, d in paired if q[1:] == d[1:])
    info["qid_matches_docid_numerically"] = {
        "checked": len(paired),
        "matching": identity,
        "fraction": round(identity / len(paired), 4) if paired else 0.0,
    }
    lines.append(
        f"\n**Numeric correspondence:** of {len(paired):,} qrel pairs checked, "
        f"**{identity:,} ({identity / len(paired):.1%})** have a query id whose "
        f"number equals its relevant doc id's number (`q3382` → `d3382`)."
    )
    return "\n".join(lines), info


def analyse_token_lengths(
    task: Any, split: str, train_doc_ids: set[str]
) -> tuple[str, dict[str, Any]]:
    """Finding 4 - token-length distribution and truncation loss."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.DENSE_MODEL_NAME, device=config.DEVICE)
    if config.MAX_SEQ_LENGTH is not None:
        model.max_seq_length = config.MAX_SEQ_LENGTH
    max_seq = int(model.max_seq_length)
    tok = model.tokenizer

    blk = split_block(task, split)
    corpus = list(blk["corpus"])
    texts = [corpus_text(r) for r in corpus]
    ids = [str(r["id"]) for r in corpus]

    # add_special_tokens=False measures the content; the model then spends 2
    # of its max_seq budget on [CLS]/[SEP], which the effective budget reflects.
    lengths: list[int] = []
    for i in range(0, len(texts), 256):
        enc = tok(texts[i : i + 256], add_special_tokens=False,
                  truncation=False, return_length=True)
        lengths.extend(enc["length"])

    effective = max_seq - 2
    train_lengths = [ln for ln, cid in zip(lengths, ids) if cid in train_doc_ids]

    def block(label: str, vals: list[int]) -> tuple[str, dict[str, Any]]:
        p = percentiles(vals)
        over = sum(1 for v in vals if v > effective)
        kept = [min(v, effective) / v for v in vals if v > 0]
        stats = {
            **p,
            "n": len(vals),
            "over_limit": over,
            "over_limit_pct": round(100 * over / len(vals), 1) if vals else 0.0,
            "mean_fraction_kept": round(100 * statistics.fmean(kept), 1) if kept else 0.0,
        }
        row = (
            f"| {label} | {stats['n']:,} | {p['p50']:,} | {p['p90']:,} | "
            f"{p['p99']:,} | {p['max']:,} | **{stats['over_limit_pct']}%** | "
            f"{stats['mean_fraction_kept']}% |"
        )
        return row, stats

    row_all, stats_all = block("Full corpus", lengths)
    row_tr, stats_tr = block("Train-relevant only", train_lengths)

    # Queries go through the same encoder and the same window. Item 4 only
    # asks about snippets, but a query that is silently cut in half is the
    # same failure on the other side of the dot product, so measure it here
    # while the tokenizer is already loaded.
    qds = blk["queries"]
    qtexts = [str(r["text"]) for r in qds]
    qlengths: list[int] = []
    for i in range(0, len(qtexts), 256):
        enc = tok(qtexts[i : i + 256], add_special_tokens=False,
                  truncation=False, return_length=True)
        qlengths.extend(enc["length"])
    row_q, stats_q = block(f"Queries (`{split}`)", qlengths)

    text = (
        f"Tokenizer: **`{config.DENSE_MODEL_NAME}`**  \n"
        f"Model `max_seq_length` = **{max_seq}** tokens "
        f"(**{effective}** for content, after `[CLS]`/`[SEP]`)  \n"
        f"`config.MAX_SEQ_LENGTH` = `{config.MAX_SEQ_LENGTH}` "
        f"({'checkpoint default' if config.MAX_SEQ_LENGTH is None else 'overridden'})\n\n"
        "| Subset | N | p50 | p90 | p99 | max | % over limit | mean % of tokens kept |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{row_all}\n{row_tr}\n{row_q}\n"
    )
    return text, {"max_seq_length": max_seq, "effective": effective,
                  "full_corpus": stats_all, "train_relevant": stats_tr,
                  "queries": stats_q}


def analyse_rels_per_query(task: Any, splits: list[str]) -> tuple[str, dict[str, Any]]:
    """Finding 5 - relevant-docs-per-query distribution."""
    lines = ["| Split | 1 rel | 2 rels | 3+ rels | max | mean |",
             "|---|---:|---:|---:|---:|---:|"]
    info: dict[str, Any] = {}
    for split in splits:
        qrels = split_block(task, split)["relevant_docs"]
        counts = [len(v) for v in qrels.values()]
        dist = Counter(counts)
        info[split] = {
            "distribution": dict(sorted(dist.items())),
            "max": max(counts),
            "mean": round(statistics.fmean(counts), 3),
        }
        three_plus = sum(c for k, c in dist.items() if k >= 3)
        lines.append(
            f"| `{split}` | {dist.get(1, 0):,} | {dist.get(2, 0):,} | "
            f"{three_plus:,} | {max(counts)} | {statistics.fmean(counts):.3f} |"
        )
    return "\n".join(lines), info


def analyse_boilerplate(task: Any, split: str) -> tuple[str, dict[str, Any]]:
    """Finding 6 - shared boilerplate prefix across queries. TRAIN ONLY."""
    blk = split_block(task, split)
    ds = blk["queries"]
    texts = [str(r["text"]) for r in ds.select(range(min(N_PREFIX_SAMPLE, ds.num_rows)))]

    lcp = longest_common_prefix(texts)
    char_lengths = percentiles([len(t) for t in texts])

    lines = [
        f"Sampled **{len(texts):,}** `{split}` queries.\n",
        f"- Longest common prefix across *every* sampled query: "
        f"{'`' + lcp + '`' if lcp else '**none** (empty string)'}",
        f"- Query length (chars): p50 {char_lengths['p50']:,}, "
        f"p90 {char_lengths['p90']:,}, p99 {char_lengths['p99']:,}, "
        f"max {char_lengths['max']:,}",
    ]
    ngrams: dict[str, Any] = {}
    for n in (1, 3, 5, 8):
        top = leading_ngram_counts(texts, n)
        ngrams[f"{n}_word"] = [
            {"phrase": p, "count": c, "pct": round(100 * c / len(texts), 1)}
            for p, c in top
        ]
        rendered = "; ".join(
            f"`{p}` ({100 * c / len(texts):.1f}%)" for p, c in top[:3]
        )
        lines.append(f"- Most common **{n}-word** opening: {rendered}")

    return "\n".join(lines), {
        "longest_common_prefix": lcp,
        "query_char_lengths": char_lengths,
        "leading_ngrams": ngrams,
    }


# =============================================================================
# main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROJECT_ROOT / "data" / "inspection_report.md",
        help="Where to write the markdown report.",
    )
    parser.add_argument(
        "--example-split",
        default="train",
        help="Split used for worked examples and boilerplate analysis "
             "(default: train — do not point this at test).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    if args.example_split == "test":
        logger.warning(
            "--example-split=test inspects TEST examples. That is how a "
            "heuristic leaks. Continuing because you asked explicitly."
        )

    splits = ["train", "test"]
    logger.info("Loading %s (splits=%s)", config.MTEB_TASK_NAME, splits)
    task = load_task(splits)

    train_qrels = split_block(task, "train")["relevant_docs"]
    train_doc_ids = {d for docs in train_qrels.values() for d in docs}

    logger.info("1/6 counts")
    counts_md, counts_data = analyse_counts(task, splits)
    logger.info("2/6 examples (%s)", args.example_split)
    examples_md, examples_data = analyse_examples(task, args.example_split)
    logger.info("3/6 id formats")
    ids_md, ids_data = analyse_id_format(task, splits)
    logger.info("4/6 token lengths (loads %s)", config.DENSE_MODEL_NAME)
    tokens_md, tokens_data = analyse_token_lengths(task, "train", train_doc_ids)
    logger.info("5/6 relevant docs per query")
    rels_md, rels_data = analyse_rels_per_query(task, splits)
    logger.info("6/6 boilerplate prefix (%s)", args.example_split)
    boiler_md, boiler_data = analyse_boilerplate(task, args.example_split)

    report = f"""# AppsRetrieval — data inspection report

Generated by `scripts/inspect_data.py`.

- Task: **`{config.MTEB_TASK_NAME}`** · dataset **`{config.DATASET_ID}`**
- Dataset revision: `{task.metadata.dataset.get('revision', 'unknown')}`
- Tokenizer / model under inspection: **`{config.DENSE_MODEL_NAME}`**
- Worked examples and query analysis use the **`{args.example_split}`** split only.

---

## 1. Counts

{counts_md}

---

## 2. Example queries and their relevant snippets (`{args.example_split}` split)

{examples_md}

## 3. Id format

{ids_md}

---

## 4. Corpus token-length distribution

{tokens_md}

---

## 5. Relevant documents per query

{rels_md}

---

## 6. Query boilerplate

{boiler_md}

---

## Machine-readable summary

{fence(json.dumps({
    "counts": counts_data,
    "ids": ids_data,
    "tokens": tokens_data,
    "rels_per_query": rels_data,
    "boilerplate": boiler_data,
    "examples": examples_data,
}, indent=2), "json")}
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    logger.info("Wrote %s (%,d bytes)".replace("%,d", "%d"), args.output,
                len(report.encode()))

    print("\n" + "=" * 60)
    print(f"  Wrote {args.output}")
    print("=" * 60)
    print(counts_md)
    print()
    print(tokens_md)
    print(rels_md)
    print()
    print(boiler_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
