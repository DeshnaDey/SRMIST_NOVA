# Experiment Log

Every evaluation run gets a row. No exceptions — an unlogged experiment is an
experiment we will run again by accident, and a number nobody can reproduce is
worth less than no number at all.

## How to log a run

1. Run the eval: `python scripts/run_eval.py --pipeline full`
2. Copy the printed NDCG@10 and MRR into a new row **at the bottom**.
3. Change **one thing at a time**. Two changes in one row means you learn
   nothing about either.
4. Record the commit SHA — that plus `src/config.py` must be enough to
   reproduce the row exactly.

## Rules

- **Smoke runs (`--limit N`) never get a row.** They are not comparable.
- **A change that makes things worse still gets a row.** Negative results stop
  the next person retrying the same idea at 3am.
- Note wall-clock time when it changes materially. A config that scores +0.01
  but takes four hours on CPU is not a config we can submit.

## Results

| Date | Who | Change made | NDCG@10 | MRR | Wall-clock | Commit | Notes |
|------|-----|-------------|---------|-----|------------|--------|-------|
| 2026-09-19 | DeshnaDey | **Baseline**: bare bi-encoder `sentence-transformers/all-MiniLM-L6-v2`, no preprocessing, no BM25, no rerank | **0.06596** | 0.05581 | 4.1 min | `7a807b2` | Full `test` split (3,765 queries, 8,765 corpus). The number every later row is measured against. recall@10 0.0991, recall@100 0.2526 — the relevant doc is in the top 100 a quarter of the time, so there is real headroom for reranking. 61.9% of queries exceed the model's 254-token window (see `data/inspection_report.md`). |
| 2026-09-19 | DeshnaDey | **Model swap**: `all-MiniLM-L6-v2` → `Snowflake/snowflake-arctic-embed-m` (512 ctx, query-only instruction prefix). Selected on full TRAIN by recall@100 over e5-base-v2, bge-base-en-v1.5, all-MiniLM | **0.08222** | 0.06799 | 26.2 min | `f20bb96` | Full `test`. **recall@100 0.30677** (baseline 0.25259, +21.4%) — the metric that matters, since it caps what a reranker can recover. recall@10 0.12855 (+29.7%). Costs 6.4x wall-clock and 8.3x query latency (161 ms vs 19 ms). jina-v2-base-code could not be benchmarked: its remote code imports `find_pruneable_heads_and_indices`, removed in transformers 5.x. |
| | | | | | | | |

## Decision log

Measured decisions that did **not** produce a scored row, recorded so nobody
relitigates them. A gate that says "no" is a result.

### 2026-09-20 — BM25 + RRF fusion: GATED OUT before building

Run `python scripts/bm25_diagnostic.py --split train`; raw numbers in
`data/bm25_diagnostic.json`. Full train split, 5,000 queries, top-100,
arctic-embed-m as the dense side.

**(a) Level — passed.** BM25-only recall@100 **0.4224** (recall@10 0.2368),
against arctic's 0.6870. That is 61.5% of dense, nowhere near zero, so the
lexical-overlap premise is sound: BM25 alone finds the gold snippet in the top
100 for 42% of queries, at 0.7 ms/query and a 0.9 s index build.

**(b) Complementarity — failed.** Fusion only pays when two systems fail on
*different* queries:

| | queries | share |
|---|---:|---:|
| both hit | 1,979 | 39.6% |
| **BM25 only** | **133** | **2.7%** |
| dense only | 1,456 | 29.1% |
| neither | 1,432 | 28.6% |

**93.7% of BM25's hits are queries arctic already had.** The union ceiling is
**0.7136** against arctic's 0.6870 — so **+0.027 is the most any fusion of
these two lists could ever add**, and that is an oracle bound. Real RRF
captures a fraction of it while also pushing some dense-only hits out of the
top 100. Not worth wiring BM25IndexBuilder / BM25Retriever for.

**The numeric-shredding confound was checked, and was not the cause.**
`10^9+7` tokenized to `['10','9','7']` while solutions carry the literal
`1000000007`, which would suppress exactly the overlap the gate tests. Fixed
(both sides normalise power-forms to the literal; `1e9+7` is scientific
notation, not `1^9+7`). Effect: BM25-only 131 → **133** queries, union ceiling
0.7132 → **0.7136**. Real, and negligible. The weak complementarity is
genuine, not a tokenizer artifact.

**Where the headroom actually is.** Dense recall@100 stratified by query
length (budget 510 tokens):

| bucket | | n | share | recall@100 |
|---|---|---:|---:|---:|
| 0–510 tok | fits | 3,770 | 75.4% | **0.7493** |
| 510–1020 tok | truncated | 1,098 | 22.0% | **0.4827** |
| 1020–2040 tok | truncated | 122 | 2.4% | 0.5902 |
| 2040+ tok | truncated | 10 | 0.2% | 0.8000 |

Queries that fit score 0.749; queries just over the budget score 0.483. And
truncation is far worse on the graded split: **24.6% of train queries exceed
510 tokens against 41.2% of test.**

Be careful how much that explains, though. Re-weighting train's per-bucket
recalls to test's length distribution gives 0.639 — about **4.8 points of the
38-point train→test drop** (0.687 → 0.307). Truncation is a real cost, not the
main one; the rest is intrinsic difficulty.

And the 27-point fits-vs-truncated gap is **correlational**: longer statements
may simply be harder problems. Before building query compression, run the
causal probe now in the backlog — artificially truncate queries that currently
fit and see whether recall actually falls. If it does not, compression is
wasted effort too, and the remaining headroom is in the encoder itself
(fine-tuning, hard-negative mining) rather than in what we feed it.

## Backlog — ideas not yet measured

Move a row into the table above once it has a number next to it.

| Idea | Owner | Rationale | Status |
|------|-------|-----------|--------|
| Swap in a code-specific bi-encoder | retrieval | General-purpose embeddings are trained on prose, not Python | not started |
| ~~Enable BM25 + RRF fusion~~ | retrieval | **GATED OUT — measured, see Decision log below.** BM25-only recall@100 is a respectable 0.4224, but it recovers only **133 queries (2.7%)** that arctic missed. Union ceiling 0.7136 vs arctic 0.6870: **+0.027 is the absolute best any fusion could reach**, before RRF gives some back. | dropped |
| Cross-encoder rerank of top-50 | retrieval | Reordering the top candidates is literally what NDCG@10 measures | not started |
| Strip comments from snippets | corpus | Less noise — but comments may be the only NL bridge to the query. Test both directions | not started |
| Truncate snippets head vs. tail | corpus | APPS solutions exceed the context window; which half carries the signal? Note TASK A: only 23.5% of snippets overflow, vs **61.9% of queries** — the query side is the bigger loss and has no knob yet. | not started |
| Compress query: drop Input/Output format sections and worked examples, keep the narrative problem core | query | Queries truncate **61.9%** at 254 tokens and the example blocks are formatting noise eating the budget. (Supersedes "strip boilerplate framing", which assumed a shared prefix that does not exist — 2.0%, not every query.) | not started |
| Index `meta_information.starter_code` alongside the snippet body | corpus | Every corpus row has a populated `meta_information` dict with a `starter_code` field; we index `text` only, so it is signal the retriever never sees. Cheap to test. | not started |
| Query category routing | query | Different `QUERY_CATEGORIES` may want different top-k or fusion weights | not started |
| Tune RRF `k` | retrieval | 60 is the paper default, not a measured optimum for this corpus | not started |
| Split snake_case/camelCase in BM25 tokenizer | corpus | Lets "binary search" match `binary_search` | not started |
| **Causal probe: truncate a fitting query, measure the recall drop** | query | Queries that fit the 510-token window score recall@100 0.7493; those just over it score 0.4827. That 27-point gap may be truncation OR may just be that longer problems are harder. Artificially truncating queries that currently FIT separates the two, and decides whether query compression is worth building at all. Cheap: embeddings for the corpus are already cached. | not started |
| Chunk long snippets with overlap | corpus | Avoids truncating away the relevant function | not started |
