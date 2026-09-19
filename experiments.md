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

## Backlog — ideas not yet measured

Move a row into the table above once it has a number next to it.

| Idea | Owner | Rationale | Status |
|------|-------|-----------|--------|
| Swap in a code-specific bi-encoder | retrieval | General-purpose embeddings are trained on prose, not Python | not started |
| Enable BM25 + RRF fusion | retrieval | Exact identifier matches are what dense retrieval blurs away | not started |
| Cross-encoder rerank of top-50 | retrieval | Reordering the top candidates is literally what NDCG@10 measures | not started |
| Strip comments from snippets | corpus | Less noise — but comments may be the only NL bridge to the query. Test both directions | not started |
| Truncate snippets head vs. tail | corpus | APPS solutions exceed the context window; which half carries the signal? Note TASK A: only 23.5% of snippets overflow, vs **61.9% of queries** — the query side is the bigger loss and has no knob yet. | not started |
| Compress query: drop Input/Output format sections and worked examples, keep the narrative problem core | query | Queries truncate **61.9%** at 254 tokens and the example blocks are formatting noise eating the budget. (Supersedes "strip boilerplate framing", which assumed a shared prefix that does not exist — 2.0%, not every query.) | not started |
| Index `meta_information.starter_code` alongside the snippet body | corpus | Every corpus row has a populated `meta_information` dict with a `starter_code` field; we index `text` only, so it is signal the retriever never sees. Cheap to test. | not started |
| Query category routing | query | Different `QUERY_CATEGORIES` may want different top-k or fusion weights | not started |
| Tune RRF `k` | retrieval | 60 is the paper default, not a measured optimum for this corpus | not started |
| Split snake_case/camelCase in BM25 tokenizer | corpus | Lets "binary search" match `binary_search` | not started |
| Chunk long snippets with overlap | corpus | Avoids truncating away the relevant function | not started |
