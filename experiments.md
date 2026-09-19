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
| Strip boilerplate framing from queries | query | ~~"Write a Python function that…" is in every query~~ **Disproved** by TASK A: queries share NO common prefix, and that phrasing opens only 2.0% of them. Low value. | dropped |
| Query category routing | query | Different `QUERY_CATEGORIES` may want different top-k or fusion weights | not started |
| Tune RRF `k` | retrieval | 60 is the paper default, not a measured optimum for this corpus | not started |
| Split snake_case/camelCase in BM25 tokenizer | corpus | Lets "binary search" match `binary_search` | not started |
| Chunk long snippets with overlap | corpus | Avoids truncating away the relevant function | not started |
