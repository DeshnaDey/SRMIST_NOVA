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
| 2026-09-15 | dataset/eval | **Baseline**: bare bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`), no preprocessing, no BM25, no rerank; MTEB AppsRetrieval test split | 0.0660 | 0.0558 | 7.71s | `HEAD` | Verified baseline on the current repo state. The train split is kept for development/tuning only; the reported score is from the official test split. |
| | | | | | | | |

## Backlog — ideas not yet measured

Move a row into the table above once it has a number next to it.

| Idea | Owner | Rationale | Status |
|------|-------|-----------|--------|
| Swap in a code-specific bi-encoder | retrieval | General-purpose embeddings are trained on prose, not Python | not started |
| Enable BM25 + RRF fusion | retrieval | Exact identifier matches are what dense retrieval blurs away | not started |
| Cross-encoder rerank of top-50 | retrieval | Reordering the top candidates is literally what NDCG@10 measures | not started |
| Strip comments from snippets | corpus | Less noise — but comments may be the only NL bridge to the query. Test both directions | not started |
| Truncate snippets head vs. tail | corpus | APPS solutions exceed the context window; which half carries the signal? | not started |
| Strip boilerplate framing from queries | query | "Write a Python function that…" is in every query and discriminates nothing | not started |
| Query category routing | query | Different `QUERY_CATEGORIES` may want different top-k or fusion weights | not started |
| Tune RRF `k` | retrieval | 60 is the paper default, not a measured optimum for this corpus | not started |
| Split snake_case/camelCase in BM25 tokenizer | corpus | Lets "binary search" match `binary_search` | not started |
| Chunk long snippets with overlap | corpus | Avoids truncating away the relevant function | not started |
