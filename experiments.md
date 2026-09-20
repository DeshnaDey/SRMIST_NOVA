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

### 2026-09-20 — Query compression: GATED OUT on a causal probe

Run `python scripts/truncation_probe.py --split train`; raw numbers in
`data/truncation_probe.json`. Full train split, the 3,770 queries that FIT
arctic's 510-token budget, top-100, corpus embeddings reused from the BM25
diagnostic.

**What was being tested.** The BM25 decision log above showed fitting queries
at recall@100 0.7493 against just-truncated ones at 0.4827, and flagged that
the 27-point gap is *correlational*: the two buckets hold different queries,
and an 800-token problem statement may simply be a harder problem than a
400-token one. So: take the queries that currently fit — whose 0.7493 is
measured with nothing cut — and cut them anyway. Same queries, same gold docs,
same corpus, same model; the only free variable is how much of the query the
encoder sees, so any drop is caused by truncation.

Queries are cut to a fraction of their OWN length, because that is the shape
of the real intervention (a 700-token query meeting a 510-token budget keeps
73% of itself). Cuts are made on character offsets, not by decoding token ids,
so every arm is a genuine prefix of the raw string and the 100% arm is the
original text byte-for-byte.

| keep | median tok kept | recall@100 | Δ vs control | lost | gained |
|---:|---:|---:|---:|---:|---:|
| **100%** (control) | 252 | **0.74934** | — | — | — |
| 75% | 189 | **0.75146** | **+0.0021** | 49 | 57 |
| 50% | 126 | 0.73899 | −0.0104 | 128 | 89 |
| 25% | 63 | 0.66950 | −0.0798 | 402 | 101 |
| 10% | 26 | 0.51247 | −0.2369 | 982 | 89 |

**The dose that matters is 79%, and at that dose nothing happens.** Genuinely
truncated queries in the 510–1020 bucket keep a median **79.4%** of themselves.
The 75% arm is a *stronger* cut than reality and recall does not fall — it
moves +0.002, losing 49 queries and gaining 57, which is noise. You have to
delete **half** of every query to lose a single point. Truncation at the rate
this dataset actually inflicts it is not costing us the 27 points.

**What is costing them is visible in the control arm.** Among queries that all
fit entirely, with nothing truncated anywhere, recall@100 still collapses with
length:

| original length | n | recall@100, nothing truncated |
|---|---:|---:|
| 0–127 tok | 653 | 0.8943 |
| 127–255 tok | 1,261 | 0.8303 |
| 255–382 tok | 1,104 | 0.7101 |
| 382–510 tok | 752 | **0.5452** |

A 35-point spread with zero truncation involved. The untruncated 382–510
bucket scores 0.545 — already most of the way to the truncated 510–1020
bucket's 0.483. Length predicts failure just as hard when nothing is being
cut, which is what "longer statements are harder problems" looks like.
Fitting those four untruncated points (−0.093 recall per +100 tokens) and
extrapolating to the truncated bucket's median length of 642 tokens predicts
**0.391** against an observed **0.4827**: truncated queries do *better* than
difficulty alone would predict. That extrapolation runs 130 tokens past its
data and is indicative only — but it cannot be read as truncation costing
points.

**Why this kills the compression proposal specifically.** The backlog item was
"drop the Input/Output format sections and worked examples, keep the narrative
problem core" — and those blocks sit at the END of a statement. Cutting the
tail is exactly what the 75% and 50% arms do, and it bought +0.002 and −0.010.
The marginal retrieval value of a query's later tokens is approximately zero,
so letting back in the 21% that a 642-token query currently loses to the budget
recovers content the encoder demonstrably does not use. Compression's whole
theory of action is "feed it more of the real content"; the probe says the
content at that margin is worth nothing.

**Honest limit of the probe.** It cuts prefixes. A real compressor would also
delete from the middle and could keep numeric constants and algorithm terms
that a blind cut discards, so it is not logically identical to the proposal.
But the proposal's own target — trailing boilerplate — is precisely what these
arms removed, with no gain, so the mechanism it relies on is the one that was
measured and found empty.

**Confirmed on the graded split, where truncation is worst.** Run
`python scripts/truncation_probe.py --split test`; raw numbers in
`data/truncation_probe_test.json`. Test truncates **41.2%** of queries
(1,551/3,765) against train's 24.6%, so this is the split where compression
would have to pay if it paid anywhere. It does not. Over the 2,214 test
queries that fit:

| keep | median tok kept | recall@100 | Δ vs control | lost | gained |
|---:|---:|---:|---:|---:|---:|
| **100%** (control) | 363 | **0.33062** | — | — | — |
| 75% | 273 | **0.33243** | **+0.0018** | 46 | 50 |
| 50% | 182 | 0.29720 | −0.0334 | 144 | 70 |
| 25% | 91 | 0.22900 | −0.1016 | 296 | 71 |
| 10% | 37 | 0.14408 | −0.1865 | 467 | 54 |

The 75% arm replicates train's null almost exactly (+0.0018 against +0.0021),
and test's genuinely truncated queries keep a median 80.6% — again a weaker
cut than the arm that showed nothing. The control's length decline reappears
too, on queries with nothing truncated: 0.575 (0–127 tok), 0.423, 0.339,
**0.276** (382–510 tok). Same story, harder split.

Note the harness check is empty for this run: the cached query vectors are
train's 5,000 and test has 3,765, so it was skipped rather than run against a
mismatched file. Two independent checks stand in. The corpus row order was
verified byte-identical between the two splits before trusting the shared
corpus cache, and the control arm's 0.33062 over the fitting subset sits just
above the logged full-test 0.30677, exactly as it must when the truncated
queries it excludes score lower.

**Where the headroom is instead: inside the encoder.** Not in what we feed it.
Fine-tuning or hard-negative mining on the train split is the remaining lever
big enough to matter. Note the control arm's own ceiling — even the shortest,
wholly-untruncated queries top out at 0.894, and long untruncated ones sit at
0.545.

**Harness check.** Before any arm ran, the probe scored the cached query
vectors and reproduced the logged numbers exactly: recall@100 **0.6870** over
all 5,000 train queries and **0.74934** over the fitting subset, against
0.6870 / 0.7493 in the rows above. The re-encoded 100% control then landed on
0.74934 as well, confirming the character-offset path adds no artifact of its
own. Not round-tripped through `validate_results.py`: that validates an MTEB
submission payload, and this diagnostic produces none — same as
`bm25_diagnostic.py`.

## Backlog — ideas not yet measured

Move a row into the table above once it has a number next to it.

| Idea | Owner | Rationale | Status |
|------|-------|-----------|--------|
| Swap in a code-specific bi-encoder | retrieval | General-purpose embeddings are trained on prose, not Python | not started |
| **Fine-tune / hard-negative mining on train** | retrieval | **Now the top lever.** The truncation probe ruled out the input side: the encoder does not use a query's later tokens, so no amount of reshaping the input helps. What is left is the encoder itself. 5,000 train pairs with one gold doc each is a usable training set. | not started |
| ~~Enable BM25 + RRF fusion~~ | retrieval | **GATED OUT — measured, see Decision log below.** BM25-only recall@100 is a respectable 0.4224, but it recovers only **133 queries (2.7%)** that arctic missed. Union ceiling 0.7136 vs arctic 0.6870: **+0.027 is the absolute best any fusion could reach**, before RRF gives some back. | dropped |
| Cross-encoder rerank of top-50 | retrieval | Reordering the top candidates is literally what NDCG@10 measures | not started |
| Strip comments from snippets | corpus | Less noise — but comments may be the only NL bridge to the query. Test both directions | not started |
| Truncate snippets head vs. tail | corpus | APPS solutions exceed the context window; which half carries the signal? Note TASK A: only 23.5% of snippets overflow, vs **61.9% of queries** — the query side is the bigger loss and has no knob yet. | not started |
| ~~Compress query: drop Input/Output format sections and worked examples~~ | query | **GATED OUT — measured, see Decision log.** Cutting 25% off the tail of a fitting query (a *stronger* cut than the 79.4% real truncated queries retain) moves recall@100 **+0.002**; cutting 50% costs 0.010. The trailing boilerplate this proposal targets is exactly what those arms removed, and removing it bought nothing. | dropped |
| Index `meta_information.starter_code` alongside the snippet body | corpus | Every corpus row has a populated `meta_information` dict with a `starter_code` field; we index `text` only, so it is signal the retriever never sees. Cheap to test. | not started |
| Query category routing | query | Different `QUERY_CATEGORIES` may want different top-k or fusion weights | not started |
| Tune RRF `k` | retrieval | 60 is the paper default, not a measured optimum for this corpus | not started |
| Split snake_case/camelCase in BM25 tokenizer | corpus | Lets "binary search" match `binary_search` | not started |
| ~~Causal probe: truncate a fitting query, measure the recall drop~~ | query | **DONE — see Decision log, `data/truncation_probe.json`.** Verdict: truncation is NOT causal at the dose this dataset inflicts. The 27-point gap is length-as-difficulty — untruncated queries alone span 0.894 (0–127 tok) to 0.545 (382–510 tok). | done |
| Chunk long snippets with overlap | corpus | Avoids truncating away the relevant function | not started |
