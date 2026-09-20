# Working guide

What is settled, what will silently break, and what to do next. Read this
before picking up a task — most of it was expensive to learn.

Companion docs: [`README.md`](README.md) (setup + architecture),
[`experiments.md`](experiments.md) (every run + the decision log),
[`data/inspection_report.md`](data/inspection_report.md) (the raw data facts).

---

## Where we are

| | |
|---|---|
| Model | `Snowflake/snowflake-arctic-embed-m` (512 ctx, query-only instruction prefix) |
| Test | NDCG@10 **0.08222** · MRR **0.06799** · **recall@100 0.30677** |
| Train | recall@100 **0.6870** · recall@10 0.5458 · NDCG@10 0.4586 |
| Pipeline | Bare bi-encoder. Every stage is still a no-op passthrough. |
| Cost | ~26 min full test run · ~4 min with all-MiniLM |

**The bottleneck is recall, not ranking.** One gold doc per query, and it is
outside the top 100 for ~69% of test queries. A cross-encoder reranker cannot
recover a document retrieval never surfaced, so work that only reorders the top
10 cannot move the headline number. Raise recall@100 first.

---

## 1. Environment — the thing that will eat your day

**Do not run the venv from `~/Desktop` or `~/Documents` if iCloud "Desktop &
Documents" sync is on.** Symptom: `import mteb` hangs indefinitely at **0% CPU**
while `fileproviderd` pegs a full core. It is not a hang in Python — every file
read is blocking on the iCloud file provider churning a 1.6 GB venv.

```bash
python3.11 -m venv ~/.prism/venv
~/.prism/venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
~/.prism/venv/bin/python -m pip install -r requirements.txt
ln -s ~/.prism/venv .venv        # documented commands keep working
```

Measured: `import mteb` went from *never completing in 7+ minutes* to ~110 s
cold, seconds warm. The HF cache (`~/.cache/huggingface`) is already outside the
synced tree — leave `PRISM_HF_CACHE` unset.

**Versions are pinned and verified. Do not bump anything.** `torch 2.14.0`
(CPU, `cuda_available=False`), `mteb 2.20.11`, `sentence-transformers 6.0.1`,
`transformers 5.17.0`, `numpy 2.4.6`. If an import fails, report it — do not
fix it by upgrading.

This box is 8 GB RAM / 8 cores and runs close to full. Run one model at a time.

---

## 2. Commands

```bash
python scripts/run_eval.py --pipeline baseline                    # full test
python scripts/run_eval.py --pipeline baseline --split train      # iterate here
python scripts/run_eval.py --pipeline baseline --split train --limit 50
python scripts/validate_results.py                                # before trusting a number
python scripts/inspect_data.py                                    # regenerate data facts
python scripts/benchmark_models.py --list
python scripts/benchmark_models.py --model arctic-m --split train
python scripts/bm25_diagnostic.py --split train
pytest -m "not slow"                                              # 38 passed, 53 skipped
```

`--limit` samples deterministically from `RANDOM_SEED`, narrows queries **and**
qrels together, leaves the corpus full, and never earns an `experiments.md` row.

---

## 3. Facts — do not re-derive these

From [`data/inspection_report.md`](data/inspection_report.md):

- Corpus **8,765 snippets, identical across splits**. Train 5,000 queries,
  test 3,765. Only the query set and qrels differ.
- **Exactly one relevant doc per query**, both splits. So NDCG@10 and MRR are
  monotonically related — they move together, and reporting both adds little.
- Queries are **full competitive-programming problem statements** (~1,050 chars
  median, prose + maths), not short natural-language asks. Snippets are Python
  solutions.
- IDs are `q<N>` / `d<N>` strings, and **every qrel is `q<N> → d<N>`**. Do not
  exploit it (ids are opaque by contract), but it is a free sanity check: if a
  run scores ~0, compare emitted ids against this pattern.
- **`title` is empty for every row** — 0/8,765 corpus, 0/5,000 queries. The
  title+body join is marked dead at its three sites. It is kept because it is
  what keeps the baseline and SearchProtocol paths encoding byte-identical
  strings. Remove all three together or none.
- Truncation at arctic's 510-token budget: **train 24.6%, test 41.2%**.
- **No common query prefix.** `"Write a function"` opens 2.0% of queries, not
  "every query". The longest common prefix across all 5,000 train queries is
  the empty string.

---

## 4. Silent failure modes

Everything here completes without an error and quietly produces a worse number.
These are the expensive ones.

**Instruction prefixes.** e5, bge and arctic each require a specific prefix;
getting it wrong is a silent scorer. Keep `QUERY_PROMPT_PREFIX` /
`DOCUMENT_PROMPT_PREFIX` in step with `DENSE_MODEL_NAME` — the schemes and their
model-card provenance are documented next to the constants in `config.py`.
Current: arctic uses a query-only instruction, document side empty.

**MTEB's result cache.** `mteb.evaluate` defaults to
`overwrite_strategy="only-missing"` with a cache at `~/.cache/mteb`, which hands
back a *previous* run's scores instead of re-evaluating. Left alone it would
compare models against cached copies of each other. `run_eval.py` forces
`"always"` — keep it that way.

**The `partition` column.** It labels each snippet `train`/`test`, and every
query's gold doc sits in its own partition. Filtering on it shrinks the pool
from 8,765 to 3,765 and manufactures a large fake gain. Never filter on it.

**Train and test are not comparable.** Test is materially harder — 41.2% of its
queries truncate vs 24.6% of train, and its gold snippets are longer. Train
recall@100 0.687 vs test 0.307 is mostly real difficulty, not a bug. Compare
models to each other *within* a split; never across.

**BM25 tokenizer mismatch.** Corpus and query sides must use the *same*
tokenizer function. A mismatch is a recall killer no test catches.

**IDs are sacred.** A snippet id must survive every stage byte-for-byte. MTEB
joins to qrels on that string; mangle it and the score drops to zero silently.

**Character caps are the wrong unit.** `MAX_SNIPPET_CHARS` / `MAX_QUERY_CHARS`
are gone; reading either raises an `AttributeError` naming the replacement. Use
`config.MAX_SNIPPET_TOKENS` / `MAX_QUERY_TOKENS`, which resolve per-model.

---

## 5. Already ruled out — with numbers

Do not redo these. Full reasoning is in the `experiments.md` decision log.

| Idea | Verdict | Evidence |
|---|---|---|
| **BM25 + RRF fusion** | **Dropped** | BM25-only recall@100 0.4224 is respectable, but it recovers only **133 queries (2.7%)** arctic missed. Union ceiling 0.7136 vs 0.6870 — **+0.027 is the oracle bound**, and real RRF gives some back. 93.7% of BM25's hits are already dense hits. |
| **jina-v2-base-code** | **Unusable** | Three stacked transformers 4.x/5.x breakages. Its remote code needs `find_pruneable_heads_and_indices` (removed in 5.x) and `config.is_decoder` (no longer defaulted); `config_kwargs` cannot reach the custom config. Needs an isolated transformers 4.x env, not more shims. |
| **Qodo-Embed-1-1.5B** | **Impractical** | 6.17 GB of fp32 weights and ~4–5 h per run on an 8 GB box. |
| **Longer context as the lever** | **Weak** | e5 cut query truncation 61.5% → 24.4% for **+1.7 points of recall@100**, recall@10 flat, at 7.6× the encode cost. |
| **Stripping boilerplate query prefixes** | **Dropped** | Premise was false: no shared prefix exists (2.0%, not "every query"). |
| **Published CoIR numbers as a guide** | **Unreliable here** | bge beat e5 on our data (NDCG@10 0.456 vs 0.437) while CoIR reports e5 11.52 vs bge 4.05. Likely CoIR used bge v1.0, not v1.5. |

---

## 6. What to do next

**First, and cheaply: the causal probe.** Queries that fit the 510-token window
score recall@100 **0.7493**; those just over it score **0.4827**. That 27-point
gap is *correlational* — longer problems may simply be harder. Artificially
truncate queries that currently **fit** and measure whether recall actually
falls. Corpus embeddings are already cached, so this is minutes.

This one probe decides the next two weeks:

- **If truncation is causal** → build query compression: drop the Input/Output
  format sections and worked examples, keep the narrative problem core. Test
  truncates 41.2% of queries, so the graded split benefits most. Note the
  honest ceiling: re-weighting train's per-bucket recalls to test's length
  distribution explains only **~4.8 of the 38-point train→test drop**, so this
  is worth points, not a transformation.
- **If it is not causal** → the headroom is inside the encoder, not in what we
  feed it. Then look at fine-tuning or hard-negative mining on the train split,
  which is the only remaining lever big enough to matter.

**Also unmeasured and cheap:** `meta_information.starter_code` is populated on
every corpus row and we index `text` only — real content the retriever never
sees.

**Deprioritise anything that only reorders the top 10** (cross-encoder rerank,
RRF `k` tuning, fusion weights) until recall@100 moves. With one gold doc per
query and 69% of them outside the top 100 on test, reordering is capped.

---

## 7. Discipline

- Work on local `main`, one commit per task, single author, no co-author
  trailer. Nothing is pushed without a human saying so.
- **Experiment on train. Test is for confirming a winner, once.** Never inspect
  test examples to design a heuristic.
- Every run seeded; `overwrite_strategy="always"`; round-trip the payload
  through `scripts/validate_results.py` before trusting any number.
- Every non-smoke run gets an `experiments.md` row with its commit SHA.
  **Negative results get a row too** — a gate that says "no" is a result, and
  the decision log exists so nobody relitigates it.
- Change one thing at a time.
- `src/interfaces.py` is frozen. Four workstreams build against those
  signatures.
- CPU only. Nothing may request a GPU.
