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
| Pipeline | **Full `PrismSearch` through MTEB's SearchProtocol.** Dense retrieval is real (`DenseIndexBuilder` + `DenseRetriever`); query/snippet preprocessing, BM25, fusion and rerank are all measured-and-off. |
| Submission | `appsretrieval_results.json` — `pipeline: full`, split `test`, validated. Regenerate: `python scripts/run_eval.py --pipeline full --split test`. |
| Cost | ~26 min full test run · ~4 min with all-MiniLM |

**The bottleneck is recall, and it is in the encoder.** One gold doc per query, and it is
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
python scripts/truncation_probe.py --split train          # causal truncation probe
python scripts/truncation_probe.py --split test --out data/truncation_probe_test.json
python scripts/corpus_variants.py --split train              # corpus-side levers
python scripts/rerank_eval.py --split train --limit 300 --pool 100   # rerank + oracle
python scripts/cache_rebuild_demo.py --split test --changed 100     # P1 rebuild proof
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
- The corpus encode is ~11.6 min cold. Document vectors are content-hash
  cached (`src/versioning/cache.py`), so a warm rebuild is **0.7 s** and a
  100-snippet edit costs **19.6 s** — measured, `data/cache_rebuild_demo.json`.
  Verify with `python scripts/cache_rebuild_demo.py --split test --changed 100`.
- **The corpus has 11 exact duplicate snippets** (8,765 rows, 8,754 distinct
  texts). They share one cache entry. Count cache hits in unique keys, not
  snippets, or your assertions are off by 11.
- **Keep heavy caches off the iCloud-synced Desktop.** A 27 MB `np.load` from
  `data/` failed with `TimeoutError: [Errno 60]` mid-session. Use
  `PRISM_DATA_DIR=~/.prism/data`; the config already supports it.
- Truncation at arctic's 510-token budget: queries **train 24.6%, test 41.2%**;
  snippets **6.70%** (587/8,765). The **23.5%** snippet figure in the inspection
  report is at all-MiniLM's **254**-token window — do not reuse it for arctic.
- `meta_information.starter_code` is non-empty on **38.8%** of rows, not all of
  them; the `meta_information` *dict* is on all 8,765 and its `url` is 100%.
- Snippets carry almost no natural language: **20.7%** have a `#` comment,
  **5.8%** a triple-quote, **32.6%** have neither a comment nor a `def`/`class`
  line. Any plan to "extract docstrings and comments" dies here.
- Whitespace normalisation is an **exact no-op**: the wordpiece tokenizer already
  discards indentation, so it yields byte-identical token ids.
- **No common query prefix.** `"Write a function"` opens 2.0% of queries, not
  "every query". The longest common prefix across all 5,000 train queries is
  the empty string.

---

## 3b. Two crashes that will eat your day

**faiss + torch SEGFAULTS this build (exit 139).** Importing faiss and then
running a torch forward pass kills the interpreter with no Python traceback —
the process just vanishes, and it vanishes *after* the corpus index is built,
so you lose the whole evaluation. faiss-cpu and torch each ship their own
OpenMP runtime; this is the classic duplicate-libomp crash on macOS.
Measured: default → SIGSEGV; `faiss.omp_set_num_threads(1)` → still SIGSEGV
(too late, the runtime is already up); `OMP_NUM_THREADS=1` in the env → works,
but pins torch to one thread and the encoder is the expensive half of every
run. So `DenseIndexBuilder` uses an exact numpy index when
`FAISS_INDEX_FACTORY == "Flat"` and never imports faiss. "Flat" is exact
brute-force inner product, so numpy is not an approximation of it — it is the
same computation. Any other factory still goes through faiss and will need
`OMP_NUM_THREADS=1`.

**`PrismSearch.mteb_model_meta` must not be None.** MTEB names its result
directory from it, so leaving it None raises `TypeError: unsupported operand
type(s) for /: 'PosixPath' and 'NoneType'` inside mteb's result cache — again
*after* the full evaluation has run. The bare-encoder path never hit this
because `BaselineEncoder` builds its own `ModelMeta`.

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
| **Query compression** | **Dropped** | Causal probe, **train and test**: cutting 25% off a fitting query's tail — a *stronger* cut than the ~80% real truncated queries keep — moves recall@100 **+0.002** (train) and **+0.0018** (test). You must delete half a query to lose a point. The 27-point fits-vs-truncated gap is length-as-difficulty, not truncation: untruncated queries alone run 0.894 (0–127 tok) down to 0.545 (382–510 tok). |
| **Cross-encoder reranking** | **Dropped — actively harmful** | MiniLM-L-4 −0.153 NDCG@10 at pool 25, −0.258 at pool 100; bge-reranker-base −0.266 at pool 25. Both above random, both below dense. Oracle NDCG@10 = recall@pool caps it at 0.307 on test regardless. |
| **`cross-encoder/ms-marco-MiniLM-L-6-v2`** | **Unusable** | Returns **NaN for every pair** on the pinned stack (finite fp32 weights, NaN from encoder layer 0, both sdpa and eager). Siblings L-4/L-12/TinyBERT-L-2 and bge are finite, so it is checkpoint-specific. NaN sorts as a no-op, so it fakes a perfect "reranking changed nothing" null — guarded in both `rerank_eval.py` and `CrossEncoderReranker`. |
| **Corpus preprocessing (all three levers)** | **Dropped** | `starter_code` +0.0016 (p=0.50), chunking +0.0012 (p=0.45), signature/comment header **−0.0088 (p=0.003)**. Chunking does work on its target — **+0.0924** on the 249 queries whose gold doc truncates — but only 4.98% of queries qualify, and max-over-chunks costs other queries by re-weighting toward long documents. |
| **Stripping boilerplate query prefixes** | **Dropped** | Premise was false: no shared prefix exists (2.0%, not "every query"). |
| **Published CoIR numbers as a guide** | **Unreliable here** | bge beat e5 on our data (NDCG@10 0.456 vs 0.437) while CoIR reports e5 11.52 vs bge 4.05. Likely CoIR used bge v1.0, not v1.5. |

---

## 6. What to do next

**The causal probe is done, and it closed the query-side lever.** Queries that
fit the 510-token window score recall@100 0.7493; those just over it score
0.4827. That 27-point gap is *not* truncation. Artificially truncating the
3,770 queries that fit shows recall is flat at the dose reality inflicts —
real truncated queries keep a median 79.4% of themselves, and a *harder* 75%
cut moves recall **+0.002**. Half a query has to go before one point does.
**Confirmed on test**, where 41.2% of queries truncate against train's 24.6%:
the 75% arm moves **+0.0018** there (control 0.33062 → 0.33243). The split
where compression would have to pay is the split that says it does not.
Full numbers in `experiments.md`'s Decision log, `data/truncation_probe.json`
and `data/truncation_probe_test.json`; rerun with
`python scripts/truncation_probe.py --split train` (arms are cached, so a
rerun is seconds).

What the gap actually is: length-as-difficulty. Among queries that all fit
entirely, with nothing cut anywhere, recall@100 runs **0.894** (0–127 tok),
0.830, 0.710, **0.545** (382–510 tok). The untruncated 382–510 bucket is
already near the truncated bucket's 0.483. Longer competitive-programming
statements are harder problems, and that is most of what the stratification
was showing.

**So the headroom is inside the encoder, not in what we feed it.** The next
lever worth two weeks is fine-tuning or hard-negative mining on the train
split — 5,000 pairs, one gold doc each. Reshaping the input is spent: the
encoder demonstrably does not use a query's later tokens, so neither
compression nor a longer context can pay (consistent with e5's +1.7 points
for cutting truncation 61.5% → 24.4%).

**The corpus side is now measured too, and none of it pays.** Appending
`meta_information.starter_code` (+0.0016, p=0.50), chunking over-budget
snippets (+0.0012, p=0.45) and prepending signature/comment headers (−0.0088,
p=0.003) all failed on train; nothing was kept and
`ENABLE_SNIPPET_PREPROCESSING` stays `False`. Numbers in the Decision log and
`data/corpus_variants.json`; rerun with `python scripts/corpus_variants.py
--split train` (document vectors are content-hash cached, so a rerun is
seconds).

**Reranking is measured and dropped — it made things worse.** Cross-encoder
rerank of the dense pool *loses* NDCG@10 at every depth (MiniLM-L-4 −0.099 at
pool 10 down to −0.258 at pool 100; bge-reranker-base −0.160 and −0.266). Both
beat a random reordering but lose to arctic's own ordering, so they overwrite
a better ranking with a weaker signal. Also note the arithmetic cap: with one
gold doc per query, **oracle NDCG@10 = recall@pool**, so even a perfect
reranker of the top 100 tops out at **0.307** on test. Deprioritise the rest
of the reordering family (RRF `k`, fusion weights) for the same reason.

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
