# PRISM — Agentic Code Intelligence

**Samsung PRISM GenAI Hackathon 2026 · Theme 1**

> **Naming.** *Samsung PRISM / PRISM GenAI* is the **programme**; **`SRMIST_NOVA`**
> is the **team** (CollegeName_TeamName), and it is the repository and release
> name. The `PRISM` in this project's module names refers to the programme's
> theme, not to the team. Do not rename the repository — `origin` is already
> [`DeshnaDey/SRMIST_NOVA`](https://github.com/DeshnaDey/SRMIST_NOVA).

Natural-language retrieval over code. Given a query in plain English and a
corpus of code snippets, rank the snippets by relevance.

- **Dataset:** [`CoIR-Retrieval/apps`](https://huggingface.co/datasets/CoIR-Retrieval/apps)
- **Evaluation:** MTEB **v2** `AppsRetrieval` task
- **Metrics:** NDCG@10 (headline) and MRR
- **Constraint:** CPU only. Nothing in this project may request a GPU.
- **Python:** 3.11
- **Team:** `SRMIST_NOVA` (CollegeName_TeamName)
- **Release tag:** `PRISM_GENAI_HACKATHON_Y2026` _(see [Submission](#submission))_

---

## Approach

**A single-stage dense retriever.** That is the whole shipped pipeline:

```
 query ──► instruction prefix + token budget ──► bi-encoder ──► FAISS ──► top-k
```

That is not where this started. Four more stages — BM25, RRF fusion,
cross-encoder reranking, and preprocessing on both the query and corpus sides —
were each put in front of a gate, and **every one of them failed its gate on
measurement.** They ship disabled rather than deleted, because a disabled stage
with a number attached is a result.

They were not all measured the same way, and the difference matters:

**Built, measured, then disabled — one stage.** `CrossEncoderReranker` is a real
implementation. It was run against the retrieval pool and it **loses**: NDCG@10
−0.0994 at pool 10 falling to −0.2578 at pool 100 (`ms-marco-MiniLM-L-4-v2`),
and −0.1598 → −0.2656 for `bge-reranker-base`. Both sit above a random
reordering and below the bi-encoder's own ordering at every depth — they carry
relevance signal, just weaker signal than arctic already has, so reranking
overwrites a better ranking with a worse one.

**Gated out on a pre-build diagnostic, with the measured bound — three stages.**
For these, a standalone diagnostic established the ceiling on what the stage
could ever buy *before* it was wired into the pipeline, and the ceiling did not
justify building it. The modules exist as stubs behind their flags:

- **BM25 + RRF fusion.** BM25 works in isolation (recall@100 **0.4224** on
  train, against dense's 0.6870). But **93.7% of its hits are queries dense
  already answered** — only 133 queries, 2.7%, are BM25-only. The union ceiling
  is 0.7136 against 0.6870, so **+0.027 is the most any fusion of these two
  lists could ever add**, and that is an oracle bound a real RRF would only
  partly reach. Fusion pays when two systems fail on *different* queries; these
  fail on the same ones.
- **Snippet preprocessing.** Three arms, none kept: `starter_code` +0.0016
  (p=0.50), chunking +0.0012 (p=0.45), signature/comment augmentation
  **−0.0088 (p=0.003)** — significantly harmful.
- **Query compression.** A causal probe cut queries that currently *fit*.
  Cutting 25% off the tail — a **stronger** cut than the 79.4% real truncated
  queries retain — moved recall@100 by **+0.002**. You have to delete half of
  every query to lose one point, so the trailing boilerplate this proposal
  targeted is worth approximately nothing to the encoder.

Full numbers, p-values and reasoning: [`experiments.md`](experiments.md).

**Where the headroom actually is.** The bottleneck is recall, and it is inside
the encoder. There is exactly one gold document per query, and it sits outside
the top 100 for **69.3%** of test queries. Nothing that reorders the top 10 can
move the headline number — a *flawless* reranker of a 100-deep pool tops out at
0.307 on test. The remaining lever is the encoder itself (fine-tuning,
hard-negative mining), not what we feed it.

---

## Results

**Full `test` split, 3,765 queries against the full 8,765-snippet corpus.** No
limit, no smoke run. This is the locked submission artifact
(`appsretrieval_results.json`), reproduced bit-for-bit by the container gate.

| Metric | **test** (full) |
|---|---:|
| **NDCG@10** | **0.08222** |
| MRR@10 | 0.06799 |
| recall@10 | 0.12855 |
| recall@100 | 0.30677 |
| Wall-clock | **624.1 s** (10.4 min), CPU only |

For reference, the same pipeline on **train** (5,000 queries) scores recall@100
**0.6870** / NDCG@10 0.45863. **Train and test are not comparable** — test is
materially harder (41.2% of its queries exceed the encoder's budget against
24.6% of train) and no number from one split may be quoted against the other.

Shipped model: **`Snowflake/snowflake-arctic-embed-m`**, selected on a full
*train* sweep by recall@100 over e5-base-v2, bge-base-en-v1.5 and all-MiniLM.

---

## Setup

```bash
git clone https://github.com/DeshnaDey/SRMIST_NOVA.git
cd SRMIST_NOVA
```

**1. Create and activate the virtual environment** (Python 3.11):

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

**2. Install PyTorch — CPU build, first and on its own:**

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

> Install torch *before* everything else. On Linux and in Docker this is
> required: the default PyPI wheels are CUDA builds (~2.5 GB) that are useless
> on CPU-only judging hardware. On macOS the PyPI wheel is already CPU-only, so
> the flag is a harmless no-op — one command for everyone, no per-OS footgun.

**3. Install the rest:**

```bash
pip install -r requirements.txt
```

> ⚠️ **MTEB v2 is required — not v1.** The pipeline is written against
> `mteb.evaluate`, `AbsEncoder` and `SearchProtocol`, none of which exist in v1.

**4. Verify the install:**

```bash
python -c "
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models import ModelMeta
assert mteb.get_task('AppsRetrieval')
assert hasattr(mteb, 'evaluate') and hasattr(mteb, 'SearchProtocol')
print('OK', mteb.__version__)"
```

```bash
pytest -m "not slow"
```

Expect **`62 passed, 41 skipped, 2 deselected`**. The passes include regression
tests that pin the MTEB v2 API surface and the traps in
[`experiments.md`](experiments.md). The 41 skips are the gated-out stages: their
tests are kept alongside the stubs as the record of what each stage was supposed
to do.

### macOS: keep the venv off iCloud Drive

If this repo lives under `~/Desktop` or `~/Documents` **and** iCloud "Desktop &
Documents" sync is on, `fileproviderd` will churn through the ~1.6 GB virtual
environment continuously. Measured on this project: `import mteb` never
completed in 7+ minutes, at 0% CPU, because every read was blocking on the
iCloud file provider.

Put the venv outside the synced tree and symlink it back:

```bash
python3.11 -m venv ~/.prism/venv
~/.prism/venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
~/.prism/venv/bin/python -m pip install -r requirements.txt
ln -s ~/.prism/venv .venv        # so `source .venv/bin/activate` still works
```

After the move the same import takes ~110 s cold and a few seconds warm.

The Hugging Face cache defaults to `~/.cache/huggingface`, which is already
outside the synced tree — leave `PRISM_HF_CACHE` unset unless you have a
reason. **`ENABLE_EMBEDDING_CACHE` is on by default** — point `PRISM_DATA_DIR`
somewhere unsynced too, since the embedding cache grows fast and is pure build
output. A 27 MB `np.load` from a synced `data/` failed outright here with
`TimeoutError: [Errno 60]`.

### Pinned versions

All versions are **pinned and verified** — resolved 2026-09-14 on macOS 15
(arm64) / CPython 3.11.15, and confirmed by a clean-room install from
`requirements.txt` alone.

Runtime dependencies live in `requirements.txt`; test- and diagnostic-only
ones live in `requirements-dev.txt` (which includes `requirements.txt`), so
the image stays honest about what the submission actually needs.
`pyproject.toml` does not carry a second list — it declares
`dynamic = ["dependencies"]` and reads `requirements.txt`.

| Package | Version | Where |
|---|---|---|
| `torch` | 2.14.0 (CPU) | runtime |
| `mteb` | **2.20.11** (v2) | runtime |
| `sentence-transformers` | 6.0.1 | runtime |
| `transformers` | 5.17.0 | runtime — **explicitly pinned**, see below |
| `faiss-cpu` | 1.15.0 | runtime |
| `datasets` | 5.0.1 | runtime |
| `huggingface_hub` | 1.31.0 | runtime |
| `numpy` | 2.4.6 | runtime |
| `tqdm` | 4.70.1 | runtime |
| `pytest` | 9.1.1 | dev |
| `scipy` | 1.16.2 | dev — `scripts/bm25_diagnostic.py` |
| `rank-bm25` | 0.2.2 | dev — referenced in docstrings only, nothing imports it |

**`transformers` is pinned explicitly, not inherited.** `src/config.py` imports
`AutoTokenizer` directly to resolve a checkpoint's context window. Until
recently it resolved only transitively through `sentence-transformers`; that
worked, but a future `sentence-transformers` release could move or drop the
pin underneath us and the symptom would be a wrong context window — quietly
worse retrieval, not an `ImportError`. It is now declared at 5.17.0, the
version the stack already resolved and every number in `experiments.md` was
measured with. A fresh `pip install -r requirements.txt` gets it directly.

**Import paths that are easy to get wrong** (all confirmed against 2.20.11 —
each of these was wrong in the first draft and caught only by checking the
installed package):

- `AbsEncoder` → `mteb.models.abs_encoder.AbsEncoder` — *not* `mteb.AbsEncoder`
  or `mteb.models.AbsEncoder`
- `ModelMeta` → `mteb.models.ModelMeta` — *not* `mteb.ModelMeta`; it has 17
  required fields, most nullable but all mandatory to pass
- `SearchProtocol.index/search` both take a keyword-only **`num_proc`**
- `TaskResult` → `mteb.results.task_result.TaskResult`, re-exported as
  `mteb.TaskResult` — *not* `mteb.load_results.task_results`, which is the
  v1-era path and fails silently if you wrap the import in a try/except

**MTEB rounds every score to 6 decimal places when it writes a result to
disk** (`TaskResult._round_scores(..., 6)`). Our `appsretrieval_results.json`
serialises the in-memory object *before* that happens, so it carries full
float precision and will not byte-match MTEB's own cached copy. That is a
superset, not a mismatch — compare the two at 6dp, which is what
`scripts/validate_results.py` does.

Don't bump anything without re-running the verification block above and
`pytest -m "not slow"`.

---

## Running the evaluation

**The submission run.** `--pipeline` defaults to `full` and `--split` to
`test`, so the bare command reproduces the submitted result and writes
`appsretrieval_results.json`:

```bash
python scripts/run_eval.py
```

### Anything that is not the submission needs `--output`

`run_eval.py` **refuses** to write `appsretrieval_results.json` from any run
that is not `--pipeline full --split test` unlimited, and refuses at
argument-parse time before loading a model. This is deliberate: a `--limit`
smoke run produces a real-looking results file from a handful of queries, and
a `--pipeline baseline` run writes the bare encoder — and on the current stack
the baseline's NDCG is *identical* to the full pipeline's, because every
optional stage is measured-and-off. The corrupted file could not be spotted by
reading it. So redirect non-submission runs:

```bash
# bare bi-encoder, for comparison
python scripts/run_eval.py --pipeline baseline --output /tmp/baseline.json

# fast smoke run (NOT a reportable score)
python scripts/run_eval.py --limit 50 --output /tmp/smoke.json
```

### The iteration loop

Experiment on **train**; keep **test** for confirming a winner, once.

```bash
python scripts/run_eval.py --split train --output /tmp/train.json
python scripts/run_eval.py --split train --limit 50 --output /tmp/probe.json
```

`--limit` samples deterministically from `config.RANDOM_SEED`, so two runs over
the same split and limit see the identical subset and a model-vs-model delta is
a real delta. It narrows the queries and their qrels together and leaves the
corpus at full size. `--limit` runs print `SMOKE RUN, not reportable` and never
earn an `experiments.md` row.

> **Train and test are not interchangeable.** Test is markedly harder: at the
> shipped model's **510**-token content budget, **41.2%** of test queries
> overflow against **24.6%** of train, and its gold snippets are longer too.
> (Older 254-token figures in this repo are `all-MiniLM-L6-v2`-only and are
> superseded — budgets now resolve per-model from the active checkpoint.)
> Expect absolute scores to drop when you confirm on test — compare models to
> each other within a split, never across splits.

> **Never filter the corpus on the `partition` column.** It labels each snippet
> `train`/`test`, and every query's gold doc sits in its own partition, so
> filtering on it shrinks the candidate pool and manufactures a large fake gain.

> **`meta_information.starter_code` — MEASURED, and it is not a lever.** Every
> corpus row carries a `meta_information` dict, but read the fields separately:
> `url` is non-empty on 100% of rows while **`starter_code` is non-empty on only
> 38.8%** (3,401/8,765). It is also largely redundant — **96.1%** of the
> `def`/`class` names it holds already appear in the document's own text.
> Indexing it alongside the body moved recall@100 **+0.0016 (p=0.50)** and was
> *negative* on the queries it touched. See the Decision log in
> [`experiments.md`](experiments.md). Unlike `partition` it was a legitimate
> thing to try; it simply does not pay.

Both write **`appsretrieval_results.json`** and print NDCG@10 / MRR. Copy those
into [`experiments.md`](experiments.md) with what you changed.

### Dataset verification and evaluation contract

This project uses the official MTEB AppsRetrieval task backed by the Hugging
Face dataset `CoIR-Retrieval/apps`.

The expected dataset shape is approximately:

- ~8.77k corpus entries (Python solution snippets)
- ~8.77k natural-language queries
- ~5k training examples for development / tuning
- ~3.77k test examples for official evaluation

The MTEB task loads the corpus, queries, and qrels through the task metadata,
then evaluates on the task's official test split. The train split may be used
for local development and hyperparameter tuning, but the evaluation result must
remain measured on the held-out test split rather than fitting on the test qrels.

The dataset is APPS-style: natural-language problem statements become the query
texts, and the corpus contains Python solution snippets for those problems.
This is the exact retrieval setting for the challenge: a query asks for a coding
solution or algorithmic pattern, and the model ranks likely matching snippets.

The official metric reporting for AppsRetrieval is:

- NDCG@10
- MRR@10

`--pipeline baseline` is the bare MTEB-compatible bi-encoder, kept as the
control condition. `--pipeline full` is the submission path: it runs
`PrismSearch` through MTEB's `SearchProtocol`, so the scores come from the
actual pipeline rather than from an encoder MTEB wraps for us. **On the current
configuration the two produce identical numbers to 5 decimal places**, and that
is the correctness proof rather than a coincidence: with every optional stage
gated out, the full pipeline must reduce to the same dense retrieval, so any
deviation would have been a wiring bug.

### Long-code handling

The shipped encoder's window is **512 tokens (510 for content**, after
`[CLS]`/`[SEP]`), and budgets resolve per-model at runtime —
`config.MAX_QUERY_TOKENS` and `config.MAX_SNIPPET_TOKENS` derive from whichever
checkpoint `DENSE_MODEL_NAME` names, so a model swap cannot silently leave 97%
of a larger window unused.

`config.ENCODER_WINDOW_CAP = 2048` is a hard ceiling on top of that, and it
exists for one document. The corpus maximum is **60,599 tokens** (`d1660`)
against a p99 of 1,023. Under a 254-token model that outlier was truncated away
for free; hand it to a model advertising 8k–32k and, because attention is
quadratic, those few documents can dominate the entire corpus encode. The cap is
inert for the current 512-token checkpoint.

The dataset is small enough for CPU execution with NumPy and FAISS on a local or
lab machine. The dominant runtime constraint is the corpus embedding pass, not
the dataset size — and reranking is not a constraint at all here, because it is
measured and off.

### Docker — the reproducible path

These are the exact commands CI runs on every push, so they are tested rather
than aspirational (see
[`.github/workflows/container-gate.yml`](.github/workflows/container-gate.yml)):

```bash
docker build -t prism-retrieval .
docker run --rm -v "$PWD/results:/app/results" prism-retrieval
```

**Output:** `results/appsretrieval_results.json` on the host — the image's
`CMD` is the full pipeline on the full test split
(`--pipeline full --split test`), writing into the mounted volume.

**It needs no network.** The model and dataset are baked in at build time by
`scripts/prefetch_assets.py`, and the image then sets `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. You can prove it by
removing the network entirely, which is what the gate does:

```bash
docker run --rm --network none -v "$PWD/results:/app/results" prism-retrieval
```

The dataset id and revision are read from the MTEB task itself rather than
hardcoded, so the bake cannot drift out of step with what `task.load_data()`
requests at runtime.

**Verified — on a clean machine, in CI, not on a developer laptop.** Be precise
about this, because it is a graded claim:

- The image has **never been built locally.** The dev machine has no Docker
  daemon, which is also why the `Dockerfile` avoids comments inside its
  backslash-continued `ENV` block — it could not be build-tested here.
- It **has** been built and run end-to-end on a clean `ubuntu-latest` runner
  from a `python:3.11-slim` base, by
  [`.github/workflows/container-gate.yml`](.github/workflows/container-gate.yml).
  The gate builds with `--pull`, runs the offline asset check, runs
  `pytest -m "not slow"` inside the container, executes the image's own `CMD`
  on the **full test split with `--network none`**, and diffs the result
  against the committed `appsretrieval_results.json`, failing on any mismatch.
- **Latest green run:** [`35660647349`](https://github.com/DeshnaDey/SRMIST_NOVA/actions/runs/35660647349),
  on commit `c49fb50`, 2026-09-21, 1h25m52s — every step passed including the
  artifact diff. It reproduces NDCG@10 **0.08222** / MRR@10 **0.06799** /
  recall@100 **0.30677** bit-for-bit.

---

## P1 — rebuild cost tracks the change, not the corpus

The requirement: a new version of the codebase must not pay to re-embed a corpus
that mostly did not change. Edit 100 of 8,765 snippets and the rebuild should
cost about 100 encodes, not 8,765.

**Measured** (`python scripts/cache_rebuild_demo.py --split test --changed 100`,
raw numbers in [`data/cache_rebuild_demo.json`](data/cache_rebuild_demo.json)):

| Arm | Wall-clock | Cache hits | Encodes |
|---|---:|---:|---:|
| **cold** — empty cache for this version | **695.3 s** | 0 | 8,754 |
| **warm** — nothing changed | **0.7 s** | 8,754 | **0** |
| **changed** — 100 snippets edited | **19.6 s** | 8,654 | **100** |

**A warm rebuild is ~1,000× faster than cold** (0.7 s against 11.6 min), and a
100-snippet edit costs 19.6 s. All four bars are asserted by the script, not
eyeballed: warm re-embeds nothing, changed re-embeds *exactly* the 100 edited
snippets, every other distinct snippet stays cached, and rebuild time tracks
change size (2.8% of cold for a 1.1% change — the residual is fixed overhead,
model load and reading 8,654 entries off disk, not re-encoding).

Implementation notes that are load-bearing: one file per entry rather than one
archive (a single `.npz` must be rewritten in full on every change, which would
fail this bar from the write side); entries sharded by the first two hex
characters of the key, because a flat directory of ~100k files is painfully slow
to list on macOS; the key covers everything that changes a vector (text,
checkpoint, `normalize`, `window_cap`, `max_seq_length`, prompt prefix, side);
and writes are atomic via `np.save` to a temp file plus `os.replace`, with a
corrupt or truncated entry treated as a **miss**, never an exception.

> The corpus contains **11 exact duplicate snippets** (8,765 rows, 8,754
> distinct texts). They share one cache entry, so count hits in unique keys, not
> snippets, or assertions come out 11 short.

---

## Project structure

```
.
├── src/
│   ├── config.py           ← every tunable lives here. Start reading here.
│   ├── interfaces.py       ← the stage contracts. Read this SECOND.
│   ├── query/              ← stage 1: preprocessing — GATED OUT (passthrough)
│   ├── corpus/             ← stage 2: preprocessing GATED OUT; dense index SHIPS
│   ├── retrieval/          ← stage 3: dense SHIPS; BM25/fusion/rerank gated out
│   ├── pipeline/           ← MTEB v2 adapters (baseline encoder + full search)
│   └── versioning/         ← content-hashed embedding cache
├── tests/                  ← pytest, categories A–H
├── scripts/run_eval.py     ← the only way to produce a score
├── experiments.md          ← the experiment log. Every run gets a row.
├── Arch diagrams/          ← architecture diagrams (.drawio)
├── requirements.txt
└── Dockerfile
```

### The two files that matter most

> **New to this repo? Read [`guide.md`](guide.md) first.** It carries what is
> already settled, the silent failure modes, what has been ruled out with
> numbers, and what to work on next — most of which was expensive to learn.
> [`TECH_STACK.md`](TECH_STACK.md) is now only the pinned-version rationale —
> what each pin is for and why it does not get bumped.

### What the data actually looks like

Before tuning anything, read [`data/inspection_report.md`](data/inspection_report.md)
(regenerate with `python scripts/inspect_data.py`). The three findings that most
often get assumed wrong:

- Queries are **full competitive-programming problem statements** (~1,050 chars
  median), not short natural-language asks.
- **Truncation is a query-side problem.** At `all-MiniLM`'s 254-token window,
  61.9% of queries overflowed against 23.5% of snippets — roughly 2.6×, and that
  asymmetry is what set the priority between stages. At the shipped model's
  **510**-token budget both shrink: **24.6%** of train queries (41.2% of test)
  and **6.70%** of snippets. Budgets are denominated in tokens
  (`config.MAX_QUERY_TOKENS` / `MAX_SNIPPET_TOKENS`) and resolve from whichever
  checkpoint is active — never reuse a 254-token figure for the shipped model.
- There is **exactly one relevant document per query**, so NDCG@10 and MRR are
  monotonically related and will move together.

**`src/config.py`** — every model name, top-k, batch size and feature flag.
A result is reproducible from a git SHA because everything that shapes it is
here. Two labels appear on values: `# UNVALIDATED DEFAULT` marks a value on the
**shipped** path that works and was never swept (sweeping one is a real
experiment — log the delta); `# PLACEHOLDER` marks a value belonging to a
**gated-out** stage, which nothing reads on the shipped path.

**`src/interfaces.py`** — the contracts all four workstreams build against.
Treat it as frozen unless the whole team agrees to a change; four people are
building against these signatures simultaneously.

---

## How the four workstreams fit together

Each stage is an abstract base class in `src/interfaces.py` with a no-op
passthrough already in place, so the pipeline runs green whether or not a stage
is filled in, and each stage is a single-flag A/B.

**Every `ENABLE_` flag as shipped**, read from `src/config.py`:

| # | Workstream | Owns | Flag | Shipped value | State |
|---|---|---|---|---|---|
| 1 | **query** | `src/query/` | `ENABLE_QUERY_PREPROCESSING` | **`False`** | Gated out pre-build (`truncation_probe.py`); stub |
| 2 | **corpus** | `src/corpus/` | `ENABLE_SNIPPET_PREPROCESSING` | **`False`** | Gated out pre-build (`corpus_variants.py`); stub |
| 2 | **corpus** | `src/corpus/` | `ENABLE_CHUNKING` | **`False`** | The `chunk` arm of the same gate |
| 3 | **retrieval** | `src/retrieval/` | `ENABLE_BM25` | **`False`** | Gated out pre-build (`bm25_diagnostic.py`); stub |
| 3 | **retrieval** | `src/retrieval/` | `ENABLE_RERANK` | **`False`** | **Implemented**, measured, disabled (`rerank_eval.py`) |
| 4 | **eval** | `src/versioning/` | `ENABLE_EMBEDDING_CACHE` | **`True`** | **The one optional stage that ships ON** |

Dense retrieval has no flag — it is the pipeline.

Stubs are marked `# TODO(owner):` with a docstring stating the expected input
and output. **`src/pipeline/prism_search.py` is already fully wired** — if a
stub is ever filled in, the full pipeline picks it up with no changes to that
file.

### Rules that hold at every stage

1. **IDs are sacred.** A snippet's `id` must survive preprocessing, indexing,
   retrieval, fusion and reranking byte-for-byte. MTEB joins our output to its
   qrels by that string — mangle it and the score silently drops to zero with
   no error anywhere.
2. **Stages are pure.** Same input and config, same output. The embedding cache
   is the one sanctioned exception.
3. **Never raise on bad input.** Degrade instead. A malformed query should not
   kill a two-hour evaluation at query 9,000.
4. **Higher is better**, and ranked lists are always sorted descending.

---

## Testing

```bash
pytest                    # everything
pytest -m "not slow"      # skip anything needing a model download (default for CI)
pytest tests/test_f_fusion.py
```

**`pytest -m "not slow"` → `62 passed, 41 skipped, 2 deselected`.** The skipped
tests belong to the gated-out stages and are kept deliberately: together with
the stubs they are the record of what each stage was specified to do. A skipped
suite next to a measured gate is evidence; deleting it would leave only an
absence.

| File | Category | Owner |
|------|----------|-------|
| `test_a_query_preprocessing.py` | A — query cleaning | query |
| `test_b_snippet_preprocessing.py` | B — snippet normalization, id preservation | corpus |
| `test_c_indexing.py` | C — dense index (runs) / BM25 (skipped) | corpus |
| `test_d_dense_retrieval.py` | D — bi-encoder search | retrieval |
| `test_e_bm25_retrieval.py` | E — lexical search | retrieval |
| `test_f_fusion.py` | F — RRF / weighted fusion | retrieval |
| `test_g_reranking.py` | G — cross-encoder reordering | retrieval |
| `test_h_pipeline_e2e.py` | H — end-to-end + MTEB contract | eval |

Placeholder tests are marked `@pytest.mark.skip` with a `TODO(owner)` reason —
drop the marker as you implement. The contract tests at the top of each file
already pass against the no-op implementations and should keep passing against
the real ones.

---

## Experiment log

[`experiments.md`](experiments.md) tracks every run: date, who, what changed,
NDCG@10, MRR, commit. **Change one thing at a time**, and log negative results
too — they stop the next person retrying the same idea at 3am.

---

## Submission

- [x] **Every version pinned** in `requirements.txt`, resolved and verified
      2026-09-14 — see [`TECH_STACK.md`](TECH_STACK.md)
- [x] **Baseline row recorded** in [`experiments.md`](experiments.md), along
      with every later run and every gate that said no
- [x] **Full pipeline run committed** as `appsretrieval_results.json`
      (`pipeline: full`, split `test`, unlimited), round-tripped through
      `scripts/validate_results.py` with 6dp agreement against MTEB's own cache
- [x] **`Dockerfile` built and run on a clean machine** — in CI on
      `ubuntu-latest`, offline with `--network none`, reproducing the locked
      artifact bit-for-bit (run
      [`35660647349`](https://github.com/DeshnaDey/SRMIST_NOVA/actions/runs/35660647349)).
      **Never built on the dev machine**, which has no Docker daemon
- [ ] Tag the release:

```bash
git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "SRMIST_NOVA — Samsung PRISM GenAI Hackathon 2026 submission"
git push origin PRISM_GENAI_HACKATHON_Y2026
```
