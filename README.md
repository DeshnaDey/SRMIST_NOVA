# PRISM — Agentic Code Intelligence

**Samsung PRISM GenAI Hackathon 2026 · Theme 1**

Natural-language retrieval over code. Given a query in plain English and a
corpus of code snippets, rank the snippets by relevance.

- **Dataset:** [`CoIR-Retrieval/apps`](https://huggingface.co/datasets/CoIR-Retrieval/apps)
- **Evaluation:** MTEB **v2** `AppsRetrieval` task
- **Metrics:** NDCG@10 (headline) and MRR
- **Constraint:** CPU only. Nothing in this project may request a GPU.
- **Python:** 3.11
- **Release tag:** `PRISM_GENAI_HACKATHON_Y2026` _(not yet tagged — see [Submission](#submission))_

---

## Approach

A hybrid retrieve → fuse → rerank pipeline:

```
 query ──► query preprocessing ──┐
                                 ├──► dense retrieval (bi-encoder + FAISS) ──┐
 corpus ─► snippet preprocessing ┘                                           ├─► RRF fusion ──► cross-encoder rerank ──► top-k
                                 └──► BM25 retrieval (lexical) ──────────────┘
```

Why this shape:

- **Dense retrieval** catches semantic matches — "sort a list without built-ins"
  finds a bubble-sort implementation that shares no vocabulary with the query.
- **BM25** catches what dense retrieval blurs away — exact identifiers, API
  names, error strings. Code queries quote these constantly.
- **RRF fusion** merges the two by *rank*, not score. Cosine similarities live
  in `[-1, 1]`; BM25 scores are unbounded. Summing them directly hands the
  ranking to BM25 for no better reason than its numbers being bigger.
- **Cross-encoder reranking** reorders the survivors. It scores query and
  snippet jointly, which is far more accurate and far too slow to run over the
  whole corpus — so it only ever sees the fused top-N.

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

Expect `38 passed, 53 skipped, 2 deselected`. The skips are unimplemented stubs; the passes
include regression tests that pin the MTEB v2 API surface.

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
reason. If you enable `ENABLE_EMBEDDING_CACHE`, point `PRISM_DATA_DIR` somewhere
unsynced too; the embedding cache grows fast and is pure build output.

### Pinned versions

All versions are **pinned and verified** — resolved 2026-09-14 on macOS 15
(arm64) / CPython 3.11.15, and confirmed by a clean-room install from
`requirements.txt` alone.

| Package | Version |
|---|---|
| `torch` | 2.14.0 (CPU) |
| `mteb` | **2.20.11** (v2) |
| `sentence-transformers` | 6.0.1 |
| `faiss-cpu` | 1.15.0 |
| `rank-bm25` | 0.2.2 |
| `datasets` | 5.0.1 |
| `huggingface_hub` | 1.31.0 |
| `numpy` | 2.4.6 |
| `tqdm` | 4.70.1 |
| `pytest` | 9.1.1 |

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

Day-one baseline — a bare bi-encoder, no pipeline stages:

```bash
python scripts/run_eval.py --pipeline baseline
```

The full hybrid pipeline:

```bash
python scripts/run_eval.py --pipeline full
```

Fast smoke run while developing (**not a reportable score**):

```bash
python scripts/run_eval.py --pipeline baseline --limit 50
```

### The iteration loop

Experiment on **train**; keep **test** for confirming a winner, once.

```bash
python scripts/run_eval.py --pipeline baseline --split train            # full train
python scripts/run_eval.py --pipeline baseline --split train --limit 50 # fast probe
```

`--limit` samples deterministically from `config.RANDOM_SEED`, so two runs over
the same split and limit see the identical subset and a model-vs-model delta is
a real delta. It narrows the queries and their qrels together and leaves the
corpus at full size. `--limit` runs print `SMOKE RUN, not reportable` and never
earn an `experiments.md` row.

> **Train and test are not interchangeable.** Test is markedly harder: 89.0% of
> test queries exceed a 254-token window against 61.9% of train, and its gold
> snippets are longer too. Expect absolute scores to drop when you confirm on
> test — compare models to each other within a split, never across splits.

> **Never filter the corpus on the `partition` column.** It labels each snippet
> `train`/`test`, and every query's gold doc sits in its own partition, so
> filtering on it shrinks the candidate pool and manufactures a large fake gain.

Both write **`appsretrieval_results.json`** and print NDCG@10 / MRR. Copy those
into [`experiments.md`](experiments.md) with what you changed.

### Docker (stub)

```bash
docker build -t prism-retrieval .
docker run --rm -v "$PWD/results:/app/results" prism-retrieval
```

The `Dockerfile` is a **stub with open TODOs** — it has the right shape and the
CPU-only constraints, but has not been built or tested yet.

---

## Project structure

```
.
├── src/
│   ├── config.py           ← every tunable lives here. Start reading here.
│   ├── interfaces.py       ← the stage contracts. Read this SECOND.
│   ├── query/              ← stage 1: query cleaning + classification
│   ├── corpus/             ← stage 2: snippet normalization + index building
│   ├── retrieval/          ← stage 3: dense, BM25, fusion, reranking
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

### What the data actually looks like

Before tuning anything, read [`data/inspection_report.md`](data/inspection_report.md)
(regenerate with `python scripts/inspect_data.py`). The three findings that most
often get assumed wrong:

- Queries are **full competitive-programming problem statements** (~1,050 chars
  median), not short natural-language asks.
- **Truncation is a query-side problem.** 61.9% of queries overflow the current
  model's 254-token window, against 23.5% of snippets — roughly 2.6x. Budgets are
  denominated in tokens (`config.MAX_QUERY_TOKENS` / `MAX_SNIPPET_TOKENS`) and
  resolve from whichever checkpoint is active.
- There is **exactly one relevant document per query**, so NDCG@10 and MRR are
  monotonically related and will move together.

**`src/config.py`** — every model name, top-k, batch size and feature flag.
A result is reproducible from a git SHA because everything that shapes it is
here. Values marked `# PLACEHOLDER` are unvalidated guesses; replace them with
something measured and log the delta.

**`src/interfaces.py`** — the contracts all four workstreams build against.
Treat it as frozen unless the whole team agrees to a change; four people are
building against these signatures simultaneously.

---

## How the four workstreams fit together

Each stage is an abstract base class in `src/interfaces.py` with a no-op
passthrough implementation already in place, so **the pipeline is green from day
one** and each stage is a single-flag A/B test.

| # | Workstream | Owns | Implements | Enable with |
|---|-----------|------|-----------|-------------|
| 1 | **query** | `src/query/` | `QueryProcessor` | `ENABLE_QUERY_PREPROCESSING` |
| 2 | **corpus** | `src/corpus/` | `SnippetProcessor`, `IndexBuilder` | `ENABLE_SNIPPET_PREPROCESSING` |
| 3 | **retrieval** | `src/retrieval/` | `Retriever`, `FusionStrategy`, `Reranker` | `ENABLE_BM25`, `ENABLE_RERANK` |
| 4 | **eval** | `tests/`, `src/versioning/` | `EmbeddingCache`, seed tests | `ENABLE_EMBEDDING_CACHE` |

Stubs are marked `# TODO(owner):` with a docstring stating the expected input
and output. **`src/pipeline/prism_search.py` is already fully wired** — as each
stub is filled in, the full pipeline starts working with no changes to that
file. Nobody needs to edit it to ship their own stage, which matters when all
four of us would otherwise be editing it at once.

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

| File | Category | Owner |
|------|----------|-------|
| `test_a_query_preprocessing.py` | A — query cleaning | query |
| `test_b_snippet_preprocessing.py` | B — snippet normalization, id preservation | corpus |
| `test_c_indexing.py` | C — FAISS / BM25 construction | corpus |
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

- [ ] Pin every version in `requirements.txt`
- [ ] Record the baseline row in `experiments.md`
- [ ] Full pipeline run committed with `appsretrieval_results.json`
- [ ] `Dockerfile` actually built and tested
- [ ] Tag the release:

```bash
git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "Samsung PRISM GenAI Hackathon 2026 submission"
git push origin PRISM_GENAI_HACKATHON_Y2026
```
