> ## ⚠️ Status update — read before acting on this document
>
> Written on `shanavi-work` before the model sweep ran. Its **analysis is
> sound**, but two of its recommendations were overtaken by measurement:
>
> **1. `jina-embeddings-v2-base-code` cannot be used.** It does not load on the
> pinned stack. Its remote modeling code targets transformers 4.x and we
> resolve 5.17.0 (via `sentence-transformers==6.0.1`): it imports
> `find_pruneable_heads_and_indices`, removed in 5.x, and reads
> `config.is_decoder`, no longer defaulted — and `config_kwargs` cannot reach
> the custom `JinaBertConfig` to supply it. `einops` is genuinely required by
> it, as this document says, but it is not what blocks the load. Using it needs
> an isolated transformers 4.x environment.
>
> **2. The shipped model is `Snowflake/snowflake-arctic-embed-m`**, selected on
> a full-train sweep by recall@100 (0.6870 vs e5-base 0.6728, bge-base 0.6642,
> all-MiniLM 0.6560) and confirmed on test: NDCG@10 0.08222, recall@100
> 0.30677. It needs no remote code.
>
> Also note `trust_remote_code` is handled by a **per-checkpoint allowlist** in
> `src/pipeline/baseline.py`, not the global `DENSE_MODEL_TRUST_REMOTE_CODE`
> flag proposed here — so a later model swap cannot inherit the permission
> silently.
>
> Full reasoning: the decision log in [`experiments.md`](experiments.md) and
> [`guide.md`](guide.md). Everything else below still stands.

# Tech Stack — Finalized

Written against the scaffold already in `src/config.py`/`README.md` (MTEB v2, CPU-only,
Python 3.11, `CoIR-Retrieval/apps` via `AppsRetrieval`).

Every decision here is now live in `src/config.py` — that file is the single
source of truth; this document is the *why* behind it, for the deck and for
anyone reviewing the diff. Nothing here is implemented yet beyond config and
the two-line loader change it required (see "What actually changed" at the
bottom) — the retrieval logic itself (`src/retrieval/`, `src/corpus/index.py`)
is still the retrieval/corpus workstreams' stubs to fill in, per the
ownership table in `README.md`.

---

## 1. Embedding model (bi-encoder)

**`jinaai/jina-embeddings-v2-base-code`** — 161M params, trained on
CodeSearchNet-style text↔code pairs across 30 languages, 8,192-token context
via ALiBi.

Why this one over the day-one placeholder (`all-MiniLM-L6-v2`, general-prose,
384-dim, 512-token limit):

- It's a **code-retrieval checkpoint**, not a general-sentence one — it
  starts much closer to `AppsRetrieval` (NL problem statement → Python
  solution) than a model trained on prose similarity ever will.
- Its **8k context directly answers the dataset brief's own warning** — "APPS
  solutions can exceed typical 512-token limits… you'll hit silent
  truncation unless you handle it." Instead of building chunking to work
  around a 512-token model, we picked a model that mostly doesn't need it.
- Still **base-sized**, so it's CPU-feasible: encoding happens once over
  ~12.5k texts (8.77k corpus + 3.77k queries) and is cached
  (`ENABLE_EMBEDDING_CACHE`, item 7), so this is a one-time cost, not a
  per-eval cost.

Trade-off, stated plainly: it ships **custom modeling code** (ALiBi isn't a
stock HF architecture), so loading it needs `trust_remote_code=True`. If the
judging sandbox blocks that, the documented fallback is dropping
`DENSE_MODEL_NAME` back to `sentence-transformers/all-MiniLM-L6-v2` — the
model the existing baseline number (NDCG@10 = 0.0660) was measured against —
one constant, no code change beyond that.

## 2. Reranker (cross-encoder)

**`cross-encoder/ms-marco-MiniLM-L-6-v2`** (kept from the placeholder — now
finalized, not guessed).

There's no established, widely-used *code-specific* cross-encoder to reach
for, so this stays a general passage-relevance reranker; "does this text
answer this query" transfers reasonably to "does this snippet solve this
problem." What actually matters here is size: unlike the bi-encoder, the
reranker runs on the **per-query hot path** with nothing cached, so it's the
tightest CPU latency budget in the pipeline. 6-layer MiniLM (~22M params) is
the right size to start; `ms-marco-MiniLM-L-12-v2` is the documented next
rung up if there's latency budget left after measuring.

## 3. Sparse / keyword search

**`rank-bm25`** (already pinned), Okapi defaults (`k1=1.5, b=0.75`) as the
starting point — tune on the **train** split's 5k queries if time allows,
never on test.

The real lever isn't the BM25 math, it's the **tokenizer**
(`src.corpus.index.tokenize_code`, corpus workstream): split
`snake_case`/`camelCase` into subtokens *and* keep the compound form, so a
query saying "binary search" matches `binary_search`. Lowercase everything.
Don't strip underscores — they're exactly the identifier signal BM25 exists
to catch (this is also the seed-test case the dataset brief calls out:
"exact-identifier case… assert hybrid beats dense-only here").

## 4. Vector search

**FAISS `Flat`** (exact brute-force over normalized vectors, cosine via inner
product) — finalized, not a placeholder pending an upgrade.

At ~8.77k corpus docs, exact search is a sub-5ms matmul. An ANN index
(IVF/HNSW) would trade retrieval accuracy away for a speed-up nothing in this
project needs. Revisit only if `TOP_K_DENSE` search is independently measured
as a bottleneck — and re-measure NDCG afterward, because ANN is lossy.

## 5. MTEB version pinning

Already done and verified in this repo: **`mteb==2.20.11`** (v2 API —
`mteb.evaluate`, `AbsEncoder`, `SearchProtocol`), confirmed against a
clean-room install. Nothing to change here; flagging it as finalized rather
than silent. One addition: once the team's environment is fully settled
(after picking the embedding model above), run
`pip freeze > requirements.lock.txt` **in the actual environment you'll
submit from** and commit it — `requirements.txt` pins direct dependencies,
the lockfile pins the full transitive tree (~60 packages) for a byte-exact
rebuild. Do this last, after every other tech-stack item below is locked in,
so it isn't immediately stale.

## 6. Tokenizer / max length

- **`MAX_SEQ_LENGTH = 1024`** tokens for the bi-encoder — a deliberate
  midpoint, not the model's max. The checkpoint extrapolates to 8,192, but
  self-attention cost grows roughly with sequence length squared, and paying
  that at 8k tokens for the *whole corpus* isn't worth it when most APPS
  solutions are far shorter. 1,024 covers the long tail without the blowup.
  **Action item for whoever owns corpus stats**: measure what fraction of
  train-split snippets actually exceed 1,024 tokens before trusting this
  number in the deck.
- **`MAX_SNIPPET_CHARS` raised 4,000 → 20,000`** as a character-level safety
  net only, sized to roughly match the token budget above (code runs ~3-4
  chars/token). `ENABLE_CHUNKING` stays off by default — it's an escape
  hatch for outliers, not the default path, given the model choice above.
- **No asymmetric query/document prefix** — confirmed against the model
  card (unlike E5/BGE/GTE, this checkpoint is symmetric). Both
  `QUERY_PROMPT_PREFIX` / `DOCUMENT_PROMPT_PREFIX` stay empty. If the model
  ever changes to an asymmetric one, this is the first thing to fix — it
  fails silently (quietly worse retrieval), not loudly.

## 7. Versioning (P1)

Architecture already scaffolded correctly in `src/versioning/cache.py`:
content-addressed cache, key = SHA-256 of `(CACHE_VERSION, model_name, sorted
extra flags, text)`, one `.npy` file per key. That's the right shape — it's
still a `TODO(eval)` stub (disk I/O not implemented), which is that
workstream's job, not a tech-stack question.

Tech-stack decisions made here: **`CACHE_VERSION` bumped `v1 → v2`**, because
`DENSE_MODEL_NAME` changed. `make_cache_key` already folds the model name
into the hash, so the bump isn't strictly load-bearing — but it's the
documented, deliberate way of saying "nothing written before this point
should be trusted," which a silent model swap alone doesn't communicate to
someone who finds old `.npy` files in the cache directory later. Bump it
again any time preprocessing changes in a way the content hash itself can't
see.

## 8. AST (bonus / demo only)

**Python's stdlib `ast` module.** No new dependency, no tree-sitter — the
corpus is 100% Python (APPS solutions), so there's no multi-language case to
justify a general parser toolkit.

Scope, deliberately narrow: `ENABLE_AST_DEMO` (added to `config.py`, default
`False`) gates a **presentation-layer** feature only — e.g. showing a
retrieved snippet's parsed function signature + docstring instead of raw
text, or an AST-level diff for the versioning demo ("this function changed,
here's what changed structurally"). It must never be read from
`src/pipeline` or `src/retrieval` — this is demo narrative for "Agentic Code
Intelligence," not a scoring input, and it should stay that way so it can't
accidentally break the graded run.

## 9. Reproducibility

Already had the right pieces, one of them wasn't wired up:

- `RANDOM_SEED = 42` existed as a config constant but nothing ever called
  `random.seed()` / `numpy.random.seed()` / `torch.manual_seed()` with it — a
  documented-but-unused seed. **Fixed**: `scripts/run_eval.py` now seeds all
  three (best-effort — torch/numpy may not be installed yet for `--help`) at
  the top of `main()`.
- `requirements.txt` pinning + verification block: already solid, keep as
  is.
- `requirements.lock.txt`: see item 5 — generate it last, from the real
  submission environment.
- Docker: still an open TODO per the README (`Dockerfile` stub, not built or
  tested) — that's a submission-mechanics item (`H` in the seed-test list),
  not a tech-stack one, but it's the other half of "reproducible from a git
  SHA."

## 10. LLM query expansion (optional)

**Recommendation: skip it for the scored submission.** Expanding all 3.77k
test queries through any LLM — local or API — adds per-query latency and a
new failure mode (a hung or rate-limited call mid-evaluation) for an
unmeasured NDCG upside, on a CPU budget that's already spent on the
reranker.

If it's worth doing anyway for the demo narrative (this *is* an "Agentic Code
Intelligence" hackathon, after all): treat it as a **demo-only** feature, not
a scored one. Run it once, offline, on a handful of illustrative queries,
with a model that needs neither a GPU nor network access at judging time —
`google/flan-t5-small` (~80M params) is a reasonable pick — and route the
expansions through the same embedding cache (item 7) so it's a one-time cost,
never a per-eval-run one. Added as disabled config
(`ENABLE_QUERY_EXPANSION = False`, `QUERY_EXPANSION_MODEL`) so the hook
exists without anyone being tempted to wire it into the scored path by
accident.

---

## What actually changed (for the PR/commit)

All in `src/config.py` unless noted:

- `DENSE_MODEL_NAME`: `all-MiniLM-L6-v2` → `jinaai/jina-embeddings-v2-base-code`
- `DENSE_MODEL_TRUST_REMOTE_CODE`: new, `True`
- `MAX_SEQ_LENGTH`: `None` → `1024`
- `MAX_SNIPPET_CHARS`: `4_000` → `20_000`
- `CACHE_VERSION`: `"v1"` → `"v2"`
- `ENABLE_AST_DEMO`: new, `False`
- `ENABLE_QUERY_EXPANSION` / `QUERY_EXPANSION_MODEL`: new, `False` / `"google/flan-t5-small"`
- `FAISS_INDEX_FACTORY`, `RERANK_MODEL_NAME`, `RERANK_TOP_N`, `BM25_K1`/`BM25_B`,
  `QUERY_PROMPT_PREFIX`/`DOCUMENT_PROMPT_PREFIX`: unchanged values, comments
  upgraded from "PLACEHOLDER, someone's guess" to "finalized, here's why"
- `config.describe()`: extended so the new decisions show up in every
  `appsretrieval_results.json` — reproducibility only matters if the config
  that produced a score is actually recorded alongside it
- `requirements.txt`: added `einops==0.8.1` (required by the new dense
  model's custom modeling code)
- `src/pipeline/baseline.py`: `SentenceTransformer(...)` call now passes
  `trust_remote_code=config.DENSE_MODEL_TRUST_REMOTE_CODE` — without this the
  new model fails to load, full stop
- `src/corpus/index.py`, `src/retrieval/dense.py`: docstrings/TODOs updated
  so whoever implements `DenseIndexBuilder.build` and `DenseRetriever`
  doesn't lose an afternoon to the same `trust_remote_code` gap
- `scripts/run_eval.py`: added `_seed_everything()`, called from `main()`

Verified: `pytest -m "not slow"` still passes 32/32 (same as before this
change — these are config values, no test hardcodes a model name), and
`config.describe()` round-trips cleanly.

**Not done yet, and deliberately out of scope for this pass**: actually
running `python scripts/run_eval.py --pipeline baseline` with the new model
to get a fresh NDCG@10 number. That requires downloading real model weights
in the real CPU environment — do that next and log the result as a new
`experiments.md` row (*"Swap dense model: MiniLM → jina-code"*, one line,
one variable changed, comparable against the existing 0.0660 baseline).
