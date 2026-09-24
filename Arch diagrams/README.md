# Architecture diagrams

## Current

**[`architecture-clean.drawio.xml`](architecture-clean.drawio.xml)** — the
shipped pipeline. This is the diagram to present.

One page, four lanes: **OFFLINE** (build the index once), **ONLINE** (answer a
query), **EVALUATION** (MTEB v2), and **VERSIONING · P1** (content-hash cache).

What it shows, and nothing else:

```
 query ──► instruction prefix + token budget ──► bi-encoder ──► FAISS ──► top-k
```

One dense stage. No BM25 index, no BM25 retrieval, no RRF fusion, no
cross-encoder reranker. The only optional component that is live is the
content-hash embedding cache (`ENABLE_EMBEDDING_CACHE = True`), which sits
beside the encode step.

## Archived

**`archive/PRISM_Theme1_diagrams.SUPERSEDED.drawio`** — superseded, kept for
history, **do not present it.**

All three of its pages (`1 · Simple`, `2 · Detailed`, `3 · DFD (Level 1)`) draw
BM25 retrieval, RRF fusion and cross-encoder reranking as shipped components,
and the DFD adds a `D2 BM25 index` datastore. None of that is in the live path:
every one of those stages is flag-gated off after measurement. See the ablation
table in [`../experiments.md`](../experiments.md) for what each one measured and
why it was gated.

It is archived rather than deleted because it is an accurate record of the
architecture that was *designed* before the gates ran.

## Two corrections made to the current diagram

Both were checked against the code before committing, and both were places the
diagram would otherwise have promised more than the repository delivers:

1. **The "Query preprocessing" box no longer claims the instruction prefix and
   token budget.** `ENABLE_QUERY_PREPROCESSING` is `False`, so
   `get_query_processor()` returns `NoOpQueryProcessor` and the query text
   passes through untouched. The instruction prefix
   (`config.QUERY_PROMPT_PREFIX`) is applied by `BaselineEncoder.encode(...,
   prompt_type="query")` in the **embed** step — `DenseRetriever.retrieve_batch`
   carries an explicit docstring saying it is *not* applied there too, because
   prefixing twice is silent and costs accuracy. The token budget is likewise
   enforced by the encoder's own `max_seq_length`. Both labels moved onto the
   "Embed query" box, and the preprocessing box is now marked as the
   passthrough it is.
2. **"Diff + content hash" → "Content hash per snippet".** There is no separate
   diff step; the `hashlib.sha256` key lookup in `src/versioning/cache.py` *is*
   the mechanism that decides what to re-embed.

The **VERSIONING · P1 lane was kept**, because that code genuinely exists and
runs: `DiskEmbeddingCache` with `ENABLE_EMBEDDING_CACHE = True`, entries
version-namespaced under `CACHE_DIR/<CACHE_VERSION>/`, atomic writes via
`os.replace`, wired into the shipped index build at `src/corpus/index.py:80`,
and measured end to end — 695.3 s cold, **0.7 s warm**, 19.6 s for a
100-snippet edit ([`../data/cache_rebuild_demo.json`](../data/cache_rebuild_demo.json)).
