# Architecture diagrams

## Current

**`architecture-clean.drawio.xml` — NOT YET ADDED.**

This directory has no diagram of the shipped pipeline. The replacement was
supplied separately and has not landed in the repository; add it here and delete
this paragraph.

The shipped pipeline it must show, and nothing else:

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
