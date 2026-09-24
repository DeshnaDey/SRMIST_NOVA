# Tech Stack — why these versions are pinned

Scope of this document: **the pins in `requirements.txt` and the reasons behind
them.** Nothing else.

It used to be a design document arguing for a stack that was never shipped. Its
"what actually changed" section listed config values that are not in the code —
a different dense model, a `DENSE_MODEL_TRUST_REMOTE_CODE` flag that does not
exist, `MAX_SEQ_LENGTH = 1024`, `MAX_SNIPPET_CHARS = 20_000`,
`CACHE_VERSION = "v2"`, an `einops` dependency — and a test count from a suite
that has since roughly doubled. Every one of those claims was false against the
code. They are deleted rather than corrected, because a document that has to be
checked against `src/config.py` before it can be trusted is worse than no
document.

**For what the system does, read [`README.md`](README.md). For why each stage
is on or off, read [`experiments.md`](experiments.md). For the live value of any
setting, read `src/config.py` — it is the single source of truth, and this file
deliberately restates none of it.**

---

## The rule

**Versions are pinned, verified, and do not get bumped.** Resolved and tested
2026-09-14 on macOS 15 (arm64), CPython 3.11.15. Every number in
`experiments.md` was measured on this exact stack. If an import fails, report
it — do not fix it by upgrading.

`requirements.txt` carries the authoritative pins and a verification block to
run after installing. `requirements-dev.txt` adds what only the tests and
diagnostics need, so the runtime image stays honest about its real dependencies.

---

## Runtime pins

| Package | Pin | Why this one |
|---|---|---|
| `torch` | 2.14.0 | Tensor backend for sentence-transformers. **CPU wheel only** — install first, on its own, from the PyTorch CPU index. On Linux and in the Docker image the default PyPI wheels are CUDA builds (~2.5 GB) that CPU-only judging hardware cannot use. On macOS the PyPI wheel is already CPU-only, so the flag is a harmless no-op there and one command works for everyone |
| `mteb` | 2.20.11 | The benchmark harness; provides `AppsRetrieval` plus NDCG/MRR. **v2 is required, not v1.** The pipeline is written against the v2 surface: `mteb.evaluate(...)` (v1 used `MTEB(tasks).run(model)`), `AbsEncoder` (v2 `encode()` takes a DataLoader of batch dicts, not a `list[str]`), and `SearchProtocol` (v2-only, so our own pipeline rather than a bare encoder produces the scores) |
| `sentence-transformers` | 6.0.1 | Loads the bi-encoder and the cross-encoder checkpoints |
| `transformers` | 5.17.0 | **Declared directly, not inherited.** `src/config.py` imports `AutoTokenizer` to resolve a checkpoint's context window. Left transitive, a future sentence-transformers release could move the pin underneath us and the failure would surface as a wrong context window — quietly worse retrieval, not an `ImportError` |
| `faiss-cpu` | 1.15.0 | Vector index over snippet embeddings. CPU build only |
| `datasets` | 5.0.1 | Pulls the `CoIR-Retrieval/apps` corpus, queries and qrels |
| `huggingface_hub` | 1.31.0 | Model/dataset download and local cache management |
| `numpy` | 2.4.6 | Embedding matrices and score arithmetic |
| `tqdm` | 4.70.1 | Progress bars for the long encode passes |

## Dev-only pins

| Package | Pin | Why it is not a runtime dependency |
|---|---|---|
| `pytest` | 9.1.1 | A test dependency. The image installs both files, so the repro gate still runs inside the container |
| `scipy` | 1.16.2 | Imported by `scripts/bm25_diagnostic.py` for the sparse term-document matrix. It was missing from requirements entirely — the diagnostic worked locally only because something else pulled it in transitively, and a clean container could not have run it |
| `rank-bm25` | 0.2.2 | **Nothing imports it.** The BM25 diagnostic implements Okapi directly over a scipy sparse matrix because `rank_bm25` measured at ~0.6 s/query here. Kept in dev only so that comparison can be re-run |

---

## Two version facts that cost real time

**`jinaai/jina-embeddings-v2-base-code` does not load on this stack.** It was
the primary model-selection candidate — 161M params, 8192 tokens via ALiBi,
trained on code — and on paper it should have removed the query truncation
problem outright. It cannot be loaded, for three stacked reasons, each found
behind the last:

1. Its `config.json` pins `attn_implementation="torch"`, which transformers 5.x
   rejects. This one is fixable, and the retry in
   `src/pipeline/baseline.py::_load_sentence_transformer` is what fixes it.
2. It needs `trust_remote_code=True`. Without it transformers silently loads a
   stock BERT, because `config.json` says `model_type: "bert"` — you get a model
   with no ALiBi that cannot address its advertised context, and no error
   anywhere.
3. Its remote modeling code imports `find_pruneable_heads_and_indices` (removed
   in 5.x) and reads `config.is_decoder` (no longer defaulted). `config_kwargs`
   cannot reach the custom `JinaBertConfig` to supply it.

We stopped there. Each shim revealed the next breakage, and the failure being
courted is the worst one available to this project: a model that loads and
quietly produces wrong embeddings. Downgrading transformers is not on the table.
The honest route, if anyone wants this model, is an isolated transformers 4.x
environment — not more shims. **`_TRUST_REMOTE_CODE` in
`src/pipeline/baseline.py` is deliberately empty**; the comment block around it
is the record of why, and is not dead code to tidy away.

**`trust_remote_code` is a per-checkpoint allowlist, not a global flag.**
Granting it runs third-party Python from the Hub on this machine, so it is a
per-checkpoint human decision that a later model swap cannot inherit silently.

---

## What is deliberately *not* pinned

Only direct dependencies are pinned. For a byte-exact lock of the whole
transitive tree (~60 packages), run `pip freeze > requirements.lock.txt` from
the real submission environment.

There is **no `PINNED_VERSIONS` unit test.** Pin protection comes from the
container gate (`.github/workflows/container-gate.yml`), which installs from
`requirements.txt` on a clean `python:3.11-slim` and runs the full pipeline
offline: an unresolvable or incompatible pin fails the build, and a pin that
changes behaviour fails the artifact diff. What that does *not* catch is a pin
that is wrong but installable and behaviourally identical. That gap is known and
deliberately left — see the submission-seal section of
[`experiments.md`](experiments.md).
