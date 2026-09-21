# Cleanup & final-push gap report (Phase 0)

Read-only diagnosis across STEPS 6–11. **Nothing was changed to produce this
report.** Gaps are ranked by what they cost if shipped as-is.

Kept stack, as configured today:

| | |
|---|---|
| Encoder | `Snowflake/snowflake-arctic-embed-m`, query-only instruction prefix |
| Index | exact inner-product (`FAISS_INDEX_FACTORY = "Flat"`, numpy backend) |
| Query preprocessing | **off** (STEP 6 gated out) |
| Snippet preprocessing / chunking | **off** (STEP 7 gated out) |
| BM25 + fusion | **off** (STEP 1 gated out) |
| Rerank | **off** (STEP 9 gated out) |
| Embedding cache | **on**, content-hash keyed, versioned |

So the shipped pipeline is dense retrieval only, and every optional stage is
off *because it was measured and lost*.

---

## P0 — would ship a wrong or unreproducible artifact

### P0-1. A bare run regenerates the WRONG artifact
`scripts/run_eval.py --pipeline` defaults to **`baseline`** (`run_eval.py:194`)
and the Dockerfile CMD is **`--pipeline baseline`** (`Dockerfile:68`). Either
one overwrites the locked `appsretrieval_results.json` with a payload stamped
`"pipeline": "baseline"` — the bare encoder, not the submitted system.

Today the *number* happens to coincide (0.08222) because the kept stack is
dense-only, which is exactly what makes this dangerous: it looks correct, so
nobody checks, and the moment any stage is enabled the artifact silently
reverts to the weaker path. The Dockerfile also omits `--split test`.

**Silent failure prevented:** submitting the baseline while believing it is
the full pipeline.

### P0-2. The container still needs the network at grade time
Both prefetch steps are commented-out TODOs (`Dockerfile:57-63`): the model
bake-in and the dataset bake-in. `mteb.evaluate` will therefore hit the Hub
during the graded run. Phase 3 requires proving an offline run, and this
cannot pass as written. The prefetch must pull the **same dataset id and
revision** that `task.load_data()` resolves (`CoIR-Retrieval/apps`, revision
`f22508f96b7a36c2415181ed8bb76f76e04ae2d5` per the validated payload), or the
cache will be a near-miss and the Hub is still contacted.

### P0-3. The results file lands outside the mounted volume
The documented invocation mounts `-v "$PWD/results:/app/results"`
(`Dockerfile:7`), but `config.RESULTS_JSON` is `PROJECT_ROOT /
"appsretrieval_results.json"` → `/app/appsretrieval_results.json`. The
artifact is written *inside* the container and discarded on exit.

### P0-4. `scipy` is imported but not declared
`scripts/bm25_diagnostic.py` imports `scipy.sparse`; `scipy` appears nowhere
in `requirements.txt`. A clean container cannot run the BM25 diagnostic. It
works locally only because something else pulled scipy in transitively.

---

## P1 — no correctness net on the stages we actually ship

### P1-5. `tests/test_correctness.py` does not exist
There is no Layer-1 file at all. The kept stages have no fast offline
assertion against their most likely silent failure:

| kept stage | most likely silent failure | assertion exists? |
|---|---|---|
| query encoding | instruction prefix missing, or applied twice | **no** |
| document encoding | query prefix wrongly applied to documents | **no** |
| id handling | corpus id mangled between index and results | **no** |
| exact index | padding position emitted as a real id | **no** |
| `retrieve_batch` | batching changes results vs `retrieve` | **no** (skipped) |
| embedding cache | stale vector served after a config change | yes (`test_h`) |
| title+body join | the three sites drift apart | **no** |

Every one of these fails *quietly* — the run completes and the number is just
worse, which is the failure mode this project has been bitten by repeatedly.

### P1-6. Tests for now-implemented code are still skipped as TODO
`tests/test_d_dense_retrieval.py` skips `retrieve`, `retrieve_batch` and the
prompt-prefix test with `TODO(retrieval): implement…`. All three are
implemented as of STEP 10. `tests/test_c_indexing.py` skips 10 of 11 tests
though `DenseIndexBuilder.build` now exists. The suite reports **41 passed,
52 skipped** and the skips no longer reflect reality.

---

## P2 — hygiene and consistency

### P2-7. `rank-bm25` is declared but never imported
`requirements.txt:44` pins `rank-bm25==0.2.2`. No module imports it — the
BM25 diagnostic implements Okapi directly over a scipy sparse matrix
precisely *because* rank_bm25 was too slow (documented in
`bm25_diagnostic.py:233`). It is at most a diagnostic/dev dependency, and
BM25 is a dropped stage.

### P2-8. Five title+body joins, only three carry the contract marker
The "remove all three together or none" contract names three marked sites:
`src/pipeline/prism_search.py:292`, `src/pipeline/mteb_compat.py:139`,
`scripts/inspect_data.py:103`. But two further copies of the same join now
exist **without** the marker: `scripts/bm25_diagnostic.py:187` and
`scripts/corpus_variants.py:93`. The contract is no longer self-describing —
someone honouring "all three" would miss two.
**Per the brief the markers stay; this is reported, not changed.**

### P2-9. Dockerfile calls itself a STUB and cannot run the tests
Header says `STATUS: STUB` (`Dockerfile:4`). It copies `src/`, `scripts/`,
`pyproject.toml` — not `tests/` — while installing `pytest`, so the repro gate
cannot run inside the image.

---

## FLAGGED — brief says one thing, repo says another (not fixed)

1. **No entrypoint script exists.** There is no `entrypoint.sh` and no
   `nproc` / `--cpus` / `cpuset` mention anywhere in the repo. The briefed
   "entrypoint comment claims nproc tracks `--cpus=N`" has no counterpart
   here. *(The underlying fact is still true and worth honouring if an
   entrypoint is ever added: `nproc` reflects `--cpuset-cpus`, not the CFS
   quota set by `--cpus=N`.)*
2. **`_FakeQueryRows` does not exist.** No such test double is defined.
   `prepare_task` narrows queries with `.filter(...)` (`run_eval.py:117`), not
   `.select`, so the briefed filter/select mismatch cannot occur as described.
3. **No payload test calls `validate_results.py`.** The CLI does expose
   `--results` (`validate_results.py:98`), but nothing in `tests/` invokes it,
   so there is no call site to agree or disagree with.
4. **Pre-existing co-author trailer on `1c765a2`.** Confirmed present
   (`Co-Authored-By: Claude Opus 5`). Flagged for the human; history is **not**
   rewritten.
5. **Repo name unresolved.** `origin` is
   `github.com/DeshnaDey/SRMIST_NOVA`; the briefs reference Samsung-PRISM and
   step 14 wants `CollegeName_TeamName`. **Phase 4 stop condition — a team
   call, not a silent pick.**

---

## Verified sound (no action)

- **Gate decisions are all recorded with the number that justified them.**
  STEP 6 query compression (75% retention arm **+0.0021** train / **+0.0018**
  test, against a real truncated-query retention of ~80%); STEP 7 three corpus
  levers (`starter` +0.0016 p=0.50, `chunk` +0.0012 p=0.45, `augment` −0.0088
  p=0.003); STEP 9 rerank (−0.153 at pool 25, −0.258 at pool 100, both models
  above random and below dense). Each has a Decision-log section and a JSON
  artifact.
- **Train/test boundary intact.** Every keep/drop decision was taken on train
  or a train dev subset. The only test-split touches are the STEP 6 *null
  confirmation* (the decision was already made on train) and the STEP 10
  submission run. No stage was selected on a test number.
- **BM25 artifacts are internally consistent.** `data/bm25_diagnostic.json`
  carries recall@100 `0.4224`, `bm25_only` `133`, `union_ceiling` `0.7136`,
  and `experiments.md` quotes the same three. No mixed pre/post sets.
- **STEP 10 artifact provenance is correct.** `appsretrieval_results.json`
  records `pipeline: full`, `split: test`, `smoke_run: false`, and its
  `config` block shows every optional stage off — i.e. the wired pipeline
  equals the kept stack. `validate_results.py` passes all checks including
  6-decimal agreement with MTEB's own cached result.
- **STEP 11 P1 property holds.** Full-corpus probe: cold 695.3 s / 8,754
  encodes; warm 0.7 s / **0** encodes; 100 snippets edited → 19.6 s /
  **exactly 100** encodes, 8,654 still cached. Rebuild cost tracks the change
  (2.8% of cold for a 1.1% change), not the corpus.
- **No dropped stage left half-built in the shipped path.** BM25, fusion and
  the Prism query/snippet processors remain unimplemented stubs behind
  default-off flags; `CrossEncoderReranker` is implemented but flag-gated with
  a NaN guard and a config comment recording why it must stay off. See the
  judgement call below.

## One judgement call for the human

`CrossEncoderReranker` is a **dropped** stage whose implementation is retained
(unreachable while `ENABLE_RERANK=False`). Strict "remove dead code from
dropped stages" would delete it. Recommendation: **keep**, because it is the
STEP 9 deliverable, it is flag-gated and documented, and it carries the guard
that stops `ms-marco-MiniLM-L-6-v2`'s all-NaN output from silently faking a
"reranking changed nothing" result. Deleting it would delete that protection.


---

# Outcomes (Phases 1–3)

## Phase 1 — fixed

| gap | commit | silent failure prevented |
|---|---|---|
| P0-1 baseline default / CMD | `ffd48b2` | shipping the bare encoder while believing it is the full pipeline |
| P0-2 no prefetch | `3b10482` | the graded run reaching for the Hub and timing out |
| P0-3 artifact outside the mount | `3b10482` | the deliverable discarded on container exit |
| P0-4 scipy undeclared | `3b10482` | clean container cannot run the diagnostic |
| P1-5 no Layer-1 tests | `b7cf313` | wrong prefix / mangled id / padding id, all scoring quietly worse |
| P1-6 stale TODO skips | `0bdae37` | green suite over an untested retrieval path |
| P2-7 rank-bm25, P2-8 join marker | `fe27765` | contract drift between the four join sites |

`run_eval.py` now also refuses to write the submission artifact from any run
that is not `--pipeline full --split test` unlimited, and refuses it at
argument-parse time rather than after eleven minutes of work.

The Layer-1 tests were **mutation-checked**, not merely observed green:
giving arctic a document prefix and reversing the index sort order each turn
the suite red.

## Phase 2 — submission re-locked

Full pipeline, full test split, current code (`9e49709`):

    NDCG@10 0.08222   MRR@10 0.06799   recall@100 0.30677   10.4 min

Identical to STEP 10 to 5dp — the check being made, since the STEP 11 cache
rewiring must not move the number. `validate_results.py`: all checks pass,
6dp agreement with MTEB's cached result.

## Phase 3 — repro gate: PARTIAL

| check | result |
|---|---|
| `pytest -m "not slow"` | **PASS** — 62 passed, 41 skipped |
| offline end-to-end, cold embedding cache, `HF_HUB_OFFLINE=1` | **PASS** |
| full test split offline, compared to the locked artifact | **PASS — reproduces exactly** |
| clean-container build + run | **NOT RUN — no Docker daemon on this machine** |

The offline run used an empty `PRISM_DATA_DIR`, so all 8,754 document vectors
were re-encoded from scratch, with `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` set. It reproduced
`ndcg_at_10`, `mrr_at_10`, `recall_at_100` and `main_score` **bit-for-bit**
against the locked artifact. That proves the pipeline needs no network and no
pre-existing cache.

What it does **not** prove is the image itself: the pinned `pip install`, the
build-time `prefetch_assets.py --verify`, and the `CMD`. `docker` is not
installed here, so `Dockerfile` changes are **unverified by execution**. The
Dockerfile was written to avoid the one parser ambiguity that could not be
checked (no comments inside backslash continuations).

**This is a Phase 4 stop condition: the clean-container half of the repro
gate has not been run.**
