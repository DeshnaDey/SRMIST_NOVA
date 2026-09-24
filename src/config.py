"""Central configuration for the PRISM code-retrieval pipeline.

Every tunable lives here so that experiments are reproducible: to reproduce a
row in `experiments.md` you should only ever need this file plus a git SHA.

CONVENTION
----------
Two labels appear on values below, and they mean different things:

``# UNVALIDATED DEFAULT``
    A value on the SHIPPED path that works and was never swept. It is a
    sensible default, not a guess pulled from nowhere, but no measurement
    justifies this number over its neighbours. Sweeping one is a real
    experiment; log the before/after in ``experiments.md``.

``# PLACEHOLDER``
    A value belonging to a stage that is gated out. Nothing reads it on the
    shipped path, so it has never had the chance to be right or wrong. Only
    relevant if that stage is ever revived.

OWNERSHIP
---------
The file is split into per-workstream sections so four people can edit it in
parallel without fighting over the same lines. Stay inside your own section.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# SHARED - paths and dataset identity (change only with team agreement)
# =============================================================================

#: Repository root, resolved from this file's location (``src/config.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Scratch space for embeddings, indexes and other large build artifacts.
#: Git-ignored. Override with PRISM_DATA_DIR to point at a fast local disk.
DATA_DIR: Path = Path(os.environ.get("PRISM_DATA_DIR", PROJECT_ROOT / "data"))

#: Where MTEB writes its own result tree.
RESULTS_DIR: Path = Path(os.environ.get("PRISM_RESULTS_DIR", PROJECT_ROOT / "results"))

#: The single deliverable file the submission is scored from.
RESULTS_JSON: Path = PROJECT_ROOT / "appsretrieval_results.json"

#: Hugging Face cache. Pointing this inside the repo keeps the Docker layer
#: self-contained; leave unset to use the machine-wide ~/.cache/huggingface.
HF_CACHE_DIR: Path | None = (
    Path(os.environ["PRISM_HF_CACHE"]) if "PRISM_HF_CACHE" in os.environ else None
)

#: MTEB task under evaluation. Do not change - this is the graded task.
MTEB_TASK_NAME: str = "AppsRetrieval"

#: Underlying dataset, for reference when loading the corpus directly.
DATASET_ID: str = "CoIR-Retrieval/apps"

#: Metrics we report. NDCG@10 is the headline number.
PRIMARY_METRIC: str = "ndcg_at_10"
SECONDARY_METRIC: str = "mrr_at_10"

#: Fixed seed for anything stochastic, so reruns are comparable.
RANDOM_SEED: int = 42

#: CPU-only build. Nothing in this project may request a GPU.
DEVICE: str = "cpu"

#: Threads for torch / faiss. None = let the library decide.
#: UNVALIDATED DEFAULT - set to physical core count if encode throughput is
#: the bottleneck. The library default has never been benchmarked against one.
NUM_THREADS: int | None = None


# =============================================================================
# SHARED - TOKEN BUDGETS (change only with team agreement)
# =============================================================================
#
# WHY THESE ARE NOT CHARACTER CAPS
# --------------------------------
# This project used to carry MAX_SNIPPET_CHARS = 4000 and MAX_QUERY_CHARS =
# 2000. Both were dead letters. The encoder truncates at its own token limit,
# and for the day-one checkpoint that limit is 254 content tokens - roughly
# 1,000 characters of Python and roughly 900 characters of English prose. A
# 4,000-character cap can never fire: the tokenizer has already cut the text
# to a quarter of that before the cap is consulted. The caps were denominated
# in the wrong unit, so they measured nothing and protected nothing.
#
# WHY THEY RESOLVE PER-MODEL
# --------------------------
# The checkpoints under consideration span 254 to 8192 tokens. Hardcoding a
# number means editing this file on every model swap and, worse, silently
# throwing away 97% of an 8k model's window if somebody forgets. These budgets
# therefore derive from whichever checkpoint is active.
#
# HOW TO READ THEM
# ----------------
#     config.MAX_SNIPPET_TOKENS   ->  int, budget for one corpus snippet
#     config.MAX_QUERY_TOKENS     ->  int, budget for one query
#
# Both are computed on first access and cached, via the module-level
# __getattr__ at the bottom of this file. They are attributes, not constants:
# resolving one may consult the Hub the first time, so do not read them inside
# a tight loop - hoist them out.

#: Tokens the encoder spends on its own special tokens ([CLS]/[SEP], <s>/</s>).
#: Subtracted from the model's window to get the usable content budget.
SPECIAL_TOKEN_ALLOWANCE: int = 2

#: Used only when a checkpoint advertises no usable limit at all.
FALLBACK_CONTEXT_TOKENS: int = 512

#: Guards against tokenizers that report a sentinel "no limit" value
#: (transformers uses 1000000000000000019884624838656 for exactly this).
_IMPLAUSIBLE_CONTEXT_TOKENS: int = 100_000


@lru_cache(maxsize=16)
def model_context_tokens(model_name: str | None = None) -> int:
    """Return the full sequence window of ``model_name``, in tokens.

    Resolution order, most authoritative first:

    1. ``MAX_SEQ_LENGTH`` if it has been set explicitly - an operator override
       always wins.
    2. ``sentence_bert_config.json`` on the Hub. This is the file
       SentenceTransformer itself reads to set ``max_seq_length``, so it is
       what will actually do the truncating. A tiny JSON download, no weights.
    3. The tokenizer's ``model_max_length``. Correct for plain transformers
       checkpoints, but note many sentence-transformers models inherit a much
       larger value here than the ST wrapper will really use - hence its
       position below the file above.
    4. ``FALLBACK_CONTEXT_TOKENS``.

    Cached, so the Hub is consulted at most once per checkpoint per process.
    Never raises: an unreachable Hub degrades to the fallback rather than
    killing an evaluation.
    """
    name = model_name or DENSE_MODEL_NAME

    if MAX_SEQ_LENGTH is not None:
        return int(MAX_SEQ_LENGTH)

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(name, "sentence_bert_config.json")
        value = json.loads(Path(path).read_text(encoding="utf-8")).get("max_seq_length")
        if value and 0 < int(value) < _IMPLAUSIBLE_CONTEXT_TOKENS:
            return int(value)
    except Exception as exc:  # noqa: BLE001 - any failure is just a miss
        logger.debug("No sentence_bert_config.json for %s (%s)", name, exc)

    try:
        from transformers import AutoTokenizer

        value = int(AutoTokenizer.from_pretrained(name).model_max_length)
        if 0 < value < _IMPLAUSIBLE_CONTEXT_TOKENS:
            return value
    except Exception as exc:  # noqa: BLE001
        logger.debug("No usable model_max_length for %s (%s)", name, exc)

    logger.warning(
        "Could not resolve a context window for %r; falling back to %d tokens. "
        "If that is wrong, set MAX_SEQ_LENGTH explicitly.",
        name, FALLBACK_CONTEXT_TOKENS,
    )
    return FALLBACK_CONTEXT_TOKENS


def max_snippet_tokens(model_name: str | None = None) -> int:
    """Token budget for one corpus snippet, for ``model_name``.

    Override with ``PRISM_MAX_SNIPPET_TOKENS`` to test a deliberately tighter
    budget than the model allows (e.g. to trade recall for encode speed).
    """
    override = os.environ.get("PRISM_MAX_SNIPPET_TOKENS")
    if override:
        return max(1, int(override))
    return max(1, model_context_tokens(model_name) - SPECIAL_TOKEN_ALLOWANCE)


def max_query_tokens(model_name: str | None = None) -> int:
    """Token budget for one query, for ``model_name``.

    Override with ``PRISM_MAX_QUERY_TOKENS``. Separate from the snippet budget
    on purpose: the two sides overflow at very different rates here, so being
    able to move one without the other is the point.
    """
    override = os.environ.get("PRISM_MAX_QUERY_TOKENS")
    if override:
        return max(1, int(override))
    return max(1, model_context_tokens(model_name) - SPECIAL_TOKEN_ALLOWANCE)


# =============================================================================
# WORKSTREAM 1 - query preprocessing              (owner: query)
# =============================================================================

#: Toggle so the rest of the team can A/B your stage without editing code.
ENABLE_QUERY_PREPROCESSING: bool = False

#: Labels your classifier may emit. Consumers must tolerate an unknown label.
#: PLACEHOLDER - replace with the taxonomy you actually settle on.
QUERY_CATEGORIES: tuple[str, ...] = (
    "algorithm",
    "data_structure",
    "string_manipulation",
    "math",
    "io_parsing",
    "other",
)

#: Drop queries shorter than this after cleaning (characters).
#: Characters are the right unit here - this is a "did cleaning destroy the
#: query" guard, not a context-window budget.
#: PLACEHOLDER
MIN_QUERY_CHARS: int = 3

#: Hard cap on query length handed to the encoder, in TOKENS.
#: Resolved per-model - see ``max_query_tokens()`` and the TOKEN BUDGETS block
#: in the shared section. Read it as ``config.MAX_QUERY_TOKENS``.
#:
#: Queries are where this actually bites: 61.9% of them overflow the current
#: model's window, against 23.5% of snippets (data/inspection_report.md).


# =============================================================================
# WORKSTREAM 2 - snippet preprocessing + indexing (owner: corpus)
# =============================================================================

ENABLE_SNIPPET_PREPROCESSING: bool = False

#: Truncate snippets to this many TOKENS before encoding.
#: Resolved per-model - see ``max_snippet_tokens()`` and the TOKEN BUDGETS
#: block in the shared section. Read it as ``config.MAX_SNIPPET_TOKENS``.
#:
#: MEASURED (data/inspection_report.md): 23.5% of snippets overflow the current
#: model's 254-token content window, median 132 tokens. Real, but it is the
#: SMALLER of the two truncation problems - queries overflow 61.9% of the time,
#: roughly 2.6x as often. Spend effort on the query side first.

#: Strip comments/docstrings from code before encoding.
#: PLACEHOLDER - may help (less noise) or hurt (comments carry NL signal). Test it.
STRIP_COMMENTS: bool = False

#: Split long snippets into overlapping windows and pool the results.
#: PLACEHOLDER
ENABLE_CHUNKING: bool = False
CHUNK_SIZE_CHARS: int = 1_500      # PLACEHOLDER
CHUNK_OVERLAP_CHARS: int = 200     # PLACEHOLDER

#: FAISS index factory string. "Flat" = exact brute force, correct but O(N).
#: UNVALIDATED DEFAULT, and deliberately the conservative one: exact search
#: cannot cost recall. Move to "IVF1024,Flat" or HNSW only if latency forces
#: it, and re-measure NDCG afterwards because ANN is lossy.
FAISS_INDEX_FACTORY: str = "Flat"

#: Cosine similarity via inner product requires L2-normalized vectors.
NORMALIZE_EMBEDDINGS: bool = True


# =============================================================================
# WORKSTREAM 3 - retrieval, fusion, reranking     (owner: retrieval)
# =============================================================================

# ---- Bi-encoder (dense) -----------------------------------------------------

#: MEASURED on the full train split - see experiments.md and
#: data/model_benchmark.json. Selected on recall@100 (0.6870), which is the
#: ceiling on anything the reranker can later recover; it also led NDCG@10 and
#: MRR, and was marginally the fastest of the 512-token candidates.
#:
#:     model         r@100    r@10   NDCG@10   encode   q-latency
#:     all-MiniLM    0.6560  0.5222  0.42349   114.8s     19.35ms
#:     e5-base-v2    0.6728  0.5226  0.43748   872.4s    167.89ms
#:     bge-base-1.5  0.6642  0.5460  0.45628   884.1s    172.25ms
#:     arctic-m      0.6870  0.5458  0.45863   842.8s    160.66ms  <- selected
DENSE_MODEL_NAME: str = "Snowflake/snowflake-arctic-embed-m"

#: Encoder batch size. Lower it if the CPU box starts swapping.
#: UNVALIDATED DEFAULT - affects throughput only, never the scores.
BATCH_SIZE: int = 32

#: Token limit for the bi-encoder; None = use the checkpoint's own default,
#: subject to ENCODER_WINDOW_CAP below.
MAX_SEQ_LENGTH: int | None = None

#: Hard ceiling on the encoder window, whatever a checkpoint advertises.
#:
#: The corpus maximum is 60,599 tokens while p99 is only 1,023, so this binds
#: on a handful of documents. It matters because attention is quadratic: under
#: a 254-token model that outlier was truncated away for free, but hand it to
#: a model advertising 8k-32k and those few documents can dominate the entire
#: corpus encode. Set to None to honour the checkpoint's full window.
ENCODER_WINDOW_CAP: int | None = 2_048

#: Some checkpoints require asymmetric prefixes and silently underperform
#: without them - the run completes and the number is just quietly bad.
#: MUST be kept in step with DENSE_MODEL_NAME. Known schemes:
#:
#:   Snowflake/snowflake-arctic-embed-m  query only, instruction below
#:       (card usage: "use the query prefix below (just on the query)")
#:   BAAI/bge-base-en-v1.5               query only, same instruction string
#:       (card "Usage for Retrieval": no instruction on passages)
#:   intfloat/e5-base-v2                 "query: " / "passage: "
#:       (card FAQ, question 1 - asymmetric retrieval uses both)
#:   sentence-transformers/all-MiniLM-*  none
#:   jinaai/jina-embeddings-v2-base-code none
QUERY_PROMPT_PREFIX: str = "Represent this sentence for searching relevant passages: "
DOCUMENT_PROMPT_PREFIX: str = ""

# ---- BM25 (sparse) ----------------------------------------------------------

ENABLE_BM25: bool = False

#: rank_bm25 Okapi defaults. PLACEHOLDER - tune on a dev slice, not on the test set.
BM25_K1: float = 1.5
BM25_B: float = 0.75

# ---- Candidate depths -------------------------------------------------------
# Pipeline shape:  retrieve TOP_K_DENSE + TOP_K_BM25 -> fuse -> TOP_K_FUSED
#                  -> rerank -> TOP_K_FINAL
# TOP_K_FINAL must stay >= 10 or NDCG@10 is truncated and the score is invalid.

TOP_K_DENSE: int = 100   # UNVALIDATED DEFAULT - live; caps recall@100
TOP_K_BM25: int = 100    # PLACEHOLDER - gated stage
TOP_K_FUSED: int = 100   # UNVALIDATED DEFAULT - live; pool depth out of fusion
TOP_K_FINAL: int = 10    # returned to MTEB; >= 10 required for NDCG@10

# ---- Fusion -----------------------------------------------------------------

#: "rrf" (rank-based, scale-free) or "weighted_sum" (needs score normalization).
#: PLACEHOLDER
FUSION_STRATEGY: str = "rrf"

#: RRF smoothing constant. 60 is the value from the original RRF paper.
RRF_K: int = 60  # PLACEHOLDER

#: Weights for (dense, bm25) when FUSION_STRATEGY == "weighted_sum".
#: PLACEHOLDER - must sum to 1.0 by convention.
FUSION_WEIGHTS: tuple[float, float] = (0.5, 0.5)

# ---- Reranking --------------------------------------------------------------

ENABLE_RERANK: bool = False

#: Cross-encoder checkpoint. Cross-encoders are ~100x slower per pair than the
#: bi-encoder, so RERANK_TOP_N is the real cost knob on CPU.
#:
#: DO NOT SET THIS BACK TO ms-marco-MiniLM-L-6-v2. That checkpoint - the
#: obvious default, and what this constant used to name - returns **NaN for
#: every pair** on the pinned stack: fp32 weights all finite, NaN out of
#: encoder layer 0, under both sdpa and eager attention. It is
#: checkpoint-specific, not a stack problem: L-4, L-12, TinyBERT-L-2 and
#: bge-reranker-base all score finite here.
#:
#: The failure is silent and flattering. Sorting by NaN preserves the input
#: order, so the pipeline reports a spotless "reranking changed nothing"
#: instead of an error - a fake null that cost an entire measurement pass
#: before it was caught. CrossEncoderReranker now guards against it.
#:
#: MEASURED (experiments.md decision log): reranking is DISABLED because it
#: LOSES. Both working candidates degrade NDCG@10 monotonically with pool
#: depth - L-4 -0.099 at pool 10 down to -0.258 at pool 100; bge-reranker-base
#: -0.160 and -0.266. Both beat a random reordering, so they carry some
#: signal; both lose to arctic's own ordering, so applying them overwrites a
#: better ranking with a worse one. This name is kept pointing at a WORKING
#: checkpoint only so that flipping ENABLE_RERANK cannot resurrect the NaN.
RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"

#: How many fused candidates actually reach the cross-encoder.
#: PLACEHOLDER - start small (25-50); CPU latency scales linearly with this.
RERANK_TOP_N: int = 50

RERANK_BATCH_SIZE: int = 32  # PLACEHOLDER


# =============================================================================
# WORKSTREAM 4 - evaluation + caching             (owner: eval)
# =============================================================================

#: Persist embeddings keyed by content hash so reruns skip re-encoding.
#: See src/versioning/cache.py.
ENABLE_EMBEDDING_CACHE: bool = True

CACHE_DIR: Path = DATA_DIR / "embedding_cache"

#: Bump to invalidate every cached embedding at once (e.g. after changing
#: preprocessing in a way the content hash cannot see).
CACHE_VERSION: str = "v1"

#: Cap evaluation to N queries for a fast smoke run. None = full evaluation.
#: MUST be None for any number recorded in experiments.md or submitted.
SMOKE_TEST_QUERY_LIMIT: int | None = None

#: Release tag for the final submission.
RELEASE_TAG: str = "PRISM_GENAI_HACKATHON_Y2026"


def __getattr__(name: str) -> Any:
    """Resolve the per-model token budgets on first attribute access.

    PEP 562 module ``__getattr__``. It exists so the budgets can be READ like
    the plain constants they replaced::

        if n_tokens > config.MAX_SNIPPET_TOKENS: ...

    while still being derived from whichever checkpoint ``DENSE_MODEL_NAME``
    currently names. Writing them as real module constants would mean either
    hardcoding a number (wrong the moment we swap models) or consulting the
    Hub at import time (which would make ``import config`` do network I/O).

    Also catches the two deleted names and says what to use instead, rather
    than letting a stale reference fail as a bare AttributeError.
    """
    if name == "MAX_SNIPPET_TOKENS":
        return max_snippet_tokens()
    if name == "MAX_QUERY_TOKENS":
        return max_query_tokens()
    if name in ("MAX_SNIPPET_CHARS", "MAX_QUERY_CHARS"):
        replacement = name.replace("_CHARS", "_TOKENS")
        raise AttributeError(
            f"config.{name} was removed: it was a character cap on a limit the "
            f"tokenizer enforces in tokens, so it never fired. Use "
            f"config.{replacement}, which resolves from the active model."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_dirs() -> None:
    """Create the scratch directories this config points at.

    Safe to call repeatedly; called by ``scripts/run_eval.py`` on startup.
    """
    for directory in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, object]:
    """Return the active configuration as a flat, JSON-serializable dict.

    Dumped alongside every evaluation run so an experiments.md row can always
    be traced back to the exact settings that produced it.
    """
    return {
        "task": MTEB_TASK_NAME,
        "dense_model": DENSE_MODEL_NAME,
        "rerank_model": RERANK_MODEL_NAME if ENABLE_RERANK else None,
        "batch_size": BATCH_SIZE,
        "query_preprocessing": ENABLE_QUERY_PREPROCESSING,
        "snippet_preprocessing": ENABLE_SNIPPET_PREPROCESSING,
        "bm25": ENABLE_BM25,
        "rerank": ENABLE_RERANK,
        "fusion_strategy": FUSION_STRATEGY if ENABLE_BM25 else None,
        "top_k_dense": TOP_K_DENSE,
        "top_k_bm25": TOP_K_BM25 if ENABLE_BM25 else None,
        "top_k_fused": TOP_K_FUSED,
        "top_k_final": TOP_K_FINAL,
        "rerank_top_n": RERANK_TOP_N if ENABLE_RERANK else None,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "seed": RANDOM_SEED,
        "smoke_limit": SMOKE_TEST_QUERY_LIMIT,
        # Resolved from the active checkpoint, so a results file records the
        # budget that actually applied rather than a constant someone guessed.
        # Carried over from shanavi-work: these shape a run and belong in the
        # record, so a results file is reproducible from the file alone.
        "max_seq_length": MAX_SEQ_LENGTH,
        "encoder_window_cap": ENCODER_WINDOW_CAP,
        "faiss_index_factory": FAISS_INDEX_FACTORY,
        "cache_version": CACHE_VERSION if ENABLE_EMBEDDING_CACHE else None,
        "model_context_tokens": model_context_tokens(),
        "max_snippet_tokens": max_snippet_tokens(),
        "max_query_tokens": max_query_tokens(),
    }
