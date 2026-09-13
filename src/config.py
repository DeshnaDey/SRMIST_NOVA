"""Central configuration for the PRISM code-retrieval pipeline.

Every tunable lives here so that experiments are reproducible: to reproduce a
row in `experiments.md` you should only ever need this file plus a git SHA.

CONVENTION
----------
Values marked ``# PLACEHOLDER`` are deliberate guesses that nobody has
validated yet. Replace them with something measured, then log the before/after
numbers in ``experiments.md``.

OWNERSHIP
---------
The file is split into per-workstream sections so four people can edit it in
parallel without fighting over the same lines. Stay inside your own section.
"""

from __future__ import annotations

import os
from pathlib import Path

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
#: PLACEHOLDER - set to physical core count if encode throughput is the bottleneck.
NUM_THREADS: int | None = None


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
#: PLACEHOLDER
MIN_QUERY_CHARS: int = 3

#: Hard cap on query length handed to the encoder (characters, pre-tokenization).
#: PLACEHOLDER
MAX_QUERY_CHARS: int = 2_000


# =============================================================================
# WORKSTREAM 2 - snippet preprocessing + indexing (owner: corpus)
# =============================================================================

ENABLE_SNIPPET_PREPROCESSING: bool = False

#: Truncate snippets to this many characters before encoding. APPS solutions
#: are long and the bi-encoder context window is short, so this matters.
#: PLACEHOLDER - measure before trusting.
MAX_SNIPPET_CHARS: int = 4_000

#: Strip comments/docstrings from code before encoding.
#: PLACEHOLDER - may help (less noise) or hurt (comments carry NL signal). Test it.
STRIP_COMMENTS: bool = False

#: Split long snippets into overlapping windows and pool the results.
#: PLACEHOLDER
ENABLE_CHUNKING: bool = False
CHUNK_SIZE_CHARS: int = 1_500      # PLACEHOLDER
CHUNK_OVERLAP_CHARS: int = 200     # PLACEHOLDER

#: FAISS index factory string. "Flat" = exact brute force, correct but O(N).
#: PLACEHOLDER - move to "IVF1024,Flat" or HNSW only if latency forces it, and
#: re-measure NDCG afterwards because ANN is lossy.
FAISS_INDEX_FACTORY: str = "Flat"

#: Cosine similarity via inner product requires L2-normalized vectors.
NORMALIZE_EMBEDDINGS: bool = True


# =============================================================================
# WORKSTREAM 3 - retrieval, fusion, reranking     (owner: retrieval)
# =============================================================================

# ---- Bi-encoder (dense) -----------------------------------------------------

#: PLACEHOLDER - the day-one baseline model. Pick something small and CPU-fast
#: first; only move to a bigger checkpoint once the harness is green end to end.
#: Candidates worth benchmarking: a general MiniLM-class model vs. a
#: code-specific bi-encoder. Log every swap in experiments.md.
DENSE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"  # PLACEHOLDER

#: Encoder batch size. Lower it if the CPU box starts swapping.
#: PLACEHOLDER
BATCH_SIZE: int = 32

#: Token limit for the bi-encoder; None = use the checkpoint's own default.
MAX_SEQ_LENGTH: int | None = None  # PLACEHOLDER

#: Some checkpoints (E5, BGE, GTE) require asymmetric prefixes and silently
#: underperform without them. Leave empty for models that don't use prompts.
#: PLACEHOLDER
QUERY_PROMPT_PREFIX: str = ""
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

TOP_K_DENSE: int = 100   # PLACEHOLDER
TOP_K_BM25: int = 100    # PLACEHOLDER
TOP_K_FUSED: int = 100   # PLACEHOLDER - how many survive fusion into the reranker
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

#: PLACEHOLDER - cross-encoder checkpoint. Cross-encoders are ~100x slower per
#: pair than the bi-encoder, so RERANK_TOP_N is the real cost knob on CPU.
RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # PLACEHOLDER

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
    }
