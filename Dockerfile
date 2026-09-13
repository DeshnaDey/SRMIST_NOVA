# =============================================================================
# PRISM Agentic Code Intelligence - CPU-only evaluation image
#
# STATUS: STUB. It captures the right shape and the CPU-only constraints, but
# nobody has built it yet. Work through the TODOs before relying on it.
#
#   docker build -t prism-retrieval .
#   docker run --rm -v "$PWD/results:/app/results" prism-retrieval
# =============================================================================

# CPU-only base. Never a CUDA base image - the judging environment has no GPU
# and the CUDA layers add gigabytes for nothing.
FROM python:3.11-slim

# TODO: add build-essential only if a dependency needs to compile from source.
# faiss-cpu and rank_bm25 ship wheels, so the slim image should be enough -
# check before adding ~300 MB of toolchain.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Keep the HF cache on one known path so it can be volume-mounted and
    # survive container restarts instead of re-downloading each run.
    HF_HOME=/app/.cache/huggingface \
    # Pin thread counts for reproducible CPU timings.
    # TODO: match these to the judging machine's core count.
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Dependencies first, in their own layer, so code edits don't re-install torch.
COPY requirements.txt .

# torch must come from the CPU index BEFORE the rest. This base image is
# Linux, where the default PyPI wheels ARE the CUDA builds (~2.5 GB) - so
# unlike on a Mac, the index-url here is load-bearing, not cosmetic.
# Version kept in lockstep with requirements.txt.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Fail the BUILD, not the graded run, if the MTEB v2 API has shifted.
RUN python -c "\
import mteb; \
from mteb.models.abs_encoder import AbsEncoder; \
from mteb.models import ModelMeta; \
assert mteb.get_task('AppsRetrieval'); \
assert hasattr(mteb, 'evaluate') and hasattr(mteb, 'SearchProtocol'); \
print('mteb OK', mteb.__version__)"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# TODO: bake the model into the image so the graded run needs no network:
#   RUN python -c "from sentence_transformers import SentenceTransformer; \
#                  SentenceTransformer('<config.DENSE_MODEL_NAME>')"
# Decide this once the checkpoint is final - it adds size but removes a
# download from the critical path (and a failure mode we cannot debug live).

# TODO: consider prefetching the dataset too, for the same reason.

# Writable output directory for appsretrieval_results.json.
RUN mkdir -p /app/results /app/data

CMD ["python", "scripts/run_eval.py", "--pipeline", "baseline"]
