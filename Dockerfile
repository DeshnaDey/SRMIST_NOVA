# =============================================================================
# PRISM Agentic Code Intelligence - CPU-only evaluation image
#
# Produces the SUBMISSION artifact, offline, from a baked model and dataset.
#
#   docker build -t prism-retrieval .
#   docker run --rm -v "$PWD/results:/app/results" prism-retrieval
#
# The run needs no network. That is enforced, not hoped for: HF_HUB_OFFLINE=1
# is set in the image, so if the prefetch ever fails to cover what the runtime
# loads, the container fails loudly at grade time instead of quietly reaching
# for the Hub and timing out.
# =============================================================================

# CPU-only base. Never a CUDA base image - the judging environment has no GPU
# and the CUDA layers add gigabytes for nothing.
FROM python:3.11-slim

WORKDIR /app

# HF_HOME        - one known cache path, so the baked assets are found at
#                  runtime and can be volume-mounted for inspection.
# PRISM_DATA_DIR - scratch for embeddings and the content-hash cache. Kept off
#                  any synced directory: an iCloud-backed data dir cost this
#                  project a 27 MB read that failed with [Errno 60] (guide.md).
# OMP/MKL        - reproducible CPU timings, and load-bearing for correctness
#                  if FAISS_INDEX_FACTORY is ever moved off "Flat": faiss-cpu
#                  and torch each ship an OpenMP runtime and loading both
#                  segfaults (exit 139). The default "Flat" path uses an exact
#                  numpy index and never imports faiss, so this is
#                  belt-and-braces rather than the fix.
#
# NOTE: no comments inside the ENV continuation below. A `#` line in the
# middle of a backslash-continued instruction is not portable across
# Dockerfile parsers, and this image cannot be build-tested on the dev
# machine (no Docker daemon), so it is written to avoid the question.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PRISM_DATA_DIR=/app/data \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Dependencies first, in their own layer, so code edits don't re-install torch.
COPY requirements.txt requirements-dev.txt ./

# torch must come from the CPU index BEFORE the rest. This base image is
# Linux, where the default PyPI wheels ARE the CUDA builds (~2.5 GB) - so
# unlike on a Mac, the index-url here is load-bearing, not cosmetic.
# Version kept in lockstep with requirements.txt.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Fail the BUILD, not the graded run, if the MTEB v2 API has shifted.
RUN python -c "\
import mteb; \
from mteb.models.abs_encoder import AbsEncoder; \
from mteb.models import ModelMeta; \
assert mteb.get_task('AppsRetrieval'); \
assert hasattr(mteb, 'evaluate') and hasattr(mteb, 'SearchProtocol'); \
print('mteb OK', mteb.__version__)"

# tests/ is copied so the repro gate can run INSIDE the image. An image that
# cannot run its own test suite cannot be verified on the judging machine.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml ./

RUN mkdir -p /app/results /app/data

# Bake the model AND the dataset, then prove the bake covers the runtime load
# by re-loading both with the Hub disabled. The id and revision are read from
# the MTEB task itself, so they cannot drift out of step with what
# task.load_data() actually requests - a hardcoded revision here would be a
# near-miss that still hits the network at grade time.
RUN python scripts/prefetch_assets.py --verify

# From here on the image is sealed: any attempt to reach the Hub is an error
# rather than a silent download.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# The SUBMISSION run: the full pipeline, the full test split, written into the
# mounted volume. Previously this was "--pipeline baseline", which regenerated
# the bare encoder's result and then discarded it inside the container.
CMD ["python", "scripts/run_eval.py", "--pipeline", "full", "--split", "test", "--output", "/app/results/appsretrieval_results.json"]
