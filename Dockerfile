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
# OMP/MKL        - 4 is only the value used while the image BUILDS. At run
#                  time the ENTRYPOINT at the bottom replaces it with the
#                  number of CPUs the container actually has, unless you pass
#                  your own (-e OMP_NUM_THREADS=N). faiss: the default
#                  FAISS_INDEX_FACTORY="Flat" uses an exact numpy index and
#                  never imports faiss. Any other factory does, and faiss-cpu
#                  and torch each ship an OpenMP runtime; loading both
#                  segfaults (exit 139) unless OMP_NUM_THREADS=1. So anyone
#                  switching to a faiss index must run with
#                  -e OMP_NUM_THREADS=1 (guide.md, section 3b).
#
# NOTE: no comments inside the ENV continuation below. A `#` line in the
# middle of a backslash-continued instruction is not portable across
# Dockerfile parsers, so it is written to avoid the question.
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
#
# BUILD TOOLCHAIN - why apt is here at all. pytrec-eval-terrier (transitive,
# via mteb) publishes linux wheels for x86_64 ONLY: every manylinux and
# musllinux artifact for 0.5.10 is x86_64, and there is no linux aarch64 wheel.
# On linux/arm64 pip therefore compiles its C/C++ extension from the sdist, and
# python:3.11-slim ships no compiler, so the install died with
#   error: [Errno 2] No such file or directory: 'gcc'
# CI never hit this because ubuntu-latest is linux/amd64 and gets the prebuilt
# wheel. The macOS wheels are universal2, which is why a native mac venv also
# installs it without a compiler.
#
# build-essential is installed for the compile and purged in the SAME layer, so
# it adds nothing to the final image. The compiled extension links only against
# libstdc++6 and libgcc-s1, which are base packages of python:3.11-slim (dpkg
# state "ii", not auto-installed), so --auto-remove cannot take them away.
#
# NOTE: no comments inside the backslash continuation below, for the same
# portability reason as the ENV block above.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Fail the BUILD, not the graded run, if the MTEB v2 API has shifted. The
# pytrec_eval import is here because it is the one extension compiled from
# source on this platform: if the purge above ever strips a library it needs,
# this fails at build time instead of as an ImportError mid-evaluation.
RUN python -c "\
import mteb; \
import pytrec_eval; \
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

# THREADS. Use every CPU the container has instead of a fixed count. The ENV
# near the top set 4 for the build; an ENV cannot be unset, only emptied, so
# it is emptied here and the entrypoint fills in $(nproc) at run time. An
# explicit -e OMP_NUM_THREADS=N or -e MKL_NUM_THREADS=N still wins, which is
# how a faiss index gets the OMP_NUM_THREADS=1 it needs (guide.md, section
# 3b). The entrypoint prints the values it chose to stderr, so every run log
# records its thread count, then execs the command unchanged: the CMD below,
# "docker run ... python ..." and "docker run -it ... bash" behave as before.
#
# These lines sit AFTER the dependency and prefetch layers on purpose, so
# changing them does not invalidate the ~1 GB of cached downloads above.
ENV OMP_NUM_THREADS= \
    MKL_NUM_THREADS=
ENTRYPOINT ["/bin/sh", "-c", "n=$(nproc); export OMP_NUM_THREADS=\"${OMP_NUM_THREADS:-$n}\" MKL_NUM_THREADS=\"${MKL_NUM_THREADS:-$n}\"; echo \"prism: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS\" >&2; exec \"$@\"", "--"]

# The SUBMISSION run: the full pipeline, the full test split, written into the
# mounted volume. Previously this was "--pipeline baseline", which regenerated
# the bare encoder's result and then discarded it inside the container.
CMD ["python", "scripts/run_eval.py", "--pipeline", "full", "--split", "test", "--output", "/app/results/appsretrieval_results.json"]
