"""Content-addressed embedding cache.  (owner: eval)

Implements :class:`src.interfaces.EmbeddingCache`.

THE CORRECTNESS RULE
--------------------
A cache key must change whenever the embedding would change. If it doesn't,
you silently serve stale vectors: the corpus preprocessing improves, the
numbers don't move, and three people spend an afternoon debugging a retriever
that was never the problem.

So the key must incorporate, at minimum:
  * the exact text being encoded (which covers preprocessing changes)
  * the model name (``config.DENSE_MODEL_NAME``)
  * ``config.CACHE_VERSION`` - the manual escape hatch, bump it when in doubt
  * any encode-time flag that alters the output: normalization, prompt
    prefixes, truncation length

When unsure whether something belongs in the key, put it in. A false cache
miss costs CPU time; a false cache hit costs a corrupted experiment log.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from src import config

logger = logging.getLogger(__name__)


class NoOpEmbeddingCache:
    """Disabled cache: every lookup misses, every store is discarded.

    Used when ``config.ENABLE_EMBEDDING_CACHE`` is False, and as the safe
    fallback if the cache directory is not writable. Satisfies the
    :class:`src.interfaces.EmbeddingCache` protocol structurally.
    """

    def get(self, key: str) -> Any | None:
        """Always a miss."""
        return None

    def put(self, key: str, value: Any) -> None:
        """Discard the value."""
        return None

    def make_key(self, text: str, model_name: str, **extra: Any) -> str:
        """Delegate to the shared key function so keys stay comparable."""
        return make_cache_key(text, model_name, **extra)


class DiskEmbeddingCache:
    """Embeddings persisted to ``config.CACHE_DIR`` as .npy files.

    Design notes
    ------------
    * One file per key, named after the key, sharded into subdirectories by the
      first two hex characters. A single flat directory holding 100k files is
      painfully slow to list on macOS.
    * ``numpy.save`` / ``numpy.load``; keep ``allow_pickle=False`` - these are
      plain float arrays and pickle here is both slower and a footgun.
    * A corrupt or truncated file must be treated as a MISS, not an exception.
      A half-written cache entry from an interrupted run should cost one
      re-encode, not the whole evaluation.
    * Per-snippet files are simple and resumable. If they turn out to be too
      slow at corpus scale, batch into shards keyed by a hash of the whole
      batch - but measure first; simple and working beats clever and unfinished.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Parameters
        ----------
        cache_dir:
            Root of the cache tree. Defaults to ``config.CACHE_DIR``.
        """
        self.cache_dir = cache_dir or config.CACHE_DIR
        #: Entries are stored under a per-VERSION subtree, so every entry is
        #: tagged with the cache version by construction rather than by a
        #: sidecar that can drift out of step with the files it describes.
        #: Two consequences that matter: entries from different versions can
        #: never be confused for one another, and retiring a version is a
        #: directory removal rather than a scan.
        self.root = self.cache_dir / config.CACHE_VERSION
        self.writable = True
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            probe = self.root / ".write_probe"
            probe.write_bytes(b"")
            probe.unlink()
        except Exception as exc:  # noqa: BLE001
            # Not fatal. A read-only or flaky cache directory should cost CPU,
            # never a run - this box has already lost a read to an iCloud
            # timeout mid-session.
            logger.warning("Cache dir %s is not writable (%s); running without "
                           "a persistent cache.", self.root, exc)
            self.writable = False

    def _path(self, key: str) -> Path:
        # Sharded by the first two hex characters: a single flat directory
        # holding ~100k files is painfully slow to list on macOS.
        return self.root / key[:2] / f"{key}.npy"

    def get(self, key: str) -> Any | None:
        """Return the cached array for ``key``, or ``None`` on a miss.

        Never raises. A truncated or corrupt entry - the signature of a run
        interrupted mid-write - is treated as a miss and costs one re-encode,
        not the evaluation.
        """
        import numpy as np

        path = self._path(key)
        try:
            if not path.exists():
                return None
            return np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Corrupt/unreadable cache entry %s (%s); miss", path, exc)
            return None

    def put(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``.

        Written to a temporary file in the same directory and ``os.replace``d
        into place. ``os.replace`` is atomic within a filesystem, so an
        interrupted run can leave a stray ``.tmp`` but never a half-written
        entry that a later run reads back as valid.
        """
        if not self.writable:
            return
        import numpy as np

        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                tmp = Path(handle.name)
                np.save(handle, np.asarray(value), allow_pickle=False)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not write cache entry %s (%s)", path, exc)
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
            except Exception:  # noqa: BLE001
                pass

    def stats(self) -> dict[str, Any]:
        """Entry count and on-disk size for the ACTIVE version.

        Used by ``scripts/cache_rebuild_demo.py`` to show that a rebuild
        touches only what changed.
        """
        total = size = 0
        if self.root.exists():
            for shard in self.root.iterdir():
                if not shard.is_dir():
                    continue
                for entry in shard.glob("*.npy"):
                    total += 1
                    size += entry.stat().st_size
        return {"version": config.CACHE_VERSION, "entries": total,
                "bytes": size, "root": str(self.root)}

    def make_key(self, text: str, model_name: str, **extra: Any) -> str:
        """Delegate to the shared key function."""
        return make_cache_key(text, model_name, **extra)


def make_cache_key(text: str, model_name: str, **extra: Any) -> str:
    """Derive a stable cache key from content and configuration.

    Implemented here (not left as a TODO) because every cache implementation
    must produce identical keys for identical inputs, or entries written by one
    run become invisible to the next.

    Parameters
    ----------
    text:
        The exact string that will be encoded, AFTER preprocessing.
    model_name:
        The encoder checkpoint identifier.
    **extra:
        Any other flag that changes the resulting vector - ``normalize``,
        ``prompt_prefix``, ``max_seq_length``. Sorted by key, so call order
        never affects the result.

    Returns
    -------
    str
        A 64-character hex SHA-256 digest.
    """
    parts = [
        config.CACHE_VERSION,
        model_name,
        *(f"{k}={extra[k]!r}" for k in sorted(extra)),
        text,
    ]
    # "\x00" is a separator that cannot occur in the parts, so ("ab", "c") and
    # ("a", "bc") can never collide onto the same key.
    payload = "\x00".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache() -> Any:
    """Return the cache selected by ``config.ENABLE_EMBEDDING_CACHE``."""
    if config.ENABLE_EMBEDDING_CACHE:
        return DiskEmbeddingCache()
    return NoOpEmbeddingCache()
