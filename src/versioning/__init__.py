"""Hash-based embedding cache.  (owner: eval)

Encoding the APPS corpus on CPU is the slowest step in the pipeline. This
package keys embeddings by a hash of (text + model + config) so that changing
only the reranker, or only the fusion weights, costs seconds instead of an
hour of re-encoding.
"""

from src.versioning.cache import DiskEmbeddingCache, NoOpEmbeddingCache, get_cache

__all__ = ["DiskEmbeddingCache", "NoOpEmbeddingCache", "get_cache"]
