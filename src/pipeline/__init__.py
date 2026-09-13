"""MTEB v2 adapters.  (shared - change only with team agreement)

Two objects can be handed to ``mteb.evaluate``:

``BaselineEncoder``
    A plain bi-encoder implementing the v2 encoder protocol. No preprocessing,
    no BM25, no reranking. MTEB wraps it into a search model automatically.
    This is the day-one baseline and the control for every experiment.

``PrismSearch``
    Implements the v2 ``SearchProtocol`` directly, wiring all four workstreams
    together. This is what produces the submitted numbers - the official
    results JSON comes from our real pipeline, not from a bare encoder.
"""

from src.pipeline.baseline import BaselineEncoder
from src.pipeline.prism_search import PrismSearch

__all__ = ["BaselineEncoder", "PrismSearch"]
