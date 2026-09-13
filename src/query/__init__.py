"""Stage 1 - query preprocessing.  (owner: query)

Turns a raw natural-language request into a cleaned string plus an optional
category label, as defined by :class:`src.interfaces.QueryProcessor`.
"""

from src.query.preprocess import NoOpQueryProcessor, get_query_processor

__all__ = ["NoOpQueryProcessor", "get_query_processor"]
