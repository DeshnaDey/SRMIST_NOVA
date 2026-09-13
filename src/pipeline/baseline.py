"""Day-one baseline: a plain bi-encoder, no pipeline stages.

This is the ONLY module in the project that is fully implemented, on purpose.
It exists so that on day one we can run the real evaluation end to end, get a
real NDCG@10, and write the first row of experiments.md. Everything the team
builds afterwards is measured against that number.

No query preprocessing, no snippet preprocessing, no BM25, no fusion, no
reranking. MTEB v2 wraps a bare encoder into a search model automatically, so
implementing ``encode()`` is enough to score the AppsRetrieval task.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src import config
from src.pipeline.mteb_compat import extract_texts, resolve_abs_encoder

logger = logging.getLogger(__name__)

#: MTEB's encoder base class when available, else ``object``. See mteb_compat.
_AbsEncoder, _HAS_ABS_ENCODER = resolve_abs_encoder()


class BaselineEncoder(_AbsEncoder):  # type: ignore[misc,valid-type]
    """A SentenceTransformer bi-encoder behind the MTEB v2 encoder protocol.

    The checkpoint is ``config.DENSE_MODEL_NAME`` - a placeholder constant, so
    swapping models is a one-line config change and never a code change.

    Notes
    -----
    The model is loaded lazily on first ``encode()`` rather than in
    ``__init__``, so that constructing the object (in a test, or to read its
    metadata) doesn't pull a few hundred MB off the Hub.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Parameters
        ----------
        model_name:
            Bi-encoder checkpoint. Defaults to ``config.DENSE_MODEL_NAME``.
        """
        if _HAS_ABS_ENCODER:
            try:
                super().__init__()
            except TypeError:
                # Some AbsEncoder revisions take required constructor args.
                # Skipping the super call is safe: we override encode() and
                # only lose the inherited similarity defaults.
                logger.debug("AbsEncoder.__init__ needs args; skipping super()")

        self.model_name = model_name or config.DENSE_MODEL_NAME
        self._model: Any | None = None
        self.mteb_model_meta = _build_model_meta(self.model_name)

    # -- model loading --------------------------------------------------------

    @property
    def model(self) -> Any:
        """The loaded SentenceTransformer, instantiated on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading bi-encoder %s on %s", self.model_name, config.DEVICE)
            self._model = SentenceTransformer(self.model_name, device=config.DEVICE)
            if config.MAX_SEQ_LENGTH is not None:
                self._model.max_seq_length = config.MAX_SEQ_LENGTH
        return self._model

    # -- the MTEB v2 contract -------------------------------------------------

    def encode(
        self,
        inputs: Any,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Embed a batch of inputs.

        Implements the MTEB v2 ``EncoderProtocol``::

            encode(inputs: DataLoader[BatchedInput], task_metadata, hf_split,
                   hf_subset, prompt_type=None, **kwargs) -> Array

        Parameters
        ----------
        inputs:
            A ``DataLoader`` yielding batch dicts keyed by modality. Note this
            is the v2 contract - v1 passed a ``list[str]``. Unpacked by
            ``mteb_compat.extract_texts``, which also accepts plain lists so
            tests don't have to build a DataLoader.
        task_metadata, hf_split, hf_subset:
            Supplied by MTEB to identify what is being encoded. Unused by this
            baseline; a real implementation could vary prompts per task.
        prompt_type:
            ``PromptType.query`` or ``PromptType.document`` (or ``None``).
            Used to pick the asymmetric prefix that E5/BGE/GTE-family models
            require - they degrade quietly without it.
        **kwargs:
            Passed through from ``mteb.evaluate(..., encode_kwargs=...)``;
            may carry ``batch_size``, ``normalize_embeddings``, etc.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_inputs, embedding_dim)``, float32, one row per input in
            input order. Row order is the contract - MTEB maps row ``i`` back
            to input item ``i``.
        """
        texts = extract_texts(inputs)
        if not texts:
            # An empty shard is legal; return a correctly-shaped empty array so
            # downstream vstack calls don't choke on a (0,) instead of (0, d).
            dim = getattr(self.model, "get_sentence_embedding_dimension", lambda: 0)()
            return np.zeros((0, dim or 0), dtype=np.float32)

        prefix = _prefix_for(prompt_type)
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=kwargs.get("batch_size", config.BATCH_SIZE),
            normalize_embeddings=kwargs.get(
                "normalize_embeddings", config.NORMALIZE_EMBEDDINGS
            ),
            convert_to_numpy=True,
            show_progress_bar=kwargs.get("show_progress_bar", False),
        )
        return np.asarray(embeddings, dtype=np.float32)


def _prefix_for(prompt_type: Any) -> str:
    """Return the asymmetric prompt prefix for a v2 ``PromptType``.

    Both prefixes default to ``""`` in config, so models that don't use
    prompts are unaffected. The comparison goes through ``str()`` because the
    enum's import path has moved between MTEB releases and its ``str`` form
    ("PromptType.query") has not.
    """
    if prompt_type is None:
        return ""
    name = str(prompt_type).lower()
    if "query" in name:
        return config.QUERY_PROMPT_PREFIX
    if "document" in name or "passage" in name or "corpus" in name:
        return config.DOCUMENT_PROMPT_PREFIX
    return ""


def _build_model_meta(model_name: str) -> Any | None:
    """Build a ``ModelMeta`` describing this model, or ``None`` if unavailable.

    MTEB uses the metadata to name the result directory and to pick the
    similarity function. Its required fields have changed across 2.x releases,
    so construction is attempted and failure is tolerated - MTEB falls back to
    deriving what it needs from the model itself.

    TODO(eval): once the mteb version is pinned, populate this properly
    (revision, release date, embedding dim, license, framework) so the results
    JSON is fully self-describing for the submission.
    """
    try:
        from mteb import ModelMeta
    except ImportError:
        return None

    try:
        return ModelMeta(
            name=model_name,
            revision=None,
            release_date=None,
            languages=["eng-Latn"],
            similarity_fn_name="cosine",
            framework=["Sentence Transformers"],
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Could not build ModelMeta (%s); letting MTEB infer it", exc)
        return None
