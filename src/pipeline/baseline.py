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
from functools import lru_cache
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
            logger.info("Loading bi-encoder %s on %s", self.model_name, config.DEVICE)
            self._model = _load_sentence_transformer(self.model_name)

            if config.MAX_SEQ_LENGTH is not None:
                self._model.max_seq_length = config.MAX_SEQ_LENGTH

            # Clamp to ENCODER_WINDOW_CAP. Attention is quadratic and the
            # corpus has a 60,599-token outlier against a p99 of 1,023, so a
            # long-context checkpoint can spend most of the encode on a
            # handful of documents. Never RAISES the window - only lowers it.
            cap = config.ENCODER_WINDOW_CAP
            current = getattr(self._model, "max_seq_length", None)
            if cap is not None and current and current > cap:
                logger.info(
                    "Capping %s window %d -> %d tokens (ENCODER_WINDOW_CAP)",
                    self.model_name, current, cap,
                )
                self._model.max_seq_length = cap
        return self._model

    # -- the MTEB v2 contract -------------------------------------------------

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Embed a batch of inputs.

        Implements the MTEB v2 ``EncoderProtocol``. VERIFIED against mteb
        2.20.11, where ``AbsEncoder.encode`` is the one abstract method::

            encode(self, inputs: DataLoader[BatchedInput], *,
                   task_metadata: TaskMetadata, hf_split: str, hf_subset: str,
                   prompt_type: PromptType | None = None,
                   **kwargs: Unpack[EncodeKwargs]) -> Array

        Everything after ``inputs`` is keyword-only, matching the base class.
        Defaults are added here so tests can call ``encode(["a", "b"])``
        directly; MTEB itself always passes them.

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


#: Attention backends tried, in order, when a checkpoint asks for one the
#: installed transformers does not accept. "sdpa" first because it is the fast
#: path; "eager" is the universally-supported fallback.
_ATTENTION_FALLBACKS: tuple[str, ...] = ("sdpa", "eager")

#: Checkpoints allowed to execute modeling code downloaded from the Hub.
#:
#: DELIBERATELY AN ALLOWLIST, NOT A FLAG. `trust_remote_code=True` runs
#: third-party Python on this machine, so it is granted per checkpoint after a
#: human decision, never as a global default that a later model swap inherits
#: silently.
#:
#: jina-embeddings-v2-base-code genuinely cannot work without it: its config
#: declares `position_embedding_type: "alibi"`, which stock BertModel does not
#: implement, and its own sentence_bert_config.json ships
#: `model_args: {"trust_remote_code": true}`. Loaded without it, transformers
#: falls back to plain BERT and you get a model that silently cannot address
#: its advertised 8192-token context - the failure this project is most
#: exposed to, given 89% of test queries overflow a 254-token window.
#: The code is fetched from the companion repo jinaai/jina-bert-v2-qk-post-norm.
_TRUST_REMOTE_CODE: frozenset[str] = frozenset({
    "jinaai/jina-embeddings-v2-base-code",
})


def _install_legacy_transformers_shims() -> list[str]:
    """Restore transformers 4.x helpers that 5.x removed, for remote code.

    WHY THIS IS HERE AND NOT A VERSION CHANGE
    -----------------------------------------
    ``jinaai/jina-embeddings-v2-base-code`` ships its own modeling code, and
    that code was written against transformers 4.x. On the pinned stack
    (transformers 5.17.0, which ``sentence-transformers==6.0.1`` resolves to)
    it dies at import::

        ImportError: cannot import name 'find_pruneable_heads_and_indices'
                     from 'transformers.pytorch_utils'

    Downgrading transformers is not an option - the versions here are pinned
    and verified, and transformers is what sentence-transformers and mteb are
    resolved against. So the missing symbol is re-supplied instead.

    WHAT IS ACTUALLY BEING RESTORED
    -------------------------------
    ``find_pruneable_heads_and_indices`` computes which attention-head slices
    survive a pruning request. It is pure, ~10 lines, and reproduced here from
    the 4.x implementation. Nothing in this project prunes heads, so on every
    normal inference path it is imported and never called - it only needs to
    exist for the module-level import to succeed.

    SCOPE
    -----
    Called only immediately before loading a checkpoint on
    ``_TRUST_REMOTE_CODE``, and it only ever ADDS a name that is absent. A
    transformers that still provides the symbol is left untouched.

    Returns
    -------
    list[str]
        Names actually injected, for logging.
    """
    import torch
    from transformers import pytorch_utils

    installed: list[str] = []

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(
            heads: Any, n_heads: int, head_size: int, already_pruned_heads: Any
        ) -> tuple[Any, Any]:
            """Verbatim behaviour of the transformers 4.x helper."""
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - set(already_pruned_heads)
            for head in heads:
                shift = sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head - shift] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )
        installed.append("find_pruneable_heads_and_indices")

    return installed


def _load_sentence_transformer(model_name: str) -> Any:
    """Load ``model_name``, working around stale ``attn_implementation`` values.

    Some checkpoints pin an attention backend in their config that a newer
    transformers no longer accepts. ``jinaai/jina-embeddings-v2-base-code``
    requests ``attn_implementation="torch"``, which transformers 5.17.0 rejects
    outright::

        ValueError: Specified `attn_implementation="torch"` is not supported.

    That is a checkpoint/library mismatch, not a dependency problem - the fix
    is to name a backend the installed library does support, NOT to move any
    pinned version. The override is applied only after the plain load has
    already failed for this specific reason, so every other model keeps
    whatever its own config asked for.

    Raises the original error if no fallback works.
    """
    from sentence_transformers import SentenceTransformer

    extra: dict[str, Any] = {}
    if model_name in _TRUST_REMOTE_CODE:
        logger.warning(
            "%s is on the remote-code allowlist: loading it EXECUTES modeling "
            "code downloaded from the Hub. See _TRUST_REMOTE_CODE.", model_name,
        )
        extra["trust_remote_code"] = True
        shimmed = _install_legacy_transformers_shims()
        if shimmed:
            logger.warning(
                "Restored transformers 4.x symbols for %s: %s. See "
                "_install_legacy_transformers_shims for why.",
                model_name, ", ".join(shimmed),
            )

    try:
        return SentenceTransformer(model_name, device=config.DEVICE, **extra)
    except ValueError as exc:
        if "attn_implementation" not in str(exc):
            raise
        original = exc

    # The rejected value lives in the checkpoint's own config.json, so it has
    # to be overridden on the CONFIG - model_kwargs loses to the already-built
    # config object that SentenceTransformer passes down.
    for backend in _ATTENTION_FALLBACKS:
        try:
            model = SentenceTransformer(
                model_name,
                device=config.DEVICE,
                config_kwargs={"attn_implementation": backend},
                **extra,
            )
        except (ValueError, TypeError) as exc:
            logger.debug("attn_implementation=%r rejected for %s (%s)",
                         backend, model_name, exc)
            continue
        logger.warning(
            "%s pins an attention backend this transformers rejects; loaded it "
            "with attn_implementation=%r instead.", model_name, backend,
        )
        return model

    raise original


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


@lru_cache(maxsize=16)
def _resolve_revision(model_name: str) -> str | None:
    """Return the Hub commit sha for ``model_name``, or None if unreachable.

    Cached and failure-tolerant: a missing revision costs us a warning and a
    less precise result path, which is never worth failing an evaluation over.
    """
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(model_name).sha
        if sha:
            logger.debug("Resolved %s revision %s", model_name, sha)
            return str(sha)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not resolve revision for %s (%s)", model_name, exc)
    return None


def _build_model_meta(model_name: str) -> Any | None:
    """Build a ``ModelMeta`` describing this model, or ``None`` if unavailable.

    MTEB uses the metadata to name the result directory and to pick the
    similarity function. Its required fields have changed across 2.x releases,
    so construction is attempted and failure is tolerated - MTEB falls back to
    deriving what it needs from the model itself.

    The revision is resolved from the Hub so results are filed under the exact
    commit that produced them. Without it MTEB warns and writes everything to
    ``no_revision_available/``, which makes two runs of "the same" model
    indistinguishable if the checkpoint is ever updated upstream.
    """
    try:
        # VERIFIED against mteb 2.20.11: ModelMeta lives in `mteb.models`, NOT
        # at the top level - `from mteb import ModelMeta` raises ImportError.
        from mteb.models import ModelMeta
    except ImportError:
        return None

    try:
        # ModelMeta is a pydantic model with 17 REQUIRED fields. Most accept
        # None, but they must all be passed explicitly - omitting any of them
        # is a validation error, not a defaulted field.
        return ModelMeta(
            loader=None,
            name=model_name,
            revision=_resolve_revision(model_name),
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["Sentence Transformers"],
            # Drives the inherited AbsEncoder.similarity implementation.
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )
    except Exception as exc:  # pydantic ValidationError is not a TypeError
        logger.debug("Could not build ModelMeta (%s); letting MTEB infer it", exc)
        return None
