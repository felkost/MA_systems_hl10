"""Model provider selection (spec Sec 16): construct the LangChain objects a
resolved (provider, model) pair points to.

`config.py` owns the resolution formula (`Settings.resolved`) because it
sits a layer below this module (kernel, versus this module's infra) and
therefore cannot depend on it; this module is the one place that turns a
resolved pair into a real client. The model-capability preflight helper the
architecture table also assigns here is not written yet -- it needs a
network call and belongs to stage 8.

`langchain_huggingface` is imported inside `build_embeddings`, not at module
scope: importing it eagerly pulls in `sentence_transformers`, the same cost
`retriever.py` already deferred for the cross-encoder -- confirmed by
`tests/test_retriever.py::test_importing_retriever_does_not_load_sentence_transformers`,
which broke the first time this import sat at the top of this file, because
`retriever.py` imports `ingest`, which imports this module.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import paths
from config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_model(settings: Settings, role: str) -> tuple[str, str]:
    """(provider, model) for one role -- thin wrapper over `Settings.resolved`."""
    return settings.resolved(role)


def resolved_map(settings: Settings) -> dict[str, tuple[str, str]]:
    """{role: (provider, model)} for every role in `Settings.ROLES`.

    Returns
    -------
    dict
        The map the stage-6 startup banner prints and stage-9 spans / eval
        metadata carry, so a run is attributable to one exact configuration.
    """
    return {role: resolve_model(settings, role) for role in Settings.ROLES}


def build_chat_model(settings: Settings, role: str) -> BaseChatModel:
    """Build the chat model `role` resolves to.

    Parameters
    ----------
    settings : Settings
    role : str
        One of `Settings.ROLES`.

    Returns
    -------
    BaseChatModel
        `ChatOpenAI` pointed at OpenAI or OpenRouter -- OpenRouter is
        OpenAI-API-compatible and differs only in `base_url` and the key, so
        no second chat-model class exists. `temperature` is passed only when
        set: some models routed through OpenRouter do not accept it.
    """
    provider, model = settings.resolved(role)
    key = (
        settings.openai_api_key if provider == "openai" else settings.openrouter_api_key
    )
    assert (
        key is not None
    )  # Settings._keys_present_for_resolved_providers guarantees this

    kwargs: dict[str, Any] = {"model": model, "api_key": key.get_secret_value()}
    if provider == "openrouter":
        kwargs["base_url"] = OPENROUTER_BASE_URL
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    return ChatOpenAI(**kwargs)


def build_embeddings(settings: Settings) -> Embeddings:
    """Build the embedder `settings.embedding_provider` selects.

    Returns
    -------
    Embeddings
        `OpenAIEmbeddings` for `"openai"`; otherwise `HuggingFaceEmbeddings`
        (`langchain_huggingface`), cached under `settings.model_cache_dir`.

    Notes
    -----
    OpenRouter has no embeddings endpoint -- this is the fact
    `embedding_provider` exists to work around (spec Sec 16).
    """
    if settings.embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        cache = paths.resolve(settings.model_cache_dir)
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            cache_folder=str(cache / "huggingface"),
        )

    assert settings.openai_api_key is not None  # guaranteed by Settings validator
    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key.get_secret_value(),
    }
    if settings.embedding_dimensions is not None:
        kwargs["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbeddings(**kwargs)


def embedding_fingerprint(settings: Settings) -> dict[str, Any]:
    """{"provider", "model", "dimensions"} -- written into `manifest.json` by
    `ingest.py` and recomputed by `retriever.py` to compare."""
    return {
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }
