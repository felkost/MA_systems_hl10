"""Model provider selection (spec Sec 16): construct the LangChain objects a
resolved (provider, model) pair points to.

`config.py` owns the resolution formula (`Settings.resolved`) because it
sits a layer below this module (kernel, versus this module's infra) and
therefore cannot depend on it; this module is the one place that turns a
resolved pair into a real client. `assert_structured_output_supported`
(stage 8, spec Sec16) is the model-capability preflight helper the
architecture table assigns here -- one request to OpenRouter's model
listing, only for the two roles that actually need strict structured
output (`agents/planner.py`, `agents/critic.py`; `agents/research.py`
explicitly carries none).

`langchain_huggingface` is imported inside `build_embeddings`, not at module
scope: importing it eagerly pulls in `sentence_transformers`, the same cost
`retriever.py` already deferred for the cross-encoder -- confirmed by
`tests/test_retriever.py::test_importing_retriever_does_not_load_sentence_transformers`,
which broke the first time this import sat at the top of this file, because
`retriever.py` imports `ingest`, which imports this module.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import paths
from config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The roles whose caller builds a strict structured-output request:
# "planner"/"critic" via ProviderStrategy(..., strict=True)
# (agents/planner.py:53, agents/critic.py:44, both unconditional
# module-level constants) -- hardcoded rather than introspected from
# agents/*.py, since this module sits below agents/ in the layer table and
# must not import upward. "judge" (stage 10a, decision D1) joins the same
# way: evals/judge.py builds its own strict structured request, and without
# this the preflight refusal a misconfigured OpenRouter judge model needs
# would happen mid-dataset-run instead of at startup.
_STRUCTURED_OUTPUT_ROLES = ("planner", "critic", "judge")


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


class UnsupportedStructuredOutputError(Exception):
    """`role` needs strict structured output but the resolved OpenRouter
    model does not report support for it -- named explicitly (spec Sec16),
    never a silent downgrade to best-effort parsing."""


async def assert_structured_output_supported(settings: Settings, role: str) -> None:
    """Refuse if `role` cannot actually get the structured output its agent
    requires from the model it resolves to (spec Sec16 preflight).

    No-op outside `{"planner", "critic", "judge"}` -- the only roles whose
    caller builds a strict structured-output request. No-op for provider
    `"openai"`: strict JSON-schema mode is a stable capability of every
    OpenAI model family this project resolves to, with no public per-model
    capability listing the way OpenRouter's is. For provider `"openrouter"`,
    fetches the model catalog once and refuses, naming the model, if it is
    absent or does not report `"structured_outputs"` support.

    Raises
    ------
    UnsupportedStructuredOutputError
    """
    if role not in _STRUCTURED_OUTPUT_ROLES:
        return
    provider, model = settings.resolved(role)
    if provider != "openrouter":
        return

    catalog = await _fetch_openrouter_models(settings)
    entry = catalog.get(model)
    if entry is None:
        raise UnsupportedStructuredOutputError(
            f"role {role!r} resolves to OpenRouter model {model!r}, which "
            f"does not appear in OpenRouter's model listing -- check "
            f"{role.upper()}_MODEL_NAME"
        )
    if not _supports_structured_output(entry):
        raise UnsupportedStructuredOutputError(
            f"role {role!r} resolves to OpenRouter model {model!r}, which "
            f"does not report support for strict structured output -- pick "
            f"a different {role.upper()}_MODEL_NAME"
        )


async def _fetch_openrouter_models(settings: Settings) -> dict[str, dict[str, Any]]:
    """`{model_id: entry}` for OpenRouter's full catalog -- one request, no
    auth required for the listing itself (verified live, 2026-08-18)."""
    headers: dict[str, str] = {}
    if settings.openrouter_api_key is not None:
        headers["Authorization"] = (
            f"Bearer {settings.openrouter_api_key.get_secret_value()}"
        )
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers)
        response.raise_for_status()
    return {entry["id"]: entry for entry in response.json()["data"]}


def _supports_structured_output(entry: dict[str, Any]) -> bool:
    """`"structured_outputs"` only -- confirmed against the live catalog
    (2026-08-18, 414 entries): `"response_format"` is a *looser* JSON-mode
    capability present on 30 models that do NOT also carry
    `"structured_outputs"` (e.g. qwen/qwen3.7-flash) -- ORing the two would
    silently accept a model that cannot honour `strict=True`."""
    return "structured_outputs" in entry.get("supported_parameters", [])
