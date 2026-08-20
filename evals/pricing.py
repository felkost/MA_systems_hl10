"""A pinned, dated model price table (spec Sec13.10, decision D2).

`llm.cost_usd` (`middleware.py::LlmSpanMiddleware`) is populated only from an
OpenRouter-shaped `response_metadata["token_usage"]["cost"]` field -- this
project's agents run on `LLM_PROVIDER=openai` (the default), which returns no
such field, so `llm.cost_usd` is absent on every agent span an evaluation run
produces. Cost is therefore computed here from the token counts both
providers *do* record (`llm.tokens.input`/`llm.tokens.output`), against
prices verified against each provider's own published catalogue rather than
billed.

Every price is an estimate as of the date it was verified, not a receipt --
a rate change on the provider's side goes stale here silently. A model absent
from the table returns `None` from `cost_of`, never a wrong zero: the
distinction between "we don't know" and "this was free" matters to every
report reading these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens, plus the date the price was verified."""

    input_usd_per_million: float
    output_usd_per_million: float
    verified_on: str


PRICE_TABLE: dict[str, ModelPrice] = {
    # This project's default agent model (LLM_PROVIDER=openai,
    # config.py::Settings.model_name), verified against OpenAI's published
    # API pricing page.
    "gpt-4.1-mini": ModelPrice(
        input_usd_per_million=0.40,
        output_usd_per_million=1.60,
        verified_on="2026-08-19",
    ),
    # This project's configured judge model through stage 10c's first
    # measurement (JUDGE_PROVIDER=openrouter), verified against
    # OpenRouter's own model catalogue page for prompts under its
    # 200k-token tier. Superseded by qwen/qwen3-coder below -- kept so the
    # stage-10b and early-10c figures it priced stay reproducible.
    "google/gemini-2.5-pro": ModelPrice(
        input_usd_per_million=1.25,
        output_usd_per_million=10.00,
        verified_on="2026-08-19",
    ),
    # This project's configured judge model as of 2026-08-20 (replaces
    # google/gemini-2.5-pro): google/gemini-2.5-pro is the family the
    # original spec measured with, but qwen/qwen-2.5-coder-32b-instruct --
    # the model actually requested -- does not report OpenRouter's
    # `structured_outputs` capability (checked live against the OpenRouter
    # catalogue, 2026-08-20) and is refused by
    # `models.assert_structured_output_supported` before it can be used.
    # qwen/qwen3-coder is the closest family match that does report the
    # capability, confirmed by a live judge() call against a real pack
    # before this price was recorded. Verified against OpenRouter's own
    # catalogue.
    "qwen/qwen3-coder": ModelPrice(
        input_usd_per_million=0.30,
        output_usd_per_million=1.00,
        verified_on="2026-08-20",
    ),
}


def cost_of(model: str, *, input_tokens: int, output_tokens: int) -> float | None:
    """USD cost for one call, or `None` when `model` is not in `PRICE_TABLE`.

    Parameters
    ----------
    model : str
        The exact model identifier as recorded on an `llm.*` span
        (`llm.model` attribute) -- must match a `PRICE_TABLE` key verbatim.
    input_tokens, output_tokens : int
        From the same span's `llm.tokens.input`/`llm.tokens.output`.
    """
    price = PRICE_TABLE.get(model)
    if price is None:
        return None
    return (
        input_tokens / 1_000_000 * price.input_usd_per_million
        + output_tokens / 1_000_000 * price.output_usd_per_million
    )
