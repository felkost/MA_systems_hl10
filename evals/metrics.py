"""`CaseMetrics` (spec Sec5.2): per-case latency, tokens, cost, tool-call and
delegation counts, derived purely from a persisted `EvidencePack` -- never
from a live run, so it is recomputable offline and testable against a
hand-built pack.

`tool_calls_by_name` is what makes the taxonomy's *wrong-tool-selection*
category (spec Sec5.3) checkable rather than impressionistic: a `web_search`
count on a case the corpus alone answers is a signal to go read that trace,
not a verdict by itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from evals.evidence import EvidencePack
from evals.pricing import cost_of

_ERROR_OUTCOME = "error"


@dataclass(frozen=True)
class CaseMetrics:
    """Everything a statistics pass needs about one case's run, with no
    dependency on the judge having run at all."""

    latency_ms: float
    llm_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    tool_calls: int
    tool_calls_by_name: dict[str, int]
    tool_errors: int
    delegations: int
    critic_revisions: int


def metrics_of(pack: EvidencePack) -> CaseMetrics:
    """Derive `CaseMetrics` from `pack`.

    `cost_usd` is `None` the moment any one call's model is absent from
    `evals.pricing.PRICE_TABLE` -- a partial sum that silently omits one
    call's spend would misrepresent the total more than an explicit "we
    don't know" does (D2).
    """
    tokens_in = sum(call.input_tokens for call in pack.llm_calls)
    tokens_out = sum(call.output_tokens for call in pack.llm_calls)

    call_costs = [
        cost_of(
            call.model, input_tokens=call.input_tokens, output_tokens=call.output_tokens
        )
        for call in pack.llm_calls
    ]
    cost_usd: float | None = (
        None
        if any(cost is None for cost in call_costs)
        else sum(cost for cost in call_costs if cost is not None)
    )

    tool_calls_by_name = dict(Counter(call.name for call in pack.tool_calls))
    tool_errors = sum(1 for call in pack.tool_calls if call.outcome == _ERROR_OUTCOME)

    return CaseMetrics(
        latency_ms=pack.latency_ms,
        llm_calls=len(pack.llm_calls),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        tool_calls=len(pack.tool_calls),
        tool_calls_by_name=tool_calls_by_name,
        tool_errors=tool_errors,
        delegations=len(pack.agent_calls),
        critic_revisions=pack.critic_revision_count,
    )
