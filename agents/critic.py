"""Critic sub-agent: independently verifies research findings and returns a
structured verdict. Bound to the same three tools as the Researcher, so the
Critic checks facts through the same sources rather than only reading the
Researcher's own text. `CriticVerificationMiddleware` forces at least one
verification call before a verdict is accepted.

Same shape as `agents/planner.py`: tools are a factory *parameter*, the model
comes from `models.build_chat_model(settings, "critic")`, and the prompt is a
plain `config` constant -- formatted with today's date at factory time, since
`CRITIC_PROMPT` carries a `{today}` placeholder and freshness is one of the
three dimensions the Critic judges. Stateless per invocation: no
checkpointer, one human message in, one `CritiqueResult` out.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

import models
from config import CRITIC_PROMPT, Settings
from middleware import (
    CriticVerificationMiddleware,
    LlmSpanMiddleware,
    a2a_agent_middleware,
)
from schemas import CritiqueResult

CRITIC_ALLOWLIST: tuple[str, ...] = ("web_search", "read_url", "knowledge_search")

# Passing the bare schema lets the framework auto-detect a strategy and
# build it *without* strict, which leaves the provider free to treat the
# schema's `required` list as a hint and return a verdict missing `verdict`
# itself -- the exact failure hl8 measured. Naming the strategy explicitly is
# what puts "strict": true on the wire.
CRITIC_RESPONSE_FORMAT = ProviderStrategy(CritiqueResult, strict=True)


def create_critic_agent(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[
    AgentState[CritiqueResult], None, InputAgentState, OutputAgentState[CritiqueResult]
]:
    """Build the Critic sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Loaded by the caller, e.g. `mcp_utils.load_agent_tools(settings,
        CRITIC_ALLOWLIST)`. Bound exactly as given -- this factory never
        imports a tool list of its own.
    model : BaseChatModel, optional
        Chat model to use instead of the one `settings` resolves. Tests
        inject a scripted fake here instead of reaching a real provider.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the parsed `CritiqueResult` in
        `result["structured_response"]`.
    """
    chat_model = model or models.build_chat_model(settings, "critic")

    agent_middleware: list[AgentMiddleware[AgentState[CritiqueResult], None, None]] = [
        *a2a_agent_middleware(tool_call_limit=settings.critic_max_tool_calls),
        CriticVerificationMiddleware(),
        # Innermost (stage 9, spec Sec16): see agents/planner.py.
        LlmSpanMiddleware("critic", *settings.resolved("critic")),
    ]

    return create_agent(
        model=chat_model,
        tools=list(tools),
        system_prompt=CRITIC_PROMPT.format(today=date.today().isoformat()),
        response_format=CRITIC_RESPONSE_FORMAT,
        middleware=agent_middleware,
    )
