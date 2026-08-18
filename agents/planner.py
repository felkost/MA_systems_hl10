"""Planner sub-agent: turns a user request into a structured `ResearchPlan`.

Bound to `web_search` and `knowledge_search` only -- reconnaissance to
understand the domain, not a deep read of any one source. Stateless per
invocation: no checkpointer, one human message in, one `ResearchPlan` out.

Ported from the hl8 donor (`../MA_systems_hl8_project/MA_system_hl8/agents/
planner.py`), with tools as a factory *parameter* rather than a module-level
import -- they arrive over the network at request time and cannot be
imported (spec Sec2, "Rebuild"). The prompt-version registry is gone
(`config.PLANNER_PROMPT`, a plain constant), and the model is built through
`models.build_chat_model(settings, "planner")` rather than an inline
`ChatOpenAI` (spec Sec16).

**Retrofitted at stage 5** from a bare `ToolCallLimitMiddleware` to the full
spec Sec10 stack (`middleware.a2a_agent_middleware`): the Planner reaches
SearchMCP over the same network as the Researcher and the Critic, and leaving
it without retries or a model-call backstop would be an asymmetry with no
reason behind it (stage-5 kickoff, author decision).
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

import models
from config import PLANNER_PROMPT, Settings
from middleware import LlmSpanMiddleware, a2a_agent_middleware
from schemas import ResearchPlan

# A small, fixed reconnaissance budget -- not a Settings field, since the
# experiments this project varies (stage 11) tune the Researcher and Critic
# budgets, not how much the Planner may look around before it decomposes.
PLANNER_TOOL_CALL_LIMIT = 4

PLANNER_ALLOWLIST: tuple[str, ...] = ("web_search", "knowledge_search")

# Passing the bare schema lets the framework auto-detect a strategy and
# build it *without* strict, which leaves the provider free to treat the
# schema's `required` list as a hint and return a plan missing a field.
# Naming the strategy explicitly is what puts "strict": true on the wire.
PLANNER_RESPONSE_FORMAT = ProviderStrategy(ResearchPlan, strict=True)


def create_planner_agent(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[
    AgentState[ResearchPlan], None, InputAgentState, OutputAgentState[ResearchPlan]
]:
    """Build the Planner sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Loaded by the caller, e.g. `mcp_utils.load_agent_tools(settings,
        PLANNER_ALLOWLIST)`. Bound exactly as given -- this factory never
        imports a tool list of its own.
    model : BaseChatModel, optional
        Chat model to use instead of the one `settings` resolves. Tests
        inject a scripted fake here instead of reaching a real provider.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the parsed `ResearchPlan` in
        `result["structured_response"]`.
    """
    chat_model = model or models.build_chat_model(settings, "planner")

    return create_agent(
        model=chat_model,
        tools=list(tools),
        system_prompt=PLANNER_PROMPT,
        response_format=PLANNER_RESPONSE_FORMAT,
        middleware=[
            *a2a_agent_middleware(tool_call_limit=PLANNER_TOOL_CALL_LIMIT),
            # Innermost (stage 9, spec Sec16): one span per actual provider
            # request, so ModelRetryMiddleware's own retries each get their
            # own span and cost.
            LlmSpanMiddleware("planner", *settings.resolved("planner")),
        ],
    )
