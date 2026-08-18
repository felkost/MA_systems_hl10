"""Researcher sub-agent: executes a plan and reports findings as free-text
Markdown with inline citations -- it does not save anything, and it has no
`response_format`. Findings are prose for the Critic and the Supervisor to
read; a plan or a verdict is data the code branches on, which is the
deliberate asymmetry with the Planner and the Critic.

Same shape as `agents/planner.py`: tools are a factory *parameter* (they
arrive over the network and cannot be imported), the model comes from
`models.build_chat_model(settings, "researcher")`, and the prompt is a plain
`config` constant. Stateless per invocation: no checkpointer, one human
message in, one findings message out.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

import models
from config import RESEARCHER_PROMPT, Settings
from middleware import LlmSpanMiddleware, ReadUrlCapMiddleware, a2a_agent_middleware

RESEARCHER_ALLOWLIST: tuple[str, ...] = ("web_search", "read_url", "knowledge_search")


def create_research_agent(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[
    AgentState[None], None, InputAgentState, OutputAgentState[None]
]:
    """Build the Researcher sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Loaded by the caller, e.g. `mcp_utils.load_agent_tools(settings,
        RESEARCHER_ALLOWLIST)`. Bound exactly as given -- this factory never
        imports a tool list of its own.
    model : BaseChatModel, optional
        Chat model to use instead of the one `settings` resolves. Tests
        inject a scripted fake here instead of reaching a real provider.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the findings as free-text
        Markdown in the last message -- there is no `structured_response`.
    """
    chat_model = model or models.build_chat_model(settings, "researcher")

    agent_middleware: list[AgentMiddleware[AgentState[None], None, None]] = [
        *a2a_agent_middleware(tool_call_limit=settings.researcher_max_tool_calls),
        ReadUrlCapMiddleware(settings.max_read_url_per_search),
        # Innermost (stage 9, spec Sec16): see agents/planner.py.
        LlmSpanMiddleware("researcher", *settings.resolved("researcher")),
    ]

    return create_agent(
        model=chat_model,
        tools=list(tools),
        system_prompt=RESEARCHER_PROMPT,
        middleware=agent_middleware,
    )
