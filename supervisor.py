"""The Supervisor: a local `create_agent` whose three tools are thin async
wrappers over `a2a.client` calls to the three A2A sub-agents (stage 6).

Reference for every A2A shape below: spec Sec5a, measured against the
installed a2a-sdk 1.1.2 -- re-verified at this stage's kickoff via a
nine-point client-path audit (2026-08-17).

**Layering (spec Sec6 rule 2):** this module never imports `a2a_servers` or
`agents/*`. Every delegation crosses a real network hop -- importing an
agent factory directly would work perfectly and violate the assignment's
central requirement invisibly.

**D1, the four-class error taxonomy applied for real:** only a genuine
policy failure (`SubAgentResponseError`) is caught inside a delegation tool.
Transport exceptions from `httpx`/`a2a.client` propagate uncaught, reaching
`ToolRetryMiddleware` -> `ToolErrorMiddleware` exactly as MCP tool failures
already do -- the donor's blanket `except Exception` is not ported.

**D2 expired at stage 7:** `save_report` is now a fourth bound tool, gated
by `HumanInTheLoopMiddleware` and nudged by `SaveReportGuardMiddleware` --
both in the same commit as the binding (a commit that binds without gating
would leave an ungated write path in history).

**D7:** the three delegation tools are module-level `@tool` objects, so they
read `load_settings()` themselves rather than closing over
`create_supervisor`'s `settings` argument -- the same precedent
`_SubAgentExecutor.execute` already sets per A2A request
(`a2a_servers.py:112`). A test needing a different port patches
`supervisor.load_settings`, never passes a `Settings`.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, get_text_parts, new_text_message
from a2a.types import Message, Role, SendMessageRequest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    HumanInTheLoopMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

import observability
from auth import auth_headers
from config import (
    SUPERVISOR_PROMPT,
    SUPERVISOR_SAVE_REPORT_RULE,
    Settings,
    load_settings,
)
from middleware import (
    LlmSpanMiddleware,
    RoundStabilityMiddleware,
    SaveReportGuardMiddleware,
    _tool_error_to_message,
)
from models import build_chat_model
from schemas import CRITIQUE_INPUT_TEMPLATE, RESEARCH_INPUT_TEMPLATE

_CLIENT_TIMEOUT_SECONDS = 120.0  # stage 4/5 measured ~115.84 s per delegation


class SupervisorState(AgentState):
    """Extends `AgentState` with three fields, each with an immediate
    consumer, all written only by `delegate_to_critic` via
    `Command(update=...)`: `critic_gaps`/`previous_critic_gaps` (D3, read by
    `RoundStabilityMiddleware`) and `verdict` (stage 7, read by
    `SaveReportGuardMiddleware` -- the machine-readable decision point spec
    Sec9 requires, never parsed from rendered prose)."""

    critic_gaps: list[str] | None
    previous_critic_gaps: list[str] | None
    verdict: Literal["APPROVE", "REVISE"] | None


class SubAgentResponseError(Exception):
    """Policy-class (spec Sec5): a sub-agent's A2A response had the wrong
    shape for this delegation. Never raised for a transport failure -- those
    propagate uncaught, for `ToolRetryMiddleware` to see."""


def _current_request(messages: list[BaseMessage]) -> str:
    """The text of the most recent `HumanMessage` in `messages` (D0).

    Parameters
    ----------
    messages : list of BaseMessage

    Returns
    -------
    str

    Notes
    -----
    The donor's `_first_human_text` returns the first `HumanMessage` ever
    seen. hl10's REPL reuses one `thread_id` across a whole multi-question
    session, so a second question's delegation must forward the request that
    started *this* run, not the session's opening one -- the same "since the
    most recent `HumanMessage`" scoping `middleware._run_tool_call_ids`
    already applies to budget counters.
    """
    request = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            request = str(message.content)
    return request


async def _send_and_get_message(
    url: str, text: str, *, headers: dict[str, str] | None = None
) -> Message:
    """Send `text` to the agent at `url` and return its reply `Message`.

    Parameters
    ----------
    url : str
        A bare A2A origin, e.g. `settings.planner_a2a_url()`.
    text : str
        The message text, already rendered through the caller's template.
    headers : dict of str to str, optional
        Extra request headers, e.g. `auth.auth_headers(settings)` (stage 8).
        Omitted (`None`, the default) it stays absent -- the same
        settings-agnostic shape `mcp_utils.read_resource`'s own `headers`
        parameter uses.

    Returns
    -------
    Message

    Raises
    ------
    SubAgentResponseError
        The response stream was empty, or its one item carries no `Message`
        (audit G: all three cards set `capabilities.streaming=False`, so
        `send_message` downgrades to non-streaming and yields exactly one
        item -- reading the first and returning is safe).

    Notes
    -----
    `role=Role.ROLE_USER` is explicit (probe E): `new_text_message`'s default
    is `ROLE_AGENT`, correct for every prior, server-side use in this
    codebase, wrong for this module's first client-side call. The externally
    supplied `httpx.AsyncClient` is closed on both the success and the error
    path -- `Client.close()` really does call `httpx_client.aclose()`
    (audit G), so this is a real leak check, not a formality.
    """
    async with await create_client(
        url,
        client_config=ClientConfig(
            httpx_client=httpx.AsyncClient(
                timeout=_CLIENT_TIMEOUT_SECONDS, headers=headers
            )
        ),
    ) as client:
        request = SendMessageRequest(
            message=new_text_message(text, role=Role.ROLE_USER)
        )
        async for response in client.send_message(request):
            if not response.HasField("message"):
                raise SubAgentResponseError(f"no Message in the response from {url}")
            return response.message
    raise SubAgentResponseError(f"empty response stream from {url}")


@tool
async def delegate_to_planner(runtime: ToolRuntime) -> str:
    """Decompose the user's research request into a structured plan.
    Call this first, before any research or critique."""
    tracer = observability.get_tracer(__name__)
    with tracer.start_as_current_span("delegate_to_planner"):
        settings = load_settings()
        request = _current_request(runtime.state["messages"])
        try:
            message = await _send_and_get_message(
                settings.planner_a2a_url(), request, headers=auth_headers(settings)
            )
        except SubAgentResponseError as error:
            return f"ERROR: Planner delegation failed: {error}"
        return "\n".join(get_text_parts(message.parts))


@tool
async def delegate_to_researcher(task: str, runtime: ToolRuntime) -> str:
    """Execute one round of research. `task` is the plan on the first call,
    or the Critic's revision feedback on a later round."""
    tracer = observability.get_tracer(__name__)
    with tracer.start_as_current_span("delegate_to_researcher"):
        settings = load_settings()
        request = _current_request(runtime.state["messages"])
        rendered = RESEARCH_INPUT_TEMPLATE.format(request=request, task=task)
        try:
            message = await _send_and_get_message(
                settings.researcher_a2a_url(), rendered, headers=auth_headers(settings)
            )
        except SubAgentResponseError as error:
            return f"ERROR: Researcher delegation failed: {error}"
        return "\n".join(get_text_parts(message.parts))


@tool
async def delegate_to_critic(findings: str, runtime: ToolRuntime) -> str | Command[Any]:
    """Get an independent verdict on the current research findings."""
    tracer = observability.get_tracer(__name__)
    with tracer.start_as_current_span("delegate_to_critic") as span:
        settings = load_settings()
        request = _current_request(runtime.state["messages"])
        rendered = CRITIQUE_INPUT_TEMPLATE.format(request=request, findings=findings)
        try:
            message = await _send_and_get_message(
                settings.critic_a2a_url(), rendered, headers=auth_headers(settings)
            )
        except SubAgentResponseError as error:
            return f"ERROR: Critic delegation failed: {error}"

        data_parts = get_data_parts(message.parts)
        if not data_parts:
            return "ERROR: Critic delegation failed: no data part in the response"

        critique = data_parts[0]
        gaps = list(critique.get("gaps", []))
        verdict = critique.get("verdict")
        # No automatic instrumentation can know this -- it is the machine-
        # readable decision point spec Sec9 requires, crossing A2A as data.
        span.set_attribute("a2a.verdict", verdict or "")
        span.set_attribute("a2a.gaps", gaps)
        rendered_text = "\n".join(get_text_parts(message.parts))
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=rendered_text,
                        tool_call_id=runtime.tool_call_id,
                        name="delegate_to_critic",
                    )
                ],
                "critic_gaps": gaps,
                "previous_critic_gaps": runtime.state.get("critic_gaps"),
                "verdict": verdict,
            }
        )


def create_supervisor(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any],
    save_report_tool: BaseTool,
) -> Any:
    """Build the Supervisor agent: four tools (stage 7 binds `save_report`),
    seven middleware entries in spec order, a durable checkpoint.

    Parameters
    ----------
    settings : Settings, optional
        Used to size the two `ToolCallLimitMiddleware` budgets and, unless
        `model` is given, to build the chat model. **Not** consulted by the
        delegation tools themselves (D7) -- they call `load_settings()`.
    model : BaseChatModel, optional
        Overrides the model `models.build_chat_model` would otherwise build.
    checkpointer : BaseCheckpointSaver
        Required (D1): the write-path-free, checkpointer-free configuration
        was stage 6's artefact, not a supported mode. `main.py` owns its
        lifetime -- `AsyncSqliteSaver.from_conn_string` is an
        `asynccontextmanager`, so this function never constructs one itself.
    save_report_tool : BaseTool
        Required (D1): the ReportMCP tool, loaded by `main.py` via
        `mcp_utils.load_report_tools` before this call.

    Returns
    -------
    CompiledStateGraph
    """
    settings = settings or load_settings()
    chat_model = model or build_chat_model(settings, "supervisor")
    prompt = SUPERVISOR_PROMPT.format(
        extra_tools=", save_report", final_step_rule=SUPERVISOR_SAVE_REPORT_RULE
    )
    supervisor_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ToolCallLimitMiddleware(
            tool_name="delegate_to_researcher",
            run_limit=1 + settings.max_revisions,
            exit_behavior="continue",
        ),
        ToolCallLimitMiddleware(
            tool_name="delegate_to_critic",
            run_limit=1 + settings.max_revisions,
            exit_behavior="continue",
        ),
        ToolErrorMiddleware(_tool_error_to_message),
        ToolRetryMiddleware(on_failure="error", jitter=True),
        RoundStabilityMiddleware(),
        SaveReportGuardMiddleware(),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_report": {"allowed_decisions": ["approve", "edit", "reject"]}
            }
        ),
        # Innermost (stage 9, spec Sec16): one span per actual provider
        # request, so a `ToolRetryMiddleware`/`ModelRetryMiddleware`-driven
        # retry gets its own span and its own cost rather than one span
        # silently covering a whole retry storm.
        LlmSpanMiddleware("supervisor", *settings.resolved("supervisor")),
    ]
    return create_agent(
        model=chat_model,
        tools=[
            delegate_to_planner,
            delegate_to_researcher,
            delegate_to_critic,
            save_report_tool,
        ],
        system_prompt=prompt,
        state_schema=SupervisorState,
        middleware=supervisor_middleware,
        checkpointer=checkpointer,
    )
