"""A2A servers for all three sub-agents -- Pattern A (exactly one `Message`
per request), one Agent Card each: Planner :8903, Researcher :8904, Critic
:8905.

Reference for every A2A shape below: spec Sec5a
(docs/superpowers/specs/2026-08-14-mcp-a2a-refactor-design-en.md), measured
against the installed a2a-sdk 1.1.2 at the stage-4 kickoff --
a2a-python/samples/ ships with neither the installed package nor this
checkout, so it is not the reference.

Stage 5 extends this file with `_SubAgentExecutor`, the base every executor
shares (get input -> load tools -> build agent -> invoke -> enqueue one
Message), plus the output-shape guardrail (spec Sec9): an absent or invalid
`structured_response` becomes a typed refusal `Message`, never a partially-
rendered plan or critique. This retrofits `PlannerExecutor`, which shipped
at stage 4 without the check -- the criterion requiring it is stage 5's, not
stage 4's (author decision, stage-5 kickoff).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    # This file already lives at the project root, so `python a2a_servers.py`
    # already puts _PROJECT_ROOT on sys.path[0] -- unlike mcp_servers/*.py,
    # one directory down, which needed this fix for real (stage 2). Kept
    # here for parity with that established pattern, and so a future
    # invocation from a different location cannot silently regress.
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn  # noqa: E402
from a2a.helpers import (  # noqa: E402
    new_data_part,
    new_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.events import EventQueue  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.routes import (  # noqa: E402
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from a2a.types import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
)
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH, DEFAULT_RPC_URL  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402
from starlette.applications import Starlette  # noqa: E402

import models  # noqa: E402
from agents.critic import CRITIC_ALLOWLIST, create_critic_agent  # noqa: E402
from agents.planner import PLANNER_ALLOWLIST, create_planner_agent  # noqa: E402
from agents.research import RESEARCHER_ALLOWLIST, create_research_agent  # noqa: E402
from auth import a2a_auth_middleware, auth_headers, build_token_verifier  # noqa: E402
from config import Settings, load_settings  # noqa: E402
from mcp_utils import load_agent_tools, read_resource  # noqa: E402
from schemas import CritiqueResult  # noqa: E402
from schemas import ResearchPlan  # noqa: E402
from schemas import render_critique, render_plan  # noqa: E402

# The Agent Card's own version, not the A2A spec version (spec Sec5a).
_AGENT_CARD_VERSION = "0.1.0"

_VALID_AGENT_NAMES: tuple[str, ...] = ("planner", "researcher", "critic")

_INVALID_OUTPUT_TEMPLATE = (
    "ERROR: the {agent} returned an invalid or incomplete response and "
    "could not complete this request."
)


class UnknownAgentError(Exception):
    """`build_agent_card`/`build_app`/`main` got a name outside
    `_VALID_AGENT_NAMES`.

    Structural (spec Sec5): the fix is correcting the name, never a retry.
    """

    def __init__(self, agent: str) -> None:
        super().__init__(
            f"unknown agent {agent!r} -- expected one of {list(_VALID_AGENT_NAMES)}"
        )


class PreflightError(Exception):
    """This agent's upstream dependency is not reachable, or its resolved
    model cannot serve this role, at startup (stage 8).

    Structural (spec Sec5): the fix is starting the dependency or changing
    the model, never a retry -- same message convention as `main.py`'s own
    `PreflightError`, a distinct class (each interface module owns its
    structural errors rather than importing another interface file's, the
    same reasoning `UnknownAgentError` already follows).
    """


# Only the two roles whose executor validates a `structured_response`
# (PlannerExecutor, CriticExecutor) -- ResearcherExecutor has none
# (agents/research.py carries no response_format), so checking it would
# always be a wasted network round-trip.
_STRUCTURED_OUTPUT_AGENTS = frozenset({"planner", "critic"})


async def _preflight(settings: Settings, agent_name: str) -> None:
    """This process's own startup checks (stage 8), scoped to what it alone
    depends on -- distinct from `main.py`'s five-server preflight.

    Raises
    ------
    PreflightError
        SearchMCP is unreachable, or (for planner/critic only) the
        resolved model cannot serve this role's structured output.
    """
    try:
        await read_resource(
            url=settings.search_mcp_url(),
            uri="resource://knowledge-base-stats",
            headers=auth_headers(settings),
        )
    except Exception as error:
        raise PreflightError(
            "SearchMCP is not reachable -- start it with "
            "`python mcp_servers/search_mcp.py`"
        ) from error

    if agent_name in _STRUCTURED_OUTPUT_AGENTS:
        try:
            await models.assert_structured_output_supported(settings, agent_name)
        except models.UnsupportedStructuredOutputError as error:
            raise PreflightError(str(error)) from error


class _SubAgentExecutor(AgentExecutor):
    """Per-request body shared by all three agents (spec Sec5a): read input,
    load tools, build an agent, invoke, enqueue **exactly one** `Message`
    (Pattern A -- mixing in the Task lifecycle raises
    `InvalidAgentResponseError`).

    Stateless (spec Sec8): a fresh agent is built and discarded every call --
    no checkpointer, and (with one server per agent) no shared client across
    requests either. Subclasses supply `allowlist`, `_build_agent` and
    `_to_message`; the output-shape guardrail lives in `_to_message` via the
    shared `_validated_structured_response` helper.
    """

    allowlist: tuple[str, ...]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request = context.get_user_input()
        settings = load_settings()
        tools = await load_agent_tools(settings, self.allowlist)
        agent = self._build_agent(settings, tools)
        result = await agent.ainvoke({"messages": [HumanMessage(request)]})
        await event_queue.enqueue_event(self._to_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Pattern A never creates a Task, so there is nothing long-running to
        # cancel (spec Sec5a).
        raise NotImplementedError(f"{type(self).__name__} has no cancellable task")

    def _build_agent(self, settings: Settings, tools: list[BaseTool]) -> Any:
        raise NotImplementedError

    def _to_message(self, result: dict[str, Any]) -> Message:
        raise NotImplementedError

    @staticmethod
    def _validated_structured_response(
        result: dict[str, Any], schema: type[Any]
    ) -> Any | None:
        """`result["structured_response"]` if it is a valid instance of
        `schema`, else `None` -- the output-shape guardrail's one check,
        shared by the Planner and the Critic."""
        structured = result.get("structured_response")
        return structured if isinstance(structured, schema) else None


class PlannerExecutor(_SubAgentExecutor):
    """One Planner run per A2A request."""

    allowlist = PLANNER_ALLOWLIST

    def _build_agent(self, settings: Settings, tools: list[BaseTool]) -> Any:
        return create_planner_agent(settings, tools)

    def _to_message(self, result: dict[str, Any]) -> Message:
        plan = self._validated_structured_response(result, ResearchPlan)
        if plan is None:
            return new_text_message(_INVALID_OUTPUT_TEMPLATE.format(agent="Planner"))
        return new_text_message(render_plan(plan))


class ResearcherExecutor(_SubAgentExecutor):
    """One Researcher run per A2A request.

    No `structured_response` to validate -- the guardrail here is that the
    final message carries non-empty findings.
    """

    allowlist = RESEARCHER_ALLOWLIST

    def _build_agent(self, settings: Settings, tools: list[BaseTool]) -> Any:
        return create_research_agent(settings, tools)

    def _to_message(self, result: dict[str, Any]) -> Message:
        content = str(result["messages"][-1].content).strip()
        if not content:
            return new_text_message(_INVALID_OUTPUT_TEMPLATE.format(agent="Researcher"))
        return new_text_message(content)


class CriticExecutor(_SubAgentExecutor):
    """One Critic run per A2A request.

    **The one deliberate deviation from the brief** (spec Sec9): the verdict
    travels as a machine-readable data part alongside the rendered markdown,
    via `new_message` from the same helper family `new_text_message` belongs
    to -- not hand-rolled Protobuf. Following the brief's `new_text_message`
    literally here would force the Supervisor to parse a verdict out of
    prose, the exact failure hl8 measured (4 of 18 runs ended without saving
    after an APPROVE).
    """

    allowlist = CRITIC_ALLOWLIST

    def _build_agent(self, settings: Settings, tools: list[BaseTool]) -> Any:
        return create_critic_agent(settings, tools)

    def _to_message(self, result: dict[str, Any]) -> Message:
        critique = self._validated_structured_response(result, CritiqueResult)
        if critique is None:
            return new_text_message(_INVALID_OUTPUT_TEMPLATE.format(agent="Critic"))
        return new_message(
            [
                new_text_part(render_critique(critique)),
                new_data_part(critique.model_dump()),
            ]
        )


_EXECUTORS: dict[str, type[_SubAgentExecutor]] = {
    "planner": PlannerExecutor,
    "researcher": ResearcherExecutor,
    "critic": CriticExecutor,
}


def _build_agent_card(
    *, port: int, name: str, description: str, skill: AgentSkill
) -> AgentCard:
    """Shared shape for the three Agent Cards.

    No top-level `url` in A2A spec v1.0 (measured, spec Sec5a): the address
    lives in `supported_interfaces` instead, so this is not an oversight.
    """
    interface = AgentInterface(
        url=f"http://127.0.0.1:{port}",
        protocol_binding="JSONRPC",
        protocol_version="1.0",
    )
    return AgentCard(
        name=name,
        description=description,
        version=_AGENT_CARD_VERSION,
        supported_interfaces=[interface],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[skill],
    )


def build_planner_agent_card(settings: Settings) -> AgentCard:
    skill = AgentSkill(
        id="research-planning",
        name="Research planning",
        description=(
            "Turns a user's research request into a structured research "
            "plan: a goal, concrete search queries, which sources to check, "
            "and the expected output format."
        ),
        tags=["planning", "research"],
        examples=["Compare RAG approaches: naive, sentence-window, and parent-child"],
    )
    return _build_agent_card(
        port=settings.planner_a2a_port,
        name="Planner",
        description=(
            "Decomposes a research request into a structured plan before "
            "any research happens."
        ),
        skill=skill,
    )


def build_researcher_agent_card(settings: Settings) -> AgentCard:
    skill = AgentSkill(
        id="research-execution",
        name="Research execution",
        description=(
            "Executes a research plan against the knowledge base and the "
            "web, and reports findings as cited Markdown."
        ),
        tags=["research", "search"],
        examples=["Find sources comparing naive and sentence-window RAG"],
    )
    return _build_agent_card(
        port=settings.researcher_a2a_port,
        name="Researcher",
        description=(
            "Gathers findings for a research plan or a revision request, "
            "citing every source."
        ),
        skill=skill,
    )


def build_critic_agent_card(settings: Settings) -> AgentCard:
    skill = AgentSkill(
        id="research-critique",
        name="Research critique",
        description=(
            "Independently verifies research findings against sources and "
            "returns a structured verdict: APPROVE or REVISE."
        ),
        tags=["critique", "verification"],
        examples=["Critique these findings on RAG retrieval strategies"],
    )
    return _build_agent_card(
        port=settings.critic_a2a_port,
        name="Critic",
        description=(
            "Verifies research findings for freshness, completeness and "
            "structure before a report is written."
        ),
        skill=skill,
    )


_AGENT_CARD_BUILDERS: dict[str, Any] = {
    "planner": build_planner_agent_card,
    "researcher": build_researcher_agent_card,
    "critic": build_critic_agent_card,
}

_PORT_ATTRS: dict[str, str] = {
    "planner": "planner_a2a_port",
    "researcher": "researcher_a2a_port",
    "critic": "critic_a2a_port",
}


def build_agent_card(settings: Settings, agent: str = "planner") -> AgentCard:
    """The named agent's published Agent Card, served at
    `/.well-known/agent-card.json`.

    Parameters
    ----------
    settings : Settings
    agent : str, default "planner"
        One of `"planner"`, `"researcher"`, `"critic"`.

    Raises
    ------
    UnknownAgentError
        `agent` names none of the three.
    """
    try:
        builder = _AGENT_CARD_BUILDERS[agent]
    except KeyError:
        raise UnknownAgentError(agent) from None
    return builder(settings)


def build_app(settings: Settings, agent: str = "planner") -> Starlette:
    """Assemble the Starlette app serving both the JSON-RPC endpoint and the
    Agent Card for `agent`. `A2AStarletteApplication` does not exist in
    a2a-sdk 1.1.2 (measured, spec Sec5a) -- route assembly replaces it.

    Raises
    ------
    UnknownAgentError
        `agent` names none of `"planner"`, `"researcher"`, `"critic"`.
    """
    if agent not in _EXECUTORS:
        raise UnknownAgentError(agent)
    agent_card = build_agent_card(settings, agent)
    # InMemoryTaskStore is a required DefaultRequestHandler argument
    # (measured, spec Sec5a) even though Pattern A never creates a Task --
    # an SDK requirement, not a retreat from spec Sec8's "no persistent
    # TaskStore". The store stays in memory and stays empty. No module-level
    # client, store, settings or agent anywhere in this file: one process
    # now serves three agents, which is exactly where cross-agent shared
    # state would appear (spec Sec6 rule 2).
    handler = DefaultRequestHandler(
        agent_executor=_EXECUTORS[agent](),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = create_jsonrpc_routes(handler, DEFAULT_RPC_URL) + create_agent_card_routes(
        agent_card, card_url=AGENT_CARD_WELL_KNOWN_PATH
    )
    verifier = build_token_verifier(settings)
    return Starlette(routes=routes, middleware=a2a_auth_middleware(verifier))


def main(agent_name: str = "planner") -> None:
    """Boot one A2A server. `python a2a_servers.py [planner|researcher|critic]`
    -- the bare invocation (no argument) keeps serving the Planner, so the
    stage-4 bootstrap regression test is unaffected. Stage 8's `servers.py`
    launches all three by name through this same contract.

    Refuses before ever binding a port if this agent's own preflight fails
    (`_preflight`) -- `sys.exit(1)` is what lets `servers.py`'s readiness
    wait detect a failed process quickly via `poll()`, instead of waiting
    out the full readiness timeout.
    """
    if agent_name not in _EXECUTORS:
        raise UnknownAgentError(agent_name)
    settings = load_settings()
    try:
        asyncio.run(_preflight(settings, agent_name))
    except PreflightError as error:
        print(f"[system] {error}")
        sys.exit(1)
    port = getattr(settings, _PORT_ATTRS[agent_name])
    uvicorn.run(build_app(settings, agent_name), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "planner")
