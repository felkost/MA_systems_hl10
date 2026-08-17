"""A2A server for the Planner sub-agent -- Pattern A (exactly one `Message`
per request), :8903, its own Agent Card.

Shaped so stage 5 adds Researcher (:8904) and Critic (:8905) without a
rewrite: the per-request body every executor will share -- read input, load
tools, build agent, invoke, enqueue one Message -- differs between the
three agents only in allowlist, factory and rendering. Whether that becomes
a shared base class is decided once the second and third executors exist to
generalise from, not from this one example (stage-4 kickoff, known risks).

Reference for every A2A shape below: spec Sec5a
(docs/superpowers/specs/2026-08-14-mcp-a2a-refactor-design-en.md), measured
against the installed a2a-sdk 1.1.2 at the stage-4 kickoff --
a2a-python/samples/ ships with neither the installed package nor this
checkout, so it is not the reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    # This file already lives at the project root, so `python a2a_servers.py`
    # already puts _PROJECT_ROOT on sys.path[0] -- unlike mcp_servers/*.py,
    # one directory down, which needed this fix for real (stage 2). Kept
    # here for parity with that established pattern, and so a future
    # invocation from a different location cannot silently regress.
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn  # noqa: E402
from a2a.helpers import new_text_message  # noqa: E402
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
)
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH, DEFAULT_RPC_URL  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from starlette.applications import Starlette  # noqa: E402

from agents.planner import PLANNER_ALLOWLIST, create_planner_agent  # noqa: E402
from config import Settings, load_settings  # noqa: E402
from mcp_utils import load_agent_tools  # noqa: E402
from schemas import render_plan  # noqa: E402

# The Agent Card's own version, not the A2A spec version (spec Sec5a).
_AGENT_CARD_VERSION = "0.1.0"


class PlannerExecutor(AgentExecutor):
    """One Planner run per A2A request.

    Stateless (spec Sec8): a fresh agent is built and discarded every call --
    no checkpointer, and (with one server per agent) no shared client across
    requests either.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request = context.get_user_input()
        settings = load_settings()
        tools = await load_agent_tools(settings, PLANNER_ALLOWLIST)
        agent = create_planner_agent(settings, tools)
        result = await agent.ainvoke({"messages": [HumanMessage(request)]})
        await event_queue.enqueue_event(
            new_text_message(render_plan(result["structured_response"]))
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Pattern A never creates a Task, so there is nothing long-running to
        # cancel (spec Sec5a).
        raise NotImplementedError("PlannerExecutor has no cancellable task")


def build_agent_card(settings: Settings) -> AgentCard:
    """The Planner's published Agent Card, served at
    `/.well-known/agent-card.json`.

    No top-level `url` in A2A spec v1.0 (measured, spec Sec5a): the address
    lives in `supported_interfaces` instead, so this is not an oversight.
    """
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
    interface = AgentInterface(
        url=f"http://127.0.0.1:{settings.planner_a2a_port}",
        protocol_binding="JSONRPC",
        protocol_version="1.0",
    )
    return AgentCard(
        name="Planner",
        description=(
            "Decomposes a research request into a structured plan before "
            "any research happens."
        ),
        version=_AGENT_CARD_VERSION,
        supported_interfaces=[interface],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[skill],
    )


def build_app(settings: Settings) -> Starlette:
    """Assemble the Starlette app serving both the JSON-RPC endpoint and the
    Agent Card. `A2AStarletteApplication` does not exist in a2a-sdk 1.1.2
    (measured, spec Sec5a) -- route assembly replaces it."""
    agent_card = build_agent_card(settings)
    # InMemoryTaskStore is a required DefaultRequestHandler argument
    # (measured, spec Sec5a) even though Pattern A never creates a Task --
    # an SDK requirement, not a retreat from spec Sec8's "no persistent
    # TaskStore". The store stays in memory and stays empty.
    handler = DefaultRequestHandler(
        agent_executor=PlannerExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = create_jsonrpc_routes(handler, DEFAULT_RPC_URL) + create_agent_card_routes(
        agent_card, card_url=AGENT_CARD_WELL_KNOWN_PATH
    )
    return Starlette(routes=routes)


def main() -> None:
    settings = load_settings()
    uvicorn.run(build_app(settings), host="127.0.0.1", port=settings.planner_a2a_port)


if __name__ == "__main__":
    main()
