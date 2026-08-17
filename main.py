"""The REPL that drives the Supervisor over one `thread_id` per session
(stage 6). `async def main()` is not a style choice: the Supervisor's
delegation tools are async-only (probe A-corollary, `supervisor.py`), so
`.invoke()`/`.stream()` raise `NotImplementedError` at the framework level --
this module drives the graph with `.ainvoke()`/`.astream()` exclusively.

No `hitl` import -- nothing can interrupt this stage (stage 7's own module,
pinned by `tests/test_main.py`'s AST check so its work is never accidentally
started here).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
from a2a.client import A2ACardResolver
from langchain_core.messages import HumanMessage

from config import Settings, load_settings
from mcp_utils import read_resource
from models import resolved_map
from supervisor import create_supervisor

_QUIT_COMMANDS = frozenset({"exit", "quit"})


class PreflightError(Exception):
    """Structural (spec Sec5): a dependency the REPL needs is not reachable
    at startup. Refused before `input()` is ever reached, naming the down
    server and the command that starts it, mirroring
    `mcp_utils.MissingToolError`'s convention."""


async def _preflight(settings: Settings) -> dict[str, Any]:
    """Read both MCP resources and resolve all three Agent Cards.

    Parameters
    ----------
    settings : Settings

    Returns
    -------
    dict of str to Any
        `search_stats`, `output_dir` (both MCP resources' JSON content) and
        `provider_map` (`models.resolved_map`, no network call).

    Raises
    ------
    PreflightError
        A connection failure -- never the resource's own in-band answer
        ("no index yet", an empty `output/`), which is a *successful* fetch.
    """
    results: dict[str, Any] = {}

    try:
        results["search_stats"] = await read_resource(
            url=settings.search_mcp_url(), uri="resource://knowledge-base-stats"
        )
    except Exception as error:
        raise PreflightError(
            "SearchMCP is not reachable -- start it with "
            "`python mcp_servers/search_mcp.py`"
        ) from error

    try:
        results["output_dir"] = await read_resource(
            url=settings.report_mcp_url(), uri="resource://output-dir"
        )
    except Exception as error:
        raise PreflightError(
            "ReportMCP is not reachable -- start it with "
            "`python mcp_servers/report_mcp.py`"
        ) from error

    agents = (
        ("Planner", settings.planner_a2a_url(), "planner"),
        ("Researcher", settings.researcher_a2a_url(), "researcher"),
        ("Critic", settings.critic_a2a_url(), "critic"),
    )
    async with httpx.AsyncClient() as client:
        for name, url, argv in agents:
            try:
                await A2ACardResolver(client, url).get_agent_card()
            except Exception as error:
                raise PreflightError(
                    f"{name} is not reachable at {url} -- start it with "
                    f"`python a2a_servers.py {argv}`"
                ) from error

    results["provider_map"] = resolved_map(settings)
    return results


def _print_banner(preflight: dict[str, Any]) -> None:
    print("Supervisor ready. Providers:")
    for role, (provider, model) in preflight["provider_map"].items():
        print(f"  {role}: {provider}/{model}")
    print(f"Knowledge base: {preflight['search_stats']}")
    print(f"Output directory: {preflight['output_dir']}")


def build_run_config(settings: Settings) -> dict[str, Any]:
    """One `thread_id` for the whole REPL session (D0's premise: the
    Supervisor forwards the *last* `HumanMessage`, which only makes sense if
    a session's questions share one thread)."""
    return {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": settings.recursion_limit,
    }


def _print_update(chunk: dict[str, Any]) -> None:
    for node, update in chunk.items():
        messages = update.get("messages", []) if isinstance(update, dict) else []
        for message in messages:
            content = getattr(message, "content", None)
            if content:
                print(f"[{node}] {content}")


async def _run_turn(
    agent: Any, config: dict[str, Any], settings: Settings, text: str
) -> None:
    """Drive one question through the graph, bounded by
    `supervisor_run_timeout_seconds` (steps/cost/wall-clock triple, alongside
    `recursion_limit` and the per-call `httpx timeout=120s`)."""
    payload = {"messages": [HumanMessage(text)]}
    try:
        async with asyncio.timeout(settings.supervisor_run_timeout_seconds):
            async for chunk in agent.astream(payload, config, stream_mode="updates"):
                _print_update(chunk)
    except TimeoutError:
        print(
            f"[system] Run exceeded {settings.supervisor_run_timeout_seconds}s "
            "and was stopped. No report was saved this turn."
        )


async def main() -> None:
    settings = load_settings()
    try:
        preflight = await _preflight(settings)
    except PreflightError as error:
        print(f"[system] {error}")
        return

    _print_banner(preflight)
    agent = create_supervisor(settings)
    config = build_run_config(settings)

    while True:
        try:
            text = input("> ")
        except EOFError:
            break
        if not text.strip():
            continue
        if text.strip().lower() in _QUIT_COMMANDS:
            break
        await _run_turn(agent, config, settings, text)


if __name__ == "__main__":
    asyncio.run(main())
