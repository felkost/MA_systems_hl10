"""The REPL that drives the Supervisor over one `thread_id` per session,
with HITL on `save_report` and a crash-safe checkpoint (stage 7).
`async def main()` is not a style choice: the Supervisor's delegation tools
are async-only (probe A-corollary, `supervisor.py`), so `.invoke()`/
`.stream()` raise `NotImplementedError` at the framework level -- this
module drives the graph with `.ainvoke()`/`.astream()` exclusively.

**D2 (stage 7):** `AsyncSqliteSaver`, not `SqliteSaver` -- the sync saver
raises `NotImplementedError` on every async method, and this module never
calls anything else. `AsyncSqliteSaver.from_conn_string` is an
`asynccontextmanager`, so its lifetime is this module's `main()`, wrapping
the whole REPL session; `create_supervisor` receives the saver rather than
constructing one.

**D3:** `asyncio.timeout(settings.supervisor_run_timeout_seconds)` wraps
only the segments where the graph is actually running -- it is exited
before the human is prompted at a HITL interrupt and re-entered fresh for
the resumed segment, so a slow approval can never cancel the run at the one
step that must not be lost.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Sequence
from typing import Any, cast

import httpx
from a2a.client import A2ACardResolver
from langchain.agents.middleware.human_in_the_loop import Decision, HITLRequest
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

import hitl
from config import Settings, load_settings
from mcp_utils import load_report_tools, read_resource
from models import resolved_map
from paths import checkpoint_path
from supervisor import create_supervisor

_QUIT_COMMANDS = frozenset({"exit", "quit"})


class PreflightError(Exception):
    """Structural (spec Sec5): a dependency the REPL needs is not reachable
    at startup. Refused before `input()` is ever reached, naming the down
    server and the command that starts it, mirroring
    `mcp_utils.MissingToolError`'s convention."""


def parse_args(argv: Sequence[str] = ()) -> argparse.Namespace:
    """Parse the REPL's command-line arguments.

    Parameters
    ----------
    argv : Sequence of str, default ()
        Explicit, never `sys.argv` implicitly -- keeps every caller (tests
        included) free of ambient argv pollution.

    Returns
    -------
    argparse.Namespace
        `.thread`: the `--thread <id>` value, or `None` to start a fresh
        session.
    """
    parser = argparse.ArgumentParser(description="Multi-agent research REPL")
    parser.add_argument(
        "--thread",
        default=None,
        help="Resume an existing session by its thread id, printed in the "
        "banner at the start of the original session.",
    )
    return parser.parse_args(list(argv))


async def _preflight(settings: Settings) -> dict[str, Any]:
    """Read both MCP resources, load `save_report`, and resolve all three
    Agent Cards.

    Parameters
    ----------
    settings : Settings

    Returns
    -------
    dict of str to Any
        `search_stats`, `output_dir` (both MCP resources' JSON content),
        `save_report_tool` (loaded via ReportMCP) and `provider_map`
        (`models.resolved_map`, no network call).

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

    try:
        results["save_report_tool"] = await load_report_tools(settings)
    except Exception as error:
        raise PreflightError(
            "ReportMCP does not offer save_report -- start it with "
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
    print(f"Thread id: {preflight['thread_id']}")


def build_run_config(settings: Settings, thread_id: str) -> RunnableConfig:
    """Config for one `thread_id`, explicit rather than generated here.

    Parameters
    ----------
    settings : Settings
    thread_id : str
        A fresh `uuid4` for a new session, or the `--thread` value being
        resumed -- the caller decides which (`main`/`_resolve_thread_config`),
        so this function stays a pure builder.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }


async def _resolve_thread_config(
    saver: AsyncSqliteSaver, thread: str | None, settings: Settings
) -> RunnableConfig:
    """Build this session's run config: a fresh thread, or a verified
    resume of an existing one.

    Parameters
    ----------
    saver : AsyncSqliteSaver
    thread : str or None
        The `--thread` value, or `None` for a new session.
    settings : Settings

    Raises
    ------
    PreflightError
        `thread` names a thread with no checkpoint. hl8's measured trap:
        resuming an unknown thread must refuse, never silently start a new,
        empty conversation under the id the human typed.
    """
    if thread is None:
        return build_run_config(settings, str(uuid.uuid4()))
    config = build_run_config(settings, thread)
    if await saver.aget(config) is None:
        raise PreflightError(
            f"No checkpoint for thread {thread!r} -- start a new session "
            "without --thread, or use the id printed at the start of the "
            "original session."
        )
    return config


def _prompt_one_decision() -> Decision:
    while True:
        raw = input("approve / edit / reject: ").strip().lower()
        if raw in ("approve", "edit", "reject"):
            choice = cast(hitl.ReplChoice, raw)
            break
    text = ""
    if choice == "edit":
        text = input("Your feedback: ")
    elif choice == "reject":
        text = input("Reason: ")
    return hitl.build_decision(choice, text)


def _resolve_interrupt(request: HITLRequest) -> list[Decision]:
    """Render the real tool-call arguments and collect one decision per
    gated action request -- outside any `asyncio.timeout` (D3)."""
    print("=" * 60)
    print("ACTION REQUIRES APPROVAL")
    print(hitl.render_interrupt(request))
    return [_prompt_one_decision() for _ in request["action_requests"]]


def _extract_messages(chunk: dict[str, Any]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for update in chunk.values():
        if isinstance(update, dict):
            messages.extend(update.get("messages", []))
    return messages


def _print_update(chunk: dict[str, Any]) -> None:
    for node, update in chunk.items():
        messages = update.get("messages", []) if isinstance(update, dict) else []
        for message in messages:
            content = getattr(message, "content", None)
            if content:
                print(f"[{node}] {content}")


async def _drive_payload(
    agent: Any, config: RunnableConfig, settings: Settings, payload: Any
) -> list[BaseMessage]:
    """Alternate graph-driving segments and HITL interrupts until the turn
    ends, bounded per segment by `supervisor_run_timeout_seconds` (D3).

    Returns
    -------
    list of BaseMessage
        Every message the turn produced, across every segment -- the
        `outcomes` `hitl.render_save_status` reads.
    """
    seen: list[BaseMessage] = []
    while True:
        interrupts: tuple[Any, ...] | None = None
        try:
            async with asyncio.timeout(settings.supervisor_run_timeout_seconds):
                async for chunk in agent.astream(
                    payload, config, stream_mode="updates"
                ):
                    if "__interrupt__" in chunk:
                        interrupts = chunk["__interrupt__"]
                        break
                    seen.extend(_extract_messages(chunk))
                    _print_update(chunk)
        except TimeoutError:
            print(
                f"[system] Run exceeded {settings.supervisor_run_timeout_seconds}s "
                "and was stopped. No report was saved this turn."
            )
            return seen
        if interrupts is None:
            return seen
        decisions = _resolve_interrupt(interrupts[0].value)
        payload = Command(resume={"decisions": decisions})


async def _drive(
    agent: Any, config: RunnableConfig, settings: Settings, text: str
) -> list[BaseMessage]:
    return await _drive_payload(
        agent, config, settings, {"messages": [HumanMessage(text)]}
    )


async def main(argv: Sequence[str] = ()) -> None:
    args = parse_args(argv)
    settings = load_settings()

    async with AsyncSqliteSaver.from_conn_string(
        str(checkpoint_path(settings.checkpoint_db))
    ) as saver:
        try:
            preflight = await _preflight(settings)
            config = await _resolve_thread_config(saver, args.thread, settings)
        except PreflightError as error:
            print(f"[system] {error}")
            return

        preflight["thread_id"] = config["configurable"]["thread_id"]
        _print_banner(preflight)

        agent = create_supervisor(
            settings,
            checkpointer=saver,
            save_report_tool=preflight["save_report_tool"],
        )

        if args.thread is not None:
            state = await agent.aget_state(config)
            if state.interrupts:
                decisions = _resolve_interrupt(state.interrupts[0].value)
                messages = await _drive_payload(
                    agent, config, settings, Command(resume={"decisions": decisions})
                )
                print(hitl.render_save_status(messages))

        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            if not text.strip():
                continue
            if text.strip().lower() in _QUIT_COMMANDS:
                break
            messages = await _drive(agent, config, settings, text)
            print(hitl.render_save_status(messages))


if __name__ == "__main__":
    import sys

    asyncio.run(main(sys.argv[1:]))
