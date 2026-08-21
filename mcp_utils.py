"""Per-agent tool loading over SearchMCP (spec Sec6 rule 3).

`MultiServerMCPClient.get_tools()` returns *everything* the connected server
offers -- it has no tool-name filter, only a server-name one (measured,
stage-4 kickoff) -- so the allowlist that decides what each agent may
actually use needs a home. This is it.

The transport/operational error split spec Sec5 originally assigned this
module is, measured, mostly already the adapter's default behaviour:
`handle_tool_errors` (kept at `True`, its own default) only governs MCP
execution errors (`isError=True`), and no tool in this project ever sets
that flag -- every operational and policy refusal already returns a plain
`ERROR:` string, by the convention `mcp_servers/*.py` established. Transport
and session failures raise regardless of this flag. So this module's real
job is the allowlist plus this documented, pinned default -- passing `False`
would only convert this project's `isError`-free results into
`ToolException`s, for no gain (measured 2026-08-16).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from opentelemetry.propagate import inject

from auth import auth_headers
from config import Settings


class MissingToolError(Exception):
    """SearchMCP does not offer a tool this agent's allowlist names.

    Structural (spec Sec5): the fix is starting SearchMCP or correcting a
    name in the allowlist, never a retry.
    """


def _with_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Explicit W3C `traceparent` injection (stage 9, D10's declared
    fallback -- measured NO-GO at the stage-9 live check, not a
    precaution). Automatic httpx instrumentation alone lost `traceparent`
    on roughly half of an MCP session's own HTTP calls: the SDK's
    streamable-HTTP session runs its request lifecycle across internal task
    boundaries that do not reliably inherit ambient OTel context. Baking
    the header into the connection's *static* headers -- computed once,
    here, from whatever span is active in the caller -- applies it to every
    request the session issues, the same way `supervisor.py`'s A2A hop
    already gets it for free from `create_client`'s single per-call
    `httpx.AsyncClient`.
    """
    carrier = dict(headers)
    inject(carrier)
    return carrier


async def _load_tools(
    url: str,
    allowlist: Sequence[str],
    *,
    connection_key: str,
    label: str,
    settings: Settings,
) -> list[BaseTool]:
    """Connect to one MCP server and return exactly the tools `allowlist`
    names, in that order. Shared by `load_agent_tools` (SearchMCP) and
    `load_report_tools` (ReportMCP, stage 7) -- the allowlist mechanism and
    the transport-vs-tool error split are the same for both servers.

    Parameters
    ----------
    url : str
        The server's Streamable HTTP endpoint.
    allowlist : Sequence of str
        Tool names expected to exist.
    connection_key : str
        The `MultiServerMCPClient` connections dict key, e.g. `"search"`.
    label : str
        Human-readable server name for the `MissingToolError` message.
    settings : Settings
        Source of the shared bearer token attached to this connection
        (stage 8); `auth_headers` returns `{}` when unset.

    Raises
    ------
    MissingToolError
        `allowlist` names a tool the server does not offer.
    """
    client = MultiServerMCPClient(
        {
            connection_key: StreamableHttpConnection(
                transport="streamable_http",
                url=url,
                headers=_with_trace_context(auth_headers(settings)),
            )
        },
        handle_tool_errors=True,
    )
    by_name = {tool.name: tool for tool in await client.get_tools()}

    missing = [name for name in allowlist if name not in by_name]
    if missing:
        raise MissingToolError(
            f"{label} does not offer {missing!r} -- it offers {sorted(by_name)}"
        )
    return [by_name[name] for name in allowlist]


async def load_agent_tools(
    settings: Settings, allowlist: Sequence[str]
) -> list[BaseTool]:
    """Load SearchMCP's tools and return exactly those named in `allowlist`.

    Parameters
    ----------
    settings : Settings
    allowlist : Sequence of str
        Tool names this agent may use, e.g. `agents.planner.PLANNER_ALLOWLIST`.
        Discovery answers "what does the server offer"; this answers "what
        may this agent use" -- different questions (spec Sec6 rule 3).

    Returns
    -------
    list of BaseTool
        One tool per name in `allowlist`, in that order.

    Raises
    ------
    MissingToolError
        `allowlist` names a tool SearchMCP does not offer.
    """
    return await _load_tools(
        settings.search_mcp_url(),
        allowlist,
        connection_key="search",
        label="SearchMCP",
        settings=settings,
    )


async def load_report_tools(settings: Settings) -> BaseTool:
    """Load ReportMCP's one tool, `save_report` (stage 7, audit item A4).

    `load_agent_tools` hard-codes a connection at `search_mcp_url()`, so it
    cannot reach ReportMCP -- this is the second loader that gives the
    Supervisor's `save_report_tool` argument a source other than a live
    server call inline in `main.py`.

    Parameters
    ----------
    settings : Settings

    Returns
    -------
    BaseTool

    Raises
    ------
    MissingToolError
        ReportMCP does not offer `save_report`.
    """
    tools = await _load_tools(
        settings.report_mcp_url(),
        ("save_report",),
        connection_key="report",
        label="ReportMCP",
        settings=settings,
    )
    return tools[0]


async def read_resource(
    *, url: str, uri: str, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """Read one MCP resource and decode it as JSON.

    Parameters
    ----------
    url : str
        The server's Streamable HTTP endpoint, e.g. `settings.search_mcp_url()`
        or `settings.report_mcp_url()`. Taken as an argument rather than
        derived from `Settings` plus a server name: the caller (`main.py`) is
        where the choice of server belongs.
    uri : str
        The resource URI, e.g. `"resource://knowledge-base-stats"`.
    headers : dict of str to str, optional
        Extra request headers, e.g. `auth.auth_headers(settings)` (stage 8).
        This function stays settings-agnostic for the same reason `url`
        does -- the caller states its own intent.

    Returns
    -------
    dict of str to Any
        The resource's JSON content.

    Notes
    -----
    Both of this project's resources report `mimeType="text/plain"` even
    though their content is JSON (measured against FastMCP 3.4.7's source,
    stage-6 kickoff) -- decoding is unconditional, never branching on
    `mimeType`.
    """
    client = MultiServerMCPClient(
        {
            "target": StreamableHttpConnection(
                transport="streamable_http",
                url=url,
                headers=_with_trace_context(headers or {}),
            )
        }
    )
    blobs = await client.get_resources("target", uris=uri)
    return dict(json.loads(blobs[0].as_string()))
