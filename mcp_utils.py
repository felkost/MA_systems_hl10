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
`ToolException`s, for no gain (insights.md 2026-08-16).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

from config import Settings


class MissingToolError(Exception):
    """SearchMCP does not offer a tool this agent's allowlist names.

    Structural (spec Sec5): the fix is starting SearchMCP or correcting a
    name in the allowlist, never a retry.
    """


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
    client = MultiServerMCPClient(
        {
            "search": StreamableHttpConnection(
                transport="streamable_http", url=settings.search_mcp_url()
            )
        },
        handle_tool_errors=True,
    )
    by_name = {tool.name: tool for tool in await client.get_tools()}

    missing = [name for name in allowlist if name not in by_name]
    if missing:
        raise MissingToolError(
            f"SearchMCP does not offer {missing!r} -- it offers {sorted(by_name)}"
        )
    return [by_name[name] for name in allowlist]


async def read_resource(*, url: str, uri: str) -> dict[str, Any]:
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
        {"target": StreamableHttpConnection(transport="streamable_http", url=url)}
    )
    blobs = await client.get_resources("target", uris=uri)
    return dict(json.loads(blobs[0].as_string()))
