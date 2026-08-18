"""Shared bearer-token authentication for all five servers (spec Sec7).

One `Settings` field, one verifier, reused two ways: FastMCP's own `auth=`
constructor kwarg (its `AuthenticationMiddleware`/`BearerAuthBackend` wire in
automatically once `.auth` is set -- read lazily at `http_app()`/
`run_http_async()` time, not at `FastMCP()` construction, measured against
the installed `fastmcp==3.4.7` source) for the two MCP servers, and
`a2a_auth_middleware` for the three A2A Starlette apps, whose SDK has no
enforcement path of its own (`AgentCard.security_schemes` is declarative
only, measured against the installed `a2a-sdk` 1.1.2).

`verifier.get_middleware()` alone does not enforce anything -- Starlette's
`AuthenticationMiddleware` only *populates* `scope["user"]`; a request with
no `Authorization` header proceeds unrejected. FastMCP itself enforces by
wrapping its MCP endpoint in `RequireAuthMiddleware`
(`fastmcp/server/http.py:473,614`), which 401s when `scope["user"]` is not
`AuthenticatedUser`. `a2a_auth_middleware` reuses that same class so the A2A
side gets identical enforcement, not a second hand-rolled implementation.

Enforcement lives at server *boot*, not at `Settings` construction:
`build_token_verifier` raises if no token is configured, but only the three
server `main()`s and `servers.py` call it -- `Settings` itself stays
permissive, so the existing tests that build a `Settings` object for
unrelated reasons are untouched.
"""

from __future__ import annotations

from fastmcp.server.auth import TokenVerifier
from fastmcp.server.auth.middleware import RequireAuthMiddleware
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from starlette.middleware import Middleware

from config import Settings

_CLIENT_ID = "mcp-a2a-shared"


class MissingSharedTokenError(Exception):
    """`settings.mcp_a2a_shared_token` is unset at server boot.

    Structural (spec Sec5): the fix is setting MCP_A2A_SHARED_TOKEN in
    `.env`, never a retry -- mirrors main.py's PreflightError convention.
    """


def build_token_verifier(settings: Settings) -> TokenVerifier:
    """The one verifier every server's `auth=`/middleware wires in.

    Raises
    ------
    MissingSharedTokenError
        No token configured. Called only from the three server `main()`
        functions and `servers.py` -- never from `Settings` construction.
    """
    if settings.mcp_a2a_shared_token is None:
        raise MissingSharedTokenError(
            "MCP_A2A_SHARED_TOKEN is not set -- add it to .env before "
            "starting any server"
        )
    token = settings.mcp_a2a_shared_token.get_secret_value()
    return StaticTokenVerifier(tokens={token: {"client_id": _CLIENT_ID}})


def a2a_auth_middleware(verifier: TokenVerifier) -> list[Middleware]:
    """The two-layer stack `a2a_servers.py` attaches to each `Starlette` app.

    `verifier.get_middleware()` populates `scope["user"]`;
    `RequireAuthMiddleware` -- the same class FastMCP wraps its own MCP
    endpoint in -- rejects whatever didn't authenticate. First entry is
    outermost (authentication must run before enforcement reads
    `scope["user"]`), the same onion-order discipline already pinned for
    `wrap_tool_call`, now at the ASGI layer.
    """
    return [
        *verifier.get_middleware(),
        Middleware(RequireAuthMiddleware, required_scopes=[]),
    ]


def auth_headers(settings: Settings) -> dict[str, str]:
    """The header every client-side call in this system sends: `mcp_utils.py`'s
    MCP connections, `main.py`'s `A2ACardResolver` client, `supervisor.py`'s
    A2A client. Empty (not an error) if no token is configured -- the
    servers are the enforcement point, not every client-side code path, so
    tests exercising client code against in-memory/stubbed transports are
    unaffected by this field being unset.
    """
    if settings.mcp_a2a_shared_token is None:
        return {}
    token = settings.mcp_a2a_shared_token.get_secret_value()
    return {"Authorization": f"Bearer {token}"}
