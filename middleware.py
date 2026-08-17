"""Middleware for the three A2A sub-agents (spec Sec10).

Two classes ported near-verbatim from the hl8 donor
(`../MA_systems_hl8_project/MA_system_hl8/middleware.py`), and one shared
list, `a2a_agent_middleware`, assembled from `langchain.agents.middleware`
public classes in the order spec Sec10 pins: `ModelCallLimit -> ToolCallLimit
-> ToolError -> ToolRetry -> ModelRetry`. `SaveReportGuardMiddleware` is
Supervisor middleware and lands at stage 7, not here.

The ordering invariant this module depends on -- `ToolErrorMiddleware` must
sit outside `ToolRetryMiddleware`, and the retry must carry
`on_failure="error"` -- was measured behaviourally at the stage-5 kickoff
against the installed LangChain 1.3.15 (`insights.md`, 2026-08-17), not
assumed from the version that was current when spec Sec10 was first written.
`tests/test_middleware_order.py` pins both clauses.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from config import CRITIC_VERIFICATION_INSTRUCTION

_VERIFICATION_TOOLS = frozenset({"web_search", "read_url", "knowledge_search"})

# Not a Settings field: nothing varies it yet, and a field with no consumer
# is noise. A generous but bounded backstop against a model that loops
# producing text without tool calls -- ToolCallLimitMiddleware cannot see
# that failure shape at all.
_MODEL_CALL_LIMIT = 20


def _run_tool_call_ids(messages: list[BaseMessage], tool_name: str) -> list[str]:
    """Ids of every call to `tool_name` since the most recent `HumanMessage`.

    A limit scoped to "this run" must reset each turn instead of
    accumulating across a checkpointed thread's whole history -- counting
    from the end of `messages` back to the most recent `HumanMessage` is
    what gives a limit that scope.

    Parameters
    ----------
    messages : list of BaseMessage
        The agent state's message list.
    tool_name : str
        The tool to count calls for.

    Returns
    -------
    list of str
        Tool-call ids, oldest first.
    """
    ids: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            ids = []
        elif isinstance(message, AIMessage):
            ids.extend(
                call["id"]
                for call in message.tool_calls
                if call["name"] == tool_name and call["id"] is not None
            )
    return ids


class ReadUrlCapMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Caps how many `read_url` calls the Researcher may make in one run.

    Without a cap the Researcher can spend its whole tool budget reading
    pages a search already found, instead of running the fresh searches the
    plan actually asks for. `max_calls=None` removes the cap.

    Defines **both** `wrap_tool_call` and `awrap_tool_call`. Every SearchMCP
    tool loaded through `langchain-mcp-adapters` is async-only (its sync
    `_run` raises `NotImplementedError`), and every A2A executor invokes via
    `ainvoke` -- a middleware that defined only the sync hook would fall
    back to `AgentMiddleware`'s default `awrap_tool_call`, which itself
    unconditionally raises `NotImplementedError`. That failure is
    indistinguishable from a real tool error once `ToolErrorMiddleware`
    converts it to a string, and no offline test in this project invokes an
    agent asynchronously -- found only by the stage-5 live check against
    real MCP tools (`insights.md`, 2026-08-17).
    """

    def __init__(self, max_calls: int | None) -> None:
        super().__init__()
        self.max_calls = max_calls

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        refusal = self._refusal(request)
        if refusal is not None:
            return refusal
        return await handler(request)

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        if self.max_calls is None or request.tool_call["name"] != "read_url":
            return None

        call_id = request.tool_call["id"]
        prior_calls = [
            prior_id
            for prior_id in _run_tool_call_ids(request.state["messages"], "read_url")
            if prior_id != call_id
        ]
        if len(prior_calls) >= self.max_calls:
            return ToolMessage(
                content=(
                    f"ERROR: read_url call limit ({self.max_calls}) reached for "
                    "this run. Run a new web_search or knowledge_search instead "
                    "of reading another page."
                ),
                tool_call_id=call_id,
                name="read_url",
                status="error",
            )
        return None


class CriticVerificationMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Forces the Critic to verify at least one claim before it verdicts.

    `response_format=CritiqueResult` lets the model end the turn with a
    verdict and no verification call at all. If none of the three research
    tools ran earlier this turn, this middleware re-runs the model call once
    with `CRITIC_VERIFICATION_INSTRUCTION` appended. The retried response is
    returned as-is, whatever it contains -- one-shot, so a model that skips
    verification twice in a row cannot make this middleware loop.

    Defines **both** `wrap_model_call` and `awrap_model_call`. Every A2A
    executor invokes via `ainvoke`; a middleware that defined only the sync
    hook would fall back to `AgentMiddleware`'s default `awrap_model_call`,
    which unconditionally raises `NotImplementedError` before the real
    model is ever called -- found only by the stage-5 live check, where the
    Critic's structured response came back empty with no other symptom
    (`insights.md`, 2026-08-17).
    """

    def __init__(self, min_verification_calls: int = 1) -> None:
        super().__init__()
        self.min_verification_calls = min_verification_calls

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        response = handler(request)
        if self._calls_a_verification_tool(response) or self._verified_earlier(request):
            return response
        return handler(self._retry_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        response = await handler(request)
        if self._calls_a_verification_tool(response) or self._verified_earlier(request):
            return response
        return await handler(self._retry_request(request))

    @staticmethod
    def _retry_request(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        # The retry forces a tool call at the API level (ChatOpenAI
        # translates "any" into "required"): the stage-5 live check observed
        # a model declining the plain-text nudge and verdicting with zero
        # verification calls. Enforcement must ride `model_settings`, not
        # the `tool_choice` field -- the ProviderStrategy bind path (the
        # Critic's actual path; `langchain/agents/factory.py:1387-1400` in
        # 1.3.15) builds its bind kwargs from the response format plus
        # `model_settings` only and silently drops `request.tool_choice`.
        # The field is still set too, for the plain-tools bind path, where
        # `model_settings` carrying tool_choice would raise a duplicate-
        # kwarg TypeError instead. The one-shot cap stays as the backstop
        # for a provider that ignores forcing.
        retry_messages = [
            *request.messages,
            HumanMessage(content=CRITIC_VERIFICATION_INSTRUCTION),
        ]
        if isinstance(request.response_format, ProviderStrategy):
            return request.override(
                messages=retry_messages,
                model_settings={**request.model_settings, "tool_choice": "any"},
            )
        return request.override(messages=retry_messages, tool_choice="any")

    def _verified_earlier(self, request: ModelRequest[ContextT]) -> bool:
        # `AgentState["messages"]` is `list[AnyMessage]`, a Union alias;
        # `list` is invariant, so mypy rejects it as a `list[BaseMessage]`
        # argument even though every member of the union is one.
        messages = cast("list[BaseMessage]", request.state["messages"])
        verified_calls = sum(
            len(_run_tool_call_ids(messages, tool_name))
            for tool_name in _VERIFICATION_TOOLS
        )
        return verified_calls >= self.min_verification_calls

    @staticmethod
    def _calls_a_verification_tool(response: ModelResponse[ResponseT]) -> bool:
        return any(
            isinstance(message, AIMessage)
            and any(call["name"] in _VERIFICATION_TOOLS for call in message.tool_calls)
            for message in response.result
        )


def _tool_error_to_message(exc: Exception, request: ToolCallRequest) -> str:
    """Names the exception type rather than echoing its message, keeping
    internal detail out of the model's context (the library's own guidance)
    while preserving hl8's "an error is data" invariant."""
    return f"ERROR: {request.tool_call['name']} failed ({type(exc).__name__})"


def a2a_agent_middleware(
    *, tool_call_limit: int, model_call_limit: int = _MODEL_CALL_LIMIT
) -> list[AgentMiddleware[Any, Any, Any]]:
    """The middleware stack spec Sec10 prescribes for every A2A agent.

    Parameters
    ----------
    tool_call_limit : int
        Per-run tool-call budget, e.g. `settings.researcher_max_tool_calls`.
    model_call_limit : int, default `_MODEL_CALL_LIMIT`
        Per-run model-call budget -- the only bound on a model that loops
        producing text without tool calls, which `ToolCallLimitMiddleware`
        cannot see.

    Returns
    -------
    list of AgentMiddleware
        `ModelCallLimit -> ToolCallLimit -> ToolError -> ToolRetry ->
        ModelRetry`, list order = outermost first. Callers append their own
        agent-specific middleware (`ReadUrlCapMiddleware`,
        `CriticVerificationMiddleware`) after this list.

    Notes
    -----
    `ToolRetryMiddleware` is constructed with `on_failure="error"` and
    `jitter=True`. Both are load-bearing, not defaults restated for
    clarity: without `on_failure="error"` the outer `ToolErrorMiddleware`'s
    handler never fires (measured, `insights.md` 2026-08-17); `jitter=True`
    exists because `evals/run_eval.py` runs dataset examples concurrently
    against one SearchMCP (spec Sec9's retry policy).
    """
    return [
        ModelCallLimitMiddleware(run_limit=model_call_limit),
        ToolCallLimitMiddleware(run_limit=tool_call_limit),
        ToolErrorMiddleware(_tool_error_to_message),
        ToolRetryMiddleware(on_failure="error", jitter=True),
        ModelRetryMiddleware(),
    ]
