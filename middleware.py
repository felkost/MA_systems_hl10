"""Middleware for the three A2A sub-agents and the Supervisor (spec Sec10).

Two classes ported near-verbatim from an earlier iteration of this system,
one shared list, `a2a_agent_middleware`, assembled from
`langchain.agents.middleware` public classes in the order spec Sec10 pins:
`ModelCallLimit -> ToolCallLimit -> ToolError -> ToolRetry -> ModelRetry`,
and two Supervisor-only classes, `RoundStabilityMiddleware` (stage 6) and
`SaveReportGuardMiddleware` (stage 7).

The ordering invariant this module depends on -- `ToolErrorMiddleware` must
sit outside `ToolRetryMiddleware`, and the retry must carry
`on_failure="error"` -- was measured behaviourally at the stage-5 kickoff
against the installed LangChain 1.3.15, not assumed from the version that
was current when spec Sec10 was first written. An automated test pins both
clauses.
"""

from __future__ import annotations

import time
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
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

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
    real MCP tools.
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
    Critic's structured response came back empty with no other symptom.
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


class LlmSpanMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """One `llm.<role>` span per actual provider request (stage 9, spec
    Sec16): `llm.provider`, `llm.model` and `role` on every span, plus
    duration and, when the response carries them, token counts and cost.

    Provider/model come from `settings.resolved(role)` at construction, not
    from the response -- pure and offline, and correct even when a test
    injects a fake `model=`, because the attributes then describe the
    *configuration under test* (stage 10b's "n>=3 runs attributable to one
    configuration" needs exactly this).

    Placement is deliberate: **last in the middleware list, innermost**, at
    every call site. `wrap_model_call` composes outermost-first (measured
    stage-5), so innermost means one span per actual provider request --
    `ModelRetryMiddleware`'s retries and `CriticVerificationMiddleware`'s
    re-request each get their own span with their own cost, instead of one
    span silently covering a whole retry storm.

    Defines **both** `wrap_model_call` and `awrap_model_call` -- the
    invariant measured at stage 5: every A2A executor and the Supervisor
    invoke via `ainvoke`, and the base class's async default raises
    `NotImplementedError` before the real model is ever called.
    """

    def __init__(self, role: str, provider: str, model: str) -> None:
        super().__init__()
        self.role = role
        self.provider = provider
        self.model = model

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"llm.{self.role}") as span:
            self._tag_span(span)
            started = time.perf_counter()
            try:
                response = handler(request)
            except Exception as error:
                self._record_failure(span, started, error)
                raise
            self._record_success(span, started, response)
            return response

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"llm.{self.role}") as span:
            self._tag_span(span)
            started = time.perf_counter()
            try:
                response = await handler(request)
            except Exception as error:
                self._record_failure(span, started, error)
                raise
            self._record_success(span, started, response)
            return response

    def _tag_span(self, span: trace.Span) -> None:
        span.set_attribute("llm.provider", self.provider)
        span.set_attribute("llm.model", self.model)
        span.set_attribute("role", self.role)

    @staticmethod
    def _record_failure(span: trace.Span, started: float, error: Exception) -> None:
        span.set_attribute("llm.duration_ms", (time.perf_counter() - started) * 1000)
        span.set_status(Status(StatusCode.ERROR, type(error).__name__))

    @staticmethod
    def _record_success(
        span: trace.Span, started: float, response: ModelResponse[ResponseT]
    ) -> None:
        span.set_attribute("llm.duration_ms", (time.perf_counter() - started) * 1000)
        message = next(
            (m for m in reversed(response.result) if isinstance(m, AIMessage)), None
        )
        if message is None:
            return
        usage = message.usage_metadata
        if usage:
            span.set_attribute("llm.tokens.input", usage.get("input_tokens", 0))
            span.set_attribute("llm.tokens.output", usage.get("output_tokens", 0))
            span.set_attribute("llm.tokens.total", usage.get("total_tokens", 0))
        cost = (message.response_metadata.get("token_usage") or {}).get("cost")
        if isinstance(cost, (int, float)):
            span.set_attribute("llm.cost_usd", cost)


def _tool_error_to_message(exc: Exception, request: ToolCallRequest) -> str:
    """Names the exception type rather than echoing its message, keeping
    internal detail out of the model's context (the library's own guidance)
    while preserving this project's "an error is data" invariant."""
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
    handler never fires (measured 2026-08-17); `jitter=True` exists because
    `evals/run_eval.py` runs dataset examples concurrently against one
    SearchMCP (spec Sec9's retry policy).
    """
    return [
        ModelCallLimitMiddleware(run_limit=model_call_limit),
        ToolCallLimitMiddleware(run_limit=tool_call_limit),
        ToolErrorMiddleware(_tool_error_to_message),
        ToolRetryMiddleware(on_failure="error", jitter=True),
        ModelRetryMiddleware(),
    ]


_STABILITY_TOOLS = frozenset({"delegate_to_researcher", "delegate_to_critic"})


def _tool_results(messages: list[BaseMessage], call_ids: list[str]) -> list[str]:
    """The `ToolMessage` content for each id in `call_ids`, in that order.

    Ids with no matching `ToolMessage` (e.g. a `Command`-returning tool that
    wrote no visible message this round) are skipped, so the caller sees only
    the results that actually exist to compare.
    """
    by_id = {
        message.tool_call_id: str(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    }
    return [by_id[call_id] for call_id in call_ids if call_id in by_id]


class RoundStabilityMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Two deterministic stop signals beyond the Supervisor's iteration
    counter (spec Sec6 amendment 2026-08-16, D4): *signal repetition* --
    `delegate_to_critic` refused once `state["critic_gaps"]` repeats
    `state["previous_critic_gaps"]` -- and *candidate stability* --
    `delegate_to_researcher` refused once its last two results in this run
    are byte-identical. Both compare the **structured** field or the raw
    tool result, never rendered prose, since `render_critique(...)`'s text
    can vary round to round even when the underlying gap does not.

    **D6 -- run-scoped, not thread-scoped.** `critic_gaps`/
    `previous_critic_gaps` survive across questions in one checkpointed
    `thread_id` (the Supervisor's REPL reuses one thread per session), so a
    comparison only fires once *this run* has produced at least two prior
    calls to the tool in question (`_run_tool_call_ids`, "since the most
    recent `HumanMessage`") -- state left over from an earlier question can
    never trigger a refusal on the first call of a new one.
    """

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
        name = request.tool_call["name"]
        if name not in _STABILITY_TOOLS:
            return None

        messages = cast(list[BaseMessage], request.state["messages"])
        call_id = request.tool_call["id"]
        prior = [
            prior_id
            for prior_id in _run_tool_call_ids(messages, name)
            if prior_id != call_id
        ]
        if len(prior) < 2:
            return None

        if name == "delegate_to_critic":
            gaps = request.state.get("critic_gaps")
            if gaps is None or gaps != request.state.get("previous_critic_gaps"):
                return None
            reason = "the Critic's gaps repeated the previous round's exactly"
        else:
            last_two = _tool_results(messages, prior[-2:])
            if len(last_two) < 2 or last_two[0] != last_two[1]:
                return None
            reason = "the Researcher's findings repeated the previous round's exactly"

        return ToolMessage(
            content=(
                f"ERROR: {name} call refused -- {reason}, so another round "
                "cannot discover anything new. Move on with what you have."
            ),
            tool_call_id=call_id,
            name=name,
            status="error",
        )


_SAVE_REPORT_NUDGE = (
    "The Critic's verdict was APPROVE and no save_report call has happened "
    "yet this run. Compose the final Markdown report and call save_report "
    "with it now."
)


class SaveReportGuardMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """One-shot nudge toward `save_report` after an APPROVE the model is
    about to leave unsaved (spec Sec10, D4/D9).

    Fires only when **all four** hold on the model's about-to-end response:
    it carries no tool calls, `state["verdict"] == "APPROVE"`, a
    `delegate_to_critic` result exists since the most recent `HumanMessage`
    (D5 -- not merely `state["verdict"]`, which a checkpointed thread
    carries across questions, the same trap D6 named for
    `RoundStabilityMiddleware`), and no `save_report` call exists since that
    same boundary. **D9: a standing REVISE is never forced** -- a run that
    exhausted its revision budget has no approved content, and forcing a
    save there would ship a report its own Critic rejected.

    One re-request with `tool_choice="any"`, whatever it returns. The
    Supervisor carries no `response_format`, so it is always on the
    plain-tools bind path -- `request.override(tool_choice=...)` is the
    correct channel here (unlike `CriticVerificationMiddleware`'s
    `ProviderStrategy` path, where the same field is measured to be a
    no-op and `model_settings` must carry it instead).

    Defines **both** `wrap_model_call` and `awrap_model_call` (the stage-5
    invariant: every A2A/Supervisor invocation in production is `ainvoke`).
    """

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        response = handler(request)
        if not self._should_nudge(request, response):
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
        if not self._should_nudge(request, response):
            return response
        return await handler(self._retry_request(request))

    def _should_nudge(
        self, request: ModelRequest[ContextT], response: ModelResponse[ResponseT]
    ) -> bool:
        if self._has_tool_calls(response):
            return False
        if request.state.get("verdict") != "APPROVE":
            return False
        messages = cast("list[BaseMessage]", request.state["messages"])
        if not _run_tool_call_ids(messages, "delegate_to_critic"):
            return False
        if _run_tool_call_ids(messages, "save_report"):
            return False
        return True

    @staticmethod
    def _retry_request(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        retry_messages = [
            *request.messages,
            HumanMessage(content=_SAVE_REPORT_NUDGE),
        ]
        return request.override(messages=retry_messages, tool_choice="any")

    @staticmethod
    def _has_tool_calls(response: ModelResponse[ResponseT]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in response.result
        )
