"""Process-local observability. Logging half (spec Sec12, stage 2) and OTel
half (spec Sec11, stage 9) -- one module, two halves, each landing when its
consumer exists.

Rotating file logs are the post-mortem artefact for a crash: OTel's
`BatchSpanProcessor` buffers spans in memory, and a hard crash loses the
unbuffered tail. A file handler flushes per line and survives it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, TraceIdRatioBased
from opentelemetry.trace import Status, StatusCode
from starlette.middleware import Middleware

import paths
from config import Settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [trace=%(trace_id)s] %(message)s"

# Span names are module constants so tests and production cannot drift
# (plan D4/D5). `mcp.tool.*` is parameterized by tool name, hence the
# prefix rather than one frozen string.
SPAN_REPL_QUESTION = "repl.question"
_MCP_TOOL_SPAN_PREFIX = "mcp.tool."

# The key `asgi_tracing_middleware`'s server_request_hook stores the just-
# started server span's context under, on the ASGI `scope` dict --
# `mcp_tool_span_middleware` reads it back to parent a tool span onto the
# HTTP request that carried it (D5), without depending on contextvars
# surviving the MCP session's own task boundary.
_SCOPE_PARENT_KEY = "observability.parent_context"


class _TraceContextFilter(logging.Filter):
    """Adds `trace_id` to every record -- `"-"` when no span is recording,
    so one format string works before `configure_observability` runs, with
    tracing disabled, and in the gate (D6, closing stage 2's deferral).

    Attached to the *handler*, not the logger: a logger-level filter is not
    applied to records that arrive from a child logger, a handler-level one
    is applied to everything the handler writes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        context = trace.get_current_span().get_span_context()
        record.trace_id = format(context.trace_id, "032x") if context.is_valid else "-"
        return True


def configure_logging(
    service: str,
    *,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Attach a rotating file handler on `paths.log_path(service, log_dir)`
    to the logger named `service`, and return it.

    Parameters
    ----------
    service : str
        Process name; also the logger name and the log file's stem.
    log_dir : str or Path, default "logs"
        Forwarded to `paths.log_path` -- explicit so a test can redirect it.
    level : int, default logging.INFO
        Logger level.
    max_bytes, backup_count : int
        `RotatingFileHandler` rollover parameters.

    Returns
    -------
    logging.Logger
        Idempotent: a logger that already carries a `RotatingFileHandler`
        pointed at this same resolved path is returned unchanged, not given
        a second handler -- otherwise importing a module that calls this
        more than once would duplicate every line it writes.
    """
    path = paths.log_path(service, log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(service)
    logger.setLevel(level)

    target = str(path.resolve())
    already_attached = any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", None) == target
        for handler in logger.handlers
    )
    if not already_attached:
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler.addFilter(_TraceContextFilter())
        logger.addHandler(handler)

    return logger


@dataclass(frozen=True)
class Observability:
    """What `configure_observability` set up for one process.

    Attributes
    ----------
    service : str
        The process name; also the OTel `service.name` resource attribute.
    tracer : trace.Tracer
        The tracer every manual span in this process is opened from.
    tracer_provider : TracerProvider or None
        `None` only when tracing is disabled and no `span_exporter` was
        injected -- callers never need it themselves, it is held so
        `shutdown_observability` can flush the batch processor.
    langfuse : Any or None
        The `Langfuse` client, or `None` under an injected exporter or when
        tracing is disabled.
    enabled : bool
    """

    service: str
    tracer: trace.Tracer
    tracer_provider: TracerProvider | None
    langfuse: Any | None
    enabled: bool


class TracingConfigurationError(Exception):
    """Tracing was requested and cannot be honoured. Structural (spec Sec5):
    the fix is `.env`, never a retry -- same convention as
    `auth.MissingSharedTokenError`.

    Defence in depth: `Settings`'s own `model_validator` already refuses
    `tracing_enabled=True` without both Langfuse keys at construction time
    (spec Sec11, "refused, not attempted"); this is the barrier for a
    `Settings` built by hand, bypassing that validator, e.g. in a test.
    """


_CONFIGURED: dict[str, Observability] = {}


class _JsonlSpanExporter(SpanExporter):
    """Writes one compact JSON line per exported span to `path` (D8) --
    `ReadableSpan.to_json(indent=None)` already carries `context.trace_id`,
    `context.span_id`, `parent_id`, `name`, `attributes` and
    `resource.attributes["service.name"]`, which is everything the smoke
    test's protocol-trajectory assertion needs to read back after shutdown.
    A `SimpleSpanProcessor`, appended file, no buffering across the export
    call -- five separate processes each hold their own instance.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._path.open("a", encoding="utf-8") as handle:
            for span in spans:
                handle.write(span.to_json(indent=None) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def configure_observability(
    settings: Settings,
    service_name: str,
    *,
    span_exporter: SpanExporter | None = None,
) -> Observability:
    """Install this process's tracer provider and httpx instrumentation, and
    attach Langfuse unless an exporter was injected. Called exactly once per
    process, next to `configure_logging`.

    Parameters
    ----------
    settings : Settings
        `tracing_enabled`, the two `langfuse_*` keys, `langfuse_host` and
        `trace_sample_rate`. `Settings` decides, never the ambient
        environment (spec Sec11).
    service_name : str
        `"main"`, `"search_mcp"`, `"report_mcp"`, `"planner"`,
        `"researcher"` or `"critic"` -- becomes `service.name` on the OTel
        `Resource`, which is what makes one trace readable as six processes.
    span_exporter : SpanExporter, optional
        Injected exporter (an `InMemorySpanExporter` in the gate, a JSONL
        dump under the smoke test). When given, Langfuse is skipped
        entirely and `settings.tracing_enabled` is not consulted -- this is
        the one seam that lets the whole stage criterion be proven offline,
        with no keys, no Docker and no network.

    Returns
    -------
    Observability
        Idempotent per `service_name`, the same contract `configure_logging`
        already keeps: a second call for the same name returns the first
        call's object unchanged, rather than attaching a second span
        processor and exporting every span twice.

    Raises
    ------
    TracingConfigurationError
        `span_exporter` is `None`, `settings.tracing_enabled` is `True` and
        a Langfuse key is missing. Normally unreachable -- see the class
        docstring.
    """
    cached = _CONFIGURED.get(service_name)
    if cached is not None:
        return cached

    if span_exporter is None and settings.span_dump_dir:
        # Activates regardless of `tracing_enabled` -- the same override
        # role `span_exporter` plays in-process, for the smoke test's
        # protocol-trajectory assertion (D8), which cannot use an
        # `InMemorySpanExporter` across five subprocesses.
        span_exporter = _JsonlSpanExporter(
            Path(settings.span_dump_dir) / f"{service_name}.jsonl"
        )

    if span_exporter is None and not settings.tracing_enabled:
        result = Observability(
            service=service_name,
            tracer=trace.get_tracer(service_name),
            tracer_provider=None,
            langfuse=None,
            enabled=False,
        )
        _CONFIGURED[service_name] = result
        return result

    if span_exporter is None and (
        settings.langfuse_public_key is None or settings.langfuse_secret_key is None
    ):
        raise TracingConfigurationError(
            "TRACING_ENABLED is true but LANGFUSE_PUBLIC_KEY/"
            "LANGFUSE_SECRET_KEY are not both set -- refused, not attempted"
        )

    provider = _build_tracer_provider(service_name, settings, span_exporter)
    trace.set_tracer_provider(provider)

    langfuse = (
        None if span_exporter is not None else _attach_langfuse(settings, provider)
    )
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)

    result = Observability(
        service=service_name,
        tracer=provider.get_tracer(service_name),
        tracer_provider=provider,
        langfuse=langfuse,
        enabled=True,
    )
    _CONFIGURED[service_name] = result
    return result


def _build_tracer_provider(
    service_name: str, settings: Settings, span_exporter: SpanExporter | None
) -> TracerProvider:
    """A fresh `TracerProvider` for `service_name`, with `span_exporter`
    attached via a `SimpleSpanProcessor` when given.

    A `SimpleSpanProcessor`, not a `BatchSpanProcessor`: the offline gate
    and the smoke test both read spans back immediately after the traced
    code runs, and a batch processor would make that depend on a flush
    deadline instead of happening synchronously. Langfuse attaches its own
    (batching) processor separately, in `_attach_langfuse`.
    """
    resource = Resource.create(
        {"service.name": service_name, "service.namespace": "ma-systems-hl10"}
    )
    sampler = (
        ALWAYS_ON
        if settings.trace_sample_rate >= 1.0
        else TraceIdRatioBased(settings.trace_sample_rate)
    )
    provider = TracerProvider(resource=resource, sampler=sampler)
    if span_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


def _attach_langfuse(settings: Settings, provider: TracerProvider) -> Any:
    """Construct the process's `Langfuse` client on `provider` -- it adds
    one span processor to a `TracerProvider` this module already owns
    (measured: `Langfuse(tracer_provider=...)` never builds its own unless
    none is registered), and its span processor is itself the OTLP exporter
    to the self-hosted stack, so no separate OTLP wiring is needed."""
    from langfuse import Langfuse

    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
        tracer_provider=provider,
        environment="dev",
    )


def get_tracer(service: str) -> trace.Tracer:
    """The tracer `configure_observability(..., service)` built, or a
    no-op tracer from the ambient global provider if that service was never
    configured in this process."""
    configured = _CONFIGURED.get(service)
    return configured.tracer if configured is not None else trace.get_tracer(service)


def langchain_callbacks() -> list[Any]:
    """The Langfuse LangChain callback handler for LLM-level detail (spec
    Sec11) -- `[]` when tracing is disabled or every configured process ran
    under an injected `span_exporter`, so call sites pass this
    unconditionally into `RunnableConfig["callbacks"]` rather than guarding
    it with an `if`."""
    if not any(configured.langfuse is not None for configured in _CONFIGURED.values()):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def shutdown_observability() -> None:
    """Force-flush every cached provider's span processor.

    OTel's `BatchSpanProcessor` (Langfuse's own) buffers spans in memory;
    without an explicit flush at process shutdown, the last few spans of a
    run can be lost -- the same reason rotating file logs exist as the
    post-mortem artefact (spec Sec12).
    """
    for configured in _CONFIGURED.values():
        if configured.tracer_provider is not None:
            configured.tracer_provider.force_flush()


def asgi_tracing_middleware(service_name: str) -> list[Middleware]:
    """The one-entry ASGI middleware stack every server appends *after* its
    own auth stack (D3): FastMCP already appends caller middleware after
    auth by construction, and A2A's `build_app` is written to match it
    deliberately, so a trace gap on one side means the same thing as on the
    other.

    Parameters
    ----------
    service_name : str
        Unused by the middleware itself (`OpenTelemetryMiddleware` reads
        `service.name` from the ambient `TracerProvider`'s `Resource`
        instead); kept as a parameter so every call site names the service
        it is instrumenting, the same self-documenting shape
        `configure_observability`'s own `service_name` argument has.

    Returns
    -------
    list of starlette.middleware.Middleware
        One entry: `OpenTelemetryMiddleware` with a `server_request_hook`
        that stores the just-started server span's context on the ASGI
        `scope`, under `_SCOPE_PARENT_KEY` -- the channel
        `mcp_tool_span_middleware` reads to parent a tool span onto the
        HTTP request that carried it, without depending on contextvars
        surviving the MCP session's own task boundary (D5).
    """

    def _server_request_hook(span: trace.Span, scope: dict[str, Any]) -> None:
        scope[_SCOPE_PARENT_KEY] = trace.set_span_in_context(span)

    return [
        Middleware(OpenTelemetryMiddleware, server_request_hook=_server_request_hook)
    ]


def _parent_context_for_tool_call() -> Any | None:
    """The OTel context a new `mcp.tool.*` span should nest under, read from
    the per-message HTTP request rather than the ambient contextvar context
    (D5).

    In stateful streamable HTTP, the MCP session's `app.run()` task is
    spawned once, at session creation -- so `trace.get_current_span()`
    inside a tool call is the span of whatever request *created the
    session*, not of this `tools/call`. `langchain-mcp-adapters` opens a
    fresh session per tool invocation, so the two usually coincide, which is
    exactly what would make this bug invisible in a live check.
    `fastmcp.server.dependencies.get_http_request()` is set per JSON-RPC
    message instead, so it stays correct regardless of session topology.
    """
    from fastmcp.server.dependencies import get_http_request

    try:
        request = get_http_request()
    except RuntimeError:
        return None
    stored = request.scope.get(_SCOPE_PARENT_KEY)
    if stored is not None:
        return stored
    return extract(dict(request.headers))


def _tool_call_outcome(result: Any) -> str:
    """`"error"` when `result.is_error`, or its first text block starts with
    `"ERROR:"` -- the convention every tool in `mcp_servers/` already
    returns by (spec Sec5/Sec9); `"ok"` otherwise. No tool in this project
    ever sets `is_error` (`mcp_utils.py`'s own docstring), so in practice
    the prefix is what decides -- reading both means a future tool that does
    set the flag is not silently counted `ok`."""
    if getattr(result, "is_error", False):
        return "error"
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.startswith("ERROR:"):
            return "error"
    return "ok"


def mcp_tool_span_middleware() -> Any:
    """Build the FastMCP server middleware that emits one
    `mcp.tool.<name>` span per tool call, carrying an `outcome` (`ok` vs
    `error`) and a `duration_ms` attribute (D5, the stage's tool-telemetry
    amendment).

    `fastmcp` is imported inside this function, not at module scope:
    `main.py` (the REPL) imports `observability` and must not pay FastMCP's
    import cost for a module it never serves.

    Returns
    -------
    fastmcp.server.middleware.Middleware
        Passed to `mcp.add_middleware(...)` in each MCP server's `main()`.
    """
    from fastmcp.server.middleware import Middleware as FastMCPMiddleware

    class _McpToolSpanMiddleware(FastMCPMiddleware):  # type: ignore[misc]
        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            name = context.message.name
            tracer = trace.get_tracer(__name__)
            started = time.perf_counter()
            with tracer.start_as_current_span(
                f"{_MCP_TOOL_SPAN_PREFIX}{name}",
                context=_parent_context_for_tool_call(),
            ) as span:
                span.set_attribute("mcp.tool.name", name)
                try:
                    result = await call_next(context)
                except Exception as error:
                    span.set_attribute("mcp.tool.outcome", "error")
                    span.set_attribute(
                        "mcp.tool.duration_ms", (time.perf_counter() - started) * 1000
                    )
                    span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                    raise
                span.set_attribute("mcp.tool.outcome", _tool_call_outcome(result))
                span.set_attribute(
                    "mcp.tool.duration_ms", (time.perf_counter() - started) * 1000
                )
                return result

    return _McpToolSpanMiddleware()
