"""`EvidencePack`: what the judge (`evals/judge.py`) is allowed to see (spec
Sec13.3, decisions D3/D4).

Amends spec Sec13.3's original design: the tool-payload source is the
`SPAN_DUMP_DIR` JSONL dump, not "Langfuse observations recorded by the
LangChain handler" -- stage 9 measured that handler runs in the Supervisor
process alone, so a Langfuse-sourced pack would carry no evidence at all for
any sub-agent tool call. The scoping discipline below (one trace id, per-line
JSONL tolerance) is the same one stage 9's own smoke test already applies to
assert span structure; this module generalises it into a payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observability import SPAN_REPL_QUESTION, TRUNCATION_MARKER

_TOOL_SPAN_PREFIX = "mcp.tool."
_LLM_SPAN_PREFIX = "llm."
_CRITIC_SPAN_NAME = "delegate_to_critic"


@dataclass(frozen=True)
class ToolCall:
    """One `mcp.tool.*` span -- the evidence the agent actually saw."""

    service: str
    name: str
    input: str
    output: str
    outcome: str
    duration_ms: float
    input_truncated: bool
    output_truncated: bool


@dataclass(frozen=True)
class LlmCall:
    """One `llm.<role>` span -- one actual provider request, never a whole
    retry storm collapsed into one (`LlmSpanMiddleware` is innermost by
    construction)."""

    role: str
    provider: str
    model: str
    duration_ms: float
    cost_usd: float | None


@dataclass(frozen=True)
class EvidencePack:
    """Everything the judge is grounded on for one golden case: the tool
    calls the agent made, the model requests it issued, the Critic's
    machine-readable verdict, and the final answer text.

    `verdict`/`gaps` stay `None`/`[]` when the run never reached the Critic
    (e.g. an out-of-scope refusal) -- absence here is itself evidence, read
    by `evals/judge.py`'s requirement-compliance check.
    """

    trace_id: str
    tool_calls: list[ToolCall]
    llm_calls: list[LlmCall]
    verdict: str | None
    gaps: list[str] = field(default_factory=list)
    final_answer: str = ""


def find_case_trace_id(span_dump_dir: str | Path, thread_id: str) -> str | None:
    """Locate the trace id of the `repl.question` root span carrying
    `thread_id` -- the seam connecting the runner's own per-case
    `thread_id` to the trace id every other span in the case shares.

    Returns
    -------
    str or None
        `None` if `main.jsonl` carries no matching root span (yet, or the
        run wrote to a different `SPAN_DUMP_DIR`).
    """
    for record in _read_jsonl(Path(span_dump_dir) / "main.jsonl"):
        if record.get("name") != SPAN_REPL_QUESTION:
            continue
        if record.get("attributes", {}).get("thread_id") == thread_id:
            return str(record["context"]["trace_id"])
    return None


def assemble(
    span_dump_dir: str | Path, trace_id: str, *, final_answer: str = ""
) -> EvidencePack:
    """Build the `EvidencePack` for one case from every `*.jsonl` file under
    `span_dump_dir`, keeping only spans that share `trace_id`.

    Parameters
    ----------
    span_dump_dir : str or Path
        The run's `SPAN_DUMP_DIR` -- one `<service>.jsonl` file per process.
    trace_id : str
        The case's root trace id, from `find_case_trace_id`.
    final_answer : str, default ""
        The runner's own captured final answer text -- not reconstructed
        from spans, since the answer text itself never becomes a span
        attribute.

    Notes
    -----
    Scoping by trace id alone already excludes preflight's own traces
    (three Agent Card GETs, two MCP resource reads never share the
    question's trace id) -- the same effect stage 9's smoke test gets by
    naming the span families it asserts on.
    """
    tool_calls: list[ToolCall] = []
    llm_calls: list[LlmCall] = []
    verdict: str | None = None
    gaps: list[str] = []

    for jsonl_path in sorted(Path(span_dump_dir).glob("*.jsonl")):
        service = jsonl_path.stem
        for record in _read_jsonl(jsonl_path):
            context = record.get("context") or {}
            if context.get("trace_id") != trace_id:
                continue
            name = record.get("name", "")
            attributes = record.get("attributes") or {}
            if name.startswith(_TOOL_SPAN_PREFIX):
                tool_calls.append(_tool_call(service, attributes))
            elif name.startswith(_LLM_SPAN_PREFIX):
                llm_calls.append(_llm_call(attributes))
            elif name == _CRITIC_SPAN_NAME:
                verdict = attributes.get("a2a.verdict") or verdict
                gaps = list(attributes.get("a2a.gaps") or gaps)

    return EvidencePack(
        trace_id=trace_id,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        verdict=verdict,
        gaps=gaps,
        final_answer=final_answer,
    )


def _tool_call(service: str, attributes: dict[str, Any]) -> ToolCall:
    input_text = str(attributes.get("mcp.tool.input", ""))
    output_text = str(attributes.get("mcp.tool.output", ""))
    return ToolCall(
        service=service,
        name=str(attributes.get("mcp.tool.name", "")),
        input=input_text,
        output=output_text,
        outcome=str(attributes.get("mcp.tool.outcome", "")),
        duration_ms=float(attributes.get("mcp.tool.duration_ms", 0.0)),
        input_truncated=input_text.endswith(TRUNCATION_MARKER),
        output_truncated=output_text.endswith(TRUNCATION_MARKER),
    )


def _llm_call(attributes: dict[str, Any]) -> LlmCall:
    return LlmCall(
        role=str(attributes.get("role", "")),
        provider=str(attributes.get("llm.provider", "")),
        model=str(attributes.get("llm.model", "")),
        duration_ms=float(attributes.get("llm.duration_ms", 0.0)),
        cost_usd=attributes.get("llm.cost_usd"),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse one JSONL span dump, tolerating a corrupt line -- concurrent
    writes across the run's processes can interleave partial lines (hl8
    measured `JSONDecodeError` from exactly this), and one bad line must
    cost one span, not the run."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
