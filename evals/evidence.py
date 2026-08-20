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
from datetime import datetime
from pathlib import Path
from typing import Any

from observability import SPAN_REPL_QUESTION, TRUNCATION_MARKER
from paths import PROJECT_ROOT

_TOOL_SPAN_PREFIX = "mcp.tool."
_LLM_SPAN_PREFIX = "llm."
_AGENT_SPAN_PREFIX = "agent."
_CRITIC_SPAN_NAME = "delegate_to_critic"
_REVISE_VERDICT = "REVISE"
_SAVE_REPORT_TOOL = "save_report"
_ERROR_PREFIX = "ERROR:"


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
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class EvidencePack:
    """Everything the judge is grounded on for one golden case: the tool
    calls the agent made, the model requests it issued, the Critic's
    machine-readable verdict, and the final answer text.

    `verdict`/`gaps` stay `None`/`[]` when the run never reached the Critic
    (e.g. an out-of-scope refusal) -- absence here is itself evidence, read
    by `evals/judge.py`'s requirement-compliance check.

    `latency_ms`/`agent_calls`/`critic_revision_count` (stage 10b, spec
    Sec5.2) are what `evals/metrics.py::CaseMetrics` is derived from --
    `verdict` only ever keeps the *last* `delegate_to_critic` span's
    verdict, so `critic_revision_count` (every REVISE round before it)
    would otherwise be unrecoverable once assembled.

    `saved_report_content` is read from the file on disk, never from the
    `save_report` span's truncated `mcp.tool.input` (stage 10b, fix F1).
    Open coding measured 31 of 35 reports in one sweep exceeding the
    2000-char payload cap, and this system writes its citation line *last*
    -- so grading citations from the span attribute fails cases for
    evidence the agent demonstrably produced. It is also the only place
    two of the sweep's three most serious findings are visible at all: the
    final answer stayed clean while the saved artifact carried a leaked
    system prompt in one run and an injected instruction in another.
    """

    trace_id: str
    tool_calls: list[ToolCall]
    llm_calls: list[LlmCall]
    verdict: str | None
    gaps: list[str] = field(default_factory=list)
    final_answer: str = ""
    latency_ms: float = 0.0
    agent_calls: list[str] = field(default_factory=list)
    critic_revision_count: int = 0
    saved_report_path: str | None = None
    saved_report_content: str = ""


def pack_filename(run_index: int, case_id: str) -> str:
    """`<run_index>__<case_id>.json` -- the exact naming
    `evals.run_eval._persist_pack` writes, and the inverse of
    `parse_pack_filename`."""
    return f"{run_index:02d}__{case_id}.json"


def parse_pack_filename(path: str | Path) -> tuple[int, str]:
    """`<run_index>__<case_id>.json` -> `(run_index, case_id)`."""
    run_index_str, _, case_id = Path(path).stem.partition("__")
    return int(run_index_str), case_id


def pack_from_dict(data: dict[str, Any]) -> EvidencePack:
    """The inverse of `dataclasses.asdict(pack)` -- reconstructs the nested
    `ToolCall`/`LlmCall` dataclasses a persisted pack flattened to plain
    dicts on write.

    Fields added after a pack was written default rather than raising: a
    run set from before fix F1 carries no `saved_report_content`, and
    should score on every item that does not need it instead of failing to
    load. `python -m evals.reassemble <run_dir>` backfills such a set
    offline.
    """
    return EvidencePack(
        trace_id=data["trace_id"],
        tool_calls=[ToolCall(**call) for call in data["tool_calls"]],
        llm_calls=[LlmCall(**call) for call in data["llm_calls"]],
        verdict=data["verdict"],
        gaps=list(data["gaps"]),
        final_answer=data["final_answer"],
        latency_ms=data.get("latency_ms", 0.0),
        agent_calls=list(data.get("agent_calls", [])),
        critic_revision_count=data.get("critic_revision_count", 0),
        saved_report_path=data.get("saved_report_path"),
        saved_report_content=data.get("saved_report_content", ""),
    )


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
    latency_ms = 0.0
    agent_calls: list[str] = []
    critic_revision_count = 0
    saved_report_path: str | None = None

    for jsonl_path in sorted(Path(span_dump_dir).glob("*.jsonl")):
        service = jsonl_path.stem
        for record in _read_jsonl(jsonl_path):
            context = record.get("context") or {}
            if context.get("trace_id") != trace_id:
                continue
            name = record.get("name", "")
            attributes = record.get("attributes") or {}
            if name.startswith(_TOOL_SPAN_PREFIX):
                call = _tool_call(service, attributes)
                tool_calls.append(call)
                if call.name == _SAVE_REPORT_TOOL and call.outcome == "ok":
                    saved_report_path = call.output.strip() or None
            elif name.startswith(_LLM_SPAN_PREFIX):
                llm_calls.append(_llm_call(attributes))
            elif name.startswith(_AGENT_SPAN_PREFIX):
                agent_calls.append(name[len(_AGENT_SPAN_PREFIX) :])
            elif name == _CRITIC_SPAN_NAME:
                verdict = attributes.get("a2a.verdict") or verdict
                if verdict == _REVISE_VERDICT:
                    critic_revision_count += 1
                # `or gaps` here would fall back to the *previous* round's
                # gaps whenever this round's own list is genuinely empty
                # (an empty list is falsy) -- an APPROVE that resolved every
                # gap would then still show the stale, already-fixed ones.
                # `None` (the attribute truly absent) is the only case that
                # should carry the previous value forward.
                span_gaps = attributes.get("a2a.gaps")
                gaps = list(span_gaps) if span_gaps is not None else gaps
            elif name == SPAN_REPL_QUESTION:
                latency_ms = _span_duration_ms(record)

    return EvidencePack(
        trace_id=trace_id,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        verdict=verdict,
        gaps=gaps,
        final_answer=final_answer,
        latency_ms=latency_ms,
        agent_calls=agent_calls,
        critic_revision_count=critic_revision_count,
        saved_report_path=saved_report_path,
        saved_report_content=_read_saved_report(saved_report_path),
    )


def _read_saved_report(path: str | None) -> str:
    """The saved report's full text, or `""` when there is nothing to read.

    `save_report` returns `str(Path(settings.output_dir) / name)`, which is
    relative whenever `OUTPUT_DIR` is -- as `run_eval._settings_for_run`
    makes it for a `--run-dir runs/sweep-01` sweep -- so a relative path is
    resolved against the project root the servers ran from.

    A missing or unreadable file costs this one field, never the pack: a
    run whose output directory was cleaned up still has usable spans, and
    the alternative (raising) would make one absent artefact destroy an
    entire sweep's evidence.
    """
    if path is None or path.startswith(_ERROR_PREFIX):
        return ""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError:
        return ""


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


def _span_duration_ms(record: dict[str, Any]) -> float:
    """`ReadableSpan.to_json()` always carries ISO-8601 `start_time`/
    `end_time` natively (verified against the installed OTel SDK) --
    unlike every other field this module reads, these two were never routed
    through a manually-set span attribute, so they are read from the record
    itself rather than `attributes`. Missing or unparseable timestamps
    yield `0.0`, never a raised exception -- a malformed record must cost
    one field, not the pack (the same tolerance `_read_jsonl` already gives
    a corrupt line)."""
    start = record.get("start_time")
    end = record.get("end_time")
    if not start or not end:
        return 0.0
    try:
        delta = datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))
    except ValueError:
        return 0.0
    return delta.total_seconds() * 1000


def _llm_call(attributes: dict[str, Any]) -> LlmCall:
    return LlmCall(
        role=str(attributes.get("role", "")),
        provider=str(attributes.get("llm.provider", "")),
        model=str(attributes.get("llm.model", "")),
        duration_ms=float(attributes.get("llm.duration_ms", 0.0)),
        cost_usd=attributes.get("llm.cost_usd"),
        input_tokens=int(attributes.get("llm.tokens.input", 0)),
        output_tokens=int(attributes.get("llm.tokens.output", 0)),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse one JSONL span dump, tolerating a corrupt line -- concurrent
    writes across the run's processes can interleave partial lines (measured
    `JSONDecodeError` from exactly this), and one bad line must
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
