"""The REPL's three-choice UX over the installed `HumanInTheLoopMiddleware`
decision contract (stage 7).

**D6 -- the assignment's `edit` payload does not match this contract.**
`docs/task-hl10.md` implies an `edit` decision carries an edited action; the
installed `EditDecision` requires `edited_action={"name": ..., "args": ...}`,
which a human's free-text feedback string is not, and sending it as-is
raises `KeyError: 'name'` inside `HumanInTheLoopMiddleware._process_decision`.
`edit` is mapped instead onto a real `reject` decision carrying the
feedback plus a revision instruction as the message: the tool call does not
execute, the model receives an error `ToolMessage` with that content and
revises. Ported from the hl8 donor's own documented deviation
(`../MA_systems_hl8_project/MA_system_hl8/hitl.py`), re-verified against the
installed LangChain 1.3.15's real TypedDicts rather than carried forward
unmeasured.

**D7 -- narrower than the donor's.** hl8 exported `build_interrupt_request`
because it had a second, hand-built graph path whose payload had to be
indistinguishable from the middleware's own. hl10 has exactly one path
(`HumanInTheLoopMiddleware` on the Supervisor), so that builder is not
ported.

**Layering (pinned by tests/test_layering.py):** application layer --
imported by `main.py` and by `middleware.SaveReportGuardMiddleware`'s retry
text, never the reverse.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    Decision,
    HITLRequest,
    RejectDecision,
)
from langchain_core.messages import BaseMessage, ToolMessage

ReplChoice = Literal["approve", "edit", "reject"]

_ARG_PREVIEW_LENGTH = 300
_EDIT_INSTRUCTION = "Revise the report and call save_report again."


def build_decision(choice: ReplChoice, text: str = "") -> Decision:
    """Turn one REPL choice into a real `HumanInTheLoopMiddleware` decision.

    Parameters
    ----------
    choice : {"approve", "edit", "reject"}
    text : str, default ""
        The feedback (`edit`) or reason (`reject`) the human typed, if any.

    Returns
    -------
    Decision

    Raises
    ------
    ValueError
        `choice` is none of the three real choices.

    Notes
    -----
    Singular, not a list: the interrupt can carry more than one gated call
    in one turn (`HITLRequest.action_requests`), so the caller prompts once
    per action request and assembles the list itself -- see
    `main._resolve_interrupt`.
    """
    if choice == "approve":
        return {"type": "approve"}
    if choice == "edit":
        feedback = text.strip()
        message = f"{feedback} {_EDIT_INSTRUCTION}" if feedback else _EDIT_INSTRUCTION
        return {"type": "reject", "message": message}
    if choice == "reject":
        reason = text.strip()
        decision: RejectDecision = {"type": "reject"}
        if reason:
            decision["message"] = reason
        return decision
    raise ValueError(f"Unknown HITL choice: {choice!r} -- expected approve/edit/reject")


def render_interrupt(request: HITLRequest) -> str:
    """Render every gated action request's real arguments for the REPL.

    Parameters
    ----------
    request : HITLRequest

    Returns
    -------
    str

    Notes
    -----
    Injection defence layer 4 (spec Sec9): shows `action_requests[i]["args"]`
    verbatim, truncated for readability only -- never summarised, never
    taken from the model's narration of the call.
    """
    return "\n\n".join(_render_action(action) for action in request["action_requests"])


def _render_action(action: ActionRequest) -> str:
    lines = [f"Tool:  {action['name']}"]
    for key, value in action["args"].items():
        lines.append(f"  {key}: {_preview(value)}")
    return "\n".join(lines)


def _preview(value: object) -> str:
    text = str(value)
    if len(text) <= _ARG_PREVIEW_LENGTH:
        return text
    return text[:_ARG_PREVIEW_LENGTH] + "..."


def render_save_status(outcomes: Sequence[BaseMessage]) -> str:
    """State what actually reached disk this turn, from the tool result.

    Parameters
    ----------
    outcomes : Sequence of BaseMessage
        Every message this turn produced.

    Returns
    -------
    str

    Notes
    -----
    D8: hl8 measured a real session where the model's closing text claimed
    "I have saved the report" after two human rejections, with nothing on
    disk. The truth source is the `save_report` `ToolMessage`'s own
    content -- never an `AIMessage`'s prose -- and `report_mcp.save_report`
    never raises, so a failed write is recognised by its `ERROR:` prefix,
    the same convention every tool in this project uses.

    A tool loaded through `langchain-mcp-adapters` (the real
    `save_report_tool`, unlike every offline test's plain-string `@tool`
    fake) returns `content` as a **list of content blocks**
    (`[{"type": "text", "text": "...", "id": "..."}]`), not a bare string --
    found only by the stage-7 live check, where the status line printed the
    raw block list instead of the path. `.text` is `BaseMessage`'s own
    accessor for exactly this: a `str` subclass that flattens both shapes.

    A `reject` (or an `edit` mapped onto one, D6) never runs the tool at
    all: `HumanInTheLoopMiddleware` builds the `ToolMessage` itself, from
    the human's own reason text, with **`status="error"`** -- also found
    live, where a rejection's free-text reason (no `ERROR:` prefix) was
    printed as if it were the saved path. `status` is checked first, before
    the `ERROR:` prefix that only distinguishes a genuine tool-level
    failure (the tool ran, `report_mcp.save_report` never raises) from a
    real success.
    """
    for message in reversed(outcomes):
        if isinstance(message, ToolMessage) and message.name == "save_report":
            if message.status == "error":
                return "[system] No report was saved this turn."
            content = message.text
            if content.startswith("ERROR:"):
                return "[system] No report was saved this turn."
            return f"Report saved to: {content}"
    return "[system] No report was saved this turn."
