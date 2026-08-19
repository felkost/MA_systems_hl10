"""The auto-approve HITL test double (spec Sec13, decision D0): substitutes
`main._resolve_interrupt` so a dataset run does not require a human at every
example.

Caller-side only -- nothing in `supervisor.py`, `middleware.py` or `hitl.py`
changes; the production HITL gate keeps exactly one behaviour. Not a
configuration flag on the production gate (the plan's own risk note): the
production loop never imports this module, so there is no code path that
lets a dataset run silently weaken the real gate.
"""

from __future__ import annotations

from langchain.agents.middleware.human_in_the_loop import Decision, HITLRequest

import hitl


def auto_approve(request: HITLRequest) -> list[Decision]:
    """Approve every gated action request in one interrupt.

    Mirrors `main._resolve_interrupt`'s own seam exactly -- one decision per
    `request["action_requests"]` entry, built through `hitl.build_decision`,
    never a hand-built dict -- so a change to the decision contract breaks
    this loudly instead of letting it drift into a shape production no
    longer speaks.

    Parameters
    ----------
    request : HITLRequest

    Returns
    -------
    list of Decision
        Length equals `len(request["action_requests"])` -- the installed
        `HumanInTheLoopMiddleware` length-checks the resume list against the
        number of interrupted calls (stage-7 measurement); a double that
        always returns exactly one decision is wrong the moment the model
        emits two gated calls in one message.
    """
    return [hitl.build_decision("approve") for _ in request["action_requests"]]
