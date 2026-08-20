"""Evaluation apparatus: golden dataset, evidence extraction, LLM judge, the
auto-approve HITL harness and the dataset runner (stage 10a).

Layering (spec Sec3): `evals/` may import anything --
it reaches both the REPL entry point and the servers, which application code
never does. Nothing outside `evals/` may import it: the evaluation apparatus
must never become load-bearing for the system it measures.
"""

from __future__ import annotations
