"""Human labels for judge validation (spec Sec13.6/Sec5.6, decision D5):
`(run_index, case_id)` rows using the same `PASS`/`FAIL`/`None` vocabulary
the judge's own `RubricItem` uses, plus a free-text note.

The pre-registration guard is the whole point of this module. Cohen's kappa
(`evals/statistics.py::cohens_kappa`) only measures what it claims to measure
if the human labels were written *before* the judge's verdicts existed for
that run -- a human who labels after seeing the judge's output is grading the
judge's homework with its own answer key open. `labelled_before_scoring` is
the human's own explicit attestation, required (not merely defaulted) so a
label file cannot silently pass this check by omission; `evals/score.py`
layers an mtime check on top, comparing the label file against any scores
file that already exists for the same run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

_STRICT_CONFIG = ConfigDict(extra="forbid")

_Verdict = Literal["PASS", "FAIL"] | None


class LabelValidationError(Exception):
    """A label file fails structural validation, is missing the
    pre-registration attestation, or references a case/run_index pair the
    target run never produced."""


class HumanLabel(BaseModel):
    """One person's rating of one `(run_index, case_id)` pair, one field per
    rubric item -- the same vocabulary `evals.judge.RubricItem.verdict`
    uses, so `cohens_kappa` compares like with like."""

    model_config = _STRICT_CONFIG

    case_id: str
    run_index: int
    answer_relevancy: _Verdict = None
    requirement_compliance: _Verdict = None
    citation_reliability: _Verdict = None
    injection_resistance: _Verdict = None
    harmful_compliance: _Verdict = None
    note: str = ""


class HumanLabelSet(BaseModel):
    """The whole label file: which run it labels, the pre-registration
    attestation, and every row.

    `labelled_before_scoring` has no default -- a label file that never
    addresses the question is refused exactly like one that answers it
    honestly `False`, since either way the ordering this module exists to
    prove has not been established.
    """

    model_config = _STRICT_CONFIG

    run_name: str
    labelled_before_scoring: bool
    labels: list[HumanLabel]


def load_labels(path: str | Path) -> HumanLabelSet:
    """Parse and validate one human-label YAML file.

    Raises
    ------
    LabelValidationError
        The file fails schema validation, or `labelled_before_scoring` is
        present but `False` -- in both cases the pre-registration ordering
        this module exists to enforce is not established.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        label_set = HumanLabelSet.model_validate(raw)
    except ValidationError as error:
        raise LabelValidationError(str(error)) from error
    if not label_set.labelled_before_scoring:
        raise LabelValidationError(
            "labelled_before_scoring must be true -- labels written after "
            "the judge's verdicts already exist measure nothing about the "
            "judge's agreement with an independent human rating"
        )
    return label_set


def validate_against_run(label_set: HumanLabelSet, run_dir: str | Path) -> None:
    """Refuse a label set naming a `(run_index, case_id)` pair the run at
    `run_dir` never actually produced a pack for.

    Parameters
    ----------
    run_dir : str or Path
        A run directory holding `packs/<run_index>__<case_id>.json` files
        (the naming `evals.run_eval._persist_pack` writes) -- checked
        directly against the filesystem rather than trusting the label
        file's own claims about what it labels.

    Raises
    ------
    LabelValidationError
        Names every missing pair, not just the first -- a human correcting
        a label file benefits from the whole list at once.
    """
    packs_dir = Path(run_dir) / "packs"
    existing = (
        {path.stem for path in packs_dir.glob("*.json")}
        if packs_dir.exists()
        else set()
    )
    missing = [
        f"{label.run_index:02d}__{label.case_id}"
        for label in label_set.labels
        if f"{label.run_index:02d}__{label.case_id}" not in existing
    ]
    if missing:
        raise LabelValidationError(
            f"label(s) reference (run_index, case_id) pair(s) not found "
            f"under {packs_dir}: {', '.join(missing)}"
        )
