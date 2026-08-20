"""The golden dataset: a Pydantic model plus a loader for
`evals/golden/*.yaml` (spec Sec13, decision D7).

Every optional expectation reads `None`, never folded into `False` -- a case
that says nothing about citations abstains the `citation reliability` check
rather than measuring a failure it never claimed, the same discipline an
earlier iteration's own evaluators used (`score: bool | None`, never `0`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

import paths

_GOLDEN_DIR_NAME = "golden"

_STRICT_CONFIG = ConfigDict(extra="forbid")


class DuplicateCaseIdError(Exception):
    """A golden case id repeats across slice files.

    The id is the uploader's upsert key and the Langfuse dataset item id --
    it must be unique across the whole dataset, not just within one slice.
    """


class Expectations(BaseModel):
    """What one golden case expects, checked by the judge's grounding
    (`evals/judge.py`) and the runner's structural checks.

    Every field is optional: an absent field means the corresponding check
    abstains, never that it measures a failure.
    """

    model_config = _STRICT_CONFIG

    verdict: Literal["APPROVE", "REVISE"] | None = None
    out_of_scope: bool | None = None
    must_cite: bool | None = None
    refuses_injection: bool | None = None


class GoldenCase(BaseModel):
    """One row of the golden dataset (spec Sec13): a question, why it is in
    the set, and what a correct run of it looks like."""

    model_config = _STRICT_CONFIG

    id: str
    question: str
    rationale: str
    expects: Expectations = Field(default_factory=Expectations)
    fixture: str | None = Field(
        default=None,
        description=(
            "Filename under evals/fixtures/ this case needs served locally "
            "(D5, the adversarial slice's indirect-injection case). When "
            "set, `question` carries the literal placeholder `{fixture_url}` "
            "wherever the served URL belongs -- `evals/run_eval.py` "
            "substitutes it with the fixture server's actual runtime URL "
            "before driving the case, since the port is only known once "
            "the run starts."
        ),
    )
    needs_poisoned_index: bool = Field(
        default=False,
        description=(
            "True for the adversarial slice's poisoned-knowledge-base case "
            "(D5): `evals/run_eval.py` builds a throwaway index from the "
            "real corpus plus evals/fixtures/poisoned_document.md before "
            "booting the stack, never touching data/ or index/ at the "
            "repository root."
        ),
    )


def golden_dir() -> Path:
    """The repository's tracked golden-dataset directory, `evals/golden/` --
    the default `load_all` reads from a real checkout."""
    return paths.PROJECT_ROOT / "evals" / _GOLDEN_DIR_NAME


def load_slice(path: str | Path) -> list[GoldenCase]:
    """Parse one slice file into typed cases.

    Parameters
    ----------
    path : str or Path
        A YAML file holding a list of case mappings.

    Returns
    -------
    list[GoldenCase]

    Raises
    ------
    pydantic.ValidationError
        A case is missing a required field or carries an unknown one --
        fails at load time, not deep inside the runner or the judge.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [GoldenCase.model_validate(row) for row in raw]


def load_all(directory: str | Path | None = None) -> list[GoldenCase]:
    """Parse every `*.yaml` slice under `directory` into one case list.

    Parameters
    ----------
    directory : str or Path, optional
        Defaults to `golden_dir()`. Explicit so a test can point at a
        `tmp_path`, the same shape `paths.manifest_path` uses for
        `index_dir`.

    Raises
    ------
    DuplicateCaseIdError
        The same case id appears in more than one slice file -- it is the
        uploader's upsert key and must be unique across the whole dataset.
    """
    resolved = Path(directory) if directory is not None else golden_dir()
    cases: list[GoldenCase] = []
    seen: dict[str, Path] = {}
    for slice_path in sorted(resolved.glob("*.yaml")):
        for case in load_slice(slice_path):
            if case.id in seen:
                raise DuplicateCaseIdError(
                    f"case id {case.id!r} appears in both {seen[case.id]} "
                    f"and {slice_path}"
                )
            seen[case.id] = slice_path
            cases.append(case)
    return cases
