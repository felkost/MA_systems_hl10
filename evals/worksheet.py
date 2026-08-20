"""A verdict-free labelling worksheet (spec Sec13.6, decision D2, stage
10c): the exact evidence the judge is grounded on, rendered for a human to
label blind to the judge's *own output* -- never blind to the Critic's
verdict, which is legitimate evidence the judge itself reads.

This stage's own adversarial review found the first draft of this module
aimed at the wrong field: `EvidencePack.verdict` is the *Critic's*
APPROVE/REVISE verdict, an in-system artefact the judge is deliberately
shown (`evals.judge.render_evidence`) and the existing human labels already
cite. What must never reach a labeller meant to be blind to it is the
*judge's* output -- the five rubric item names and `offending_span`, which
live only in `scores-*.jsonl`/`summary-*.json`, files this module never
imports and never opens. The worksheet reuses `render_evidence` verbatim, so
the labeller sees byte-for-byte what the judge saw.

```
python -m evals.worksheet --run-set runs/sweep-01 --run-index 2 --run-index 3
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from evals.dataset import GoldenCase, load_all
from evals.evidence import EvidencePack, pack_from_dict, parse_pack_filename
from evals.judge import render_evidence

# The judge's own vocabulary (evals/judge.py::JudgeVerdict field names, plus
# the one free-text field a FAIL is grounded on) -- never the Critic's
# APPROVE/REVISE, which is legitimate evidence and deliberately present.
_JUDGE_OUTPUT_TOKENS = (
    "answer_relevancy",
    "requirement_compliance",
    "citation_reliability",
    "injection_resistance",
    "harmful_compliance",
    "offending_span",
)


class WorksheetError(Exception):
    """A requested `(run_index, case_id)` pair has no matching pack or case,
    or the rendered text would expose judge output to a labeller meant to
    be blind to it."""


@dataclass(frozen=True)
class WorksheetEntry:
    """One case-run a labeller will see: the case it answers and the
    evidence pack it produced."""

    run_index: int
    case_id: str
    case: GoldenCase
    pack: EvidencePack


def load_entries(
    run_dir: Path,
    cases: Sequence[GoldenCase],
    pairs: Sequence[tuple[int, str]],
) -> list[WorksheetEntry]:
    """Build one `WorksheetEntry` per `(run_index, case_id)` pair, reading
    the matching pack from `run_dir / "packs"`.

    Deliberately pair-driven rather than a `run_index` filter: this is the
    one code path both the CLI (rendering a fresh worksheet for a labelling
    pass) and `evals.score`'s pre-registration guard (re-rendering the exact
    worksheet a given label file claims to be labelled from) go through, and
    the guard only ever knows the pairs a label file names, not which
    `run_index`es exist on disk.

    Raises
    ------
    WorksheetError
        A pair names a pack that does not exist on disk, or a case id not
        present in `cases`.
    """
    resolved_cases = {case.id: case for case in cases}
    packs_dir = run_dir / "packs"
    entries: list[WorksheetEntry] = []
    for run_index, case_id in sorted(set(pairs)):
        path = packs_dir / f"{run_index:02d}__{case_id}.json"
        if not path.exists():
            raise WorksheetError(
                f"no pack at {path} for (run_index={run_index}, case_id={case_id!r})"
            )
        case = resolved_cases.get(case_id)
        if case is None:
            raise WorksheetError(
                f"pack {path.name} references unknown case id {case_id!r}"
            )
        pack = pack_from_dict(json.loads(path.read_text(encoding="utf-8")))
        entries.append(
            WorksheetEntry(run_index=run_index, case_id=case_id, case=case, pack=pack)
        )
    return entries


def pairs_for_run_indices(
    run_dir: Path, run_indices: Sequence[int]
) -> list[tuple[int, str]]:
    """Every `(run_index, case_id)` pair on disk under `run_dir / "packs"`
    whose `run_index` is in `run_indices` -- what the CLI resolves
    `--run-index` against."""
    packs_dir = run_dir / "packs"
    if not packs_dir.exists():
        return []
    wanted = set(run_indices)
    return sorted(
        pair
        for pair in (parse_pack_filename(path) for path in packs_dir.glob("*.json"))
        if pair[0] in wanted
    )


def render_worksheet(entries: Sequence[WorksheetEntry]) -> str:
    """Render `entries` as one Markdown document, sorted by
    `(run_index, case_id)` so the same entry set always renders
    byte-identical text regardless of input order -- what makes a
    `worksheet_digest` reproducible from the packs alone.

    Raises
    ------
    WorksheetError
        The rendered text contains a judge-output token. This can only fire
        if a future change to `render_evidence` starts rendering
        judge-shaped content; the check exists to catch that change here,
        at the one place a labeller would otherwise see it, not because
        today's evidence is expected to trip it.
    """
    sections = [
        f"## run {entry.run_index:02d} -- {entry.case_id}\n\n"
        f"Question: {entry.case.question}\n\n"
        f"{render_evidence(entry.pack)}"
        for entry in sorted(entries, key=lambda e: (e.run_index, e.case_id))
    ]
    text = "\n\n".join(sections)
    _assert_no_judge_output(text)
    return text


def _assert_no_judge_output(text: str) -> None:
    lowered = text.lower()
    found = [token for token in _JUDGE_OUTPUT_TOKENS if token in lowered]
    if found:
        raise WorksheetError(
            f"worksheet text contains judge-output token(s) {found} -- a "
            "labelling worksheet must expose only what the judge was given, "
            "never what the judge said"
        )


def worksheet_digest(text: str) -> str:
    """SHA-256 hex digest of `text` -- what a `HumanLabelSet.worksheet_digest`
    is checked against in `evals.labels.verify_worksheet`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render a verdict-free labelling worksheet for independent "
        "labelling"
    )
    parser.add_argument(
        "--run-set", required=True, help="Path to the run(-set) directory"
    )
    parser.add_argument(
        "--run-index",
        action="append",
        type=int,
        required=True,
        help="Repeatable: which run_index(es) to render, e.g. "
        "--run-index 2 --run-index 3",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path; defaults to <run-set>/labelling-worksheet-<selector>.md",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_set)
    pairs = pairs_for_run_indices(run_dir, args.run_index)
    entries = load_entries(run_dir, load_all(), pairs)
    text = render_worksheet(entries)
    selector = "-".join(str(i) for i in sorted(set(args.run_index)))
    out_path = (
        Path(args.out) if args.out else run_dir / f"labelling-worksheet-{selector}.md"
    )
    out_path.write_text(text, encoding="utf-8")
    digest = worksheet_digest(text)
    print(f"[system] {len(entries)} entr(ies) rendered -- {out_path}")
    print(f"[system] worksheet_digest: {digest}")


if __name__ == "__main__":
    main(sys.argv[1:])
