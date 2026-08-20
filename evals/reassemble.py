"""Rebuild a completed sweep's evidence packs from its own span dump
(stage 10b, fix F1).

A fix to `evals.evidence.assemble` has to reach traces that were already
paid for. Everything `assemble` needs survives the run: `spans/*.jsonl` per
service, the saved report files under `output/`, and the per-case
`trace_id`/`final_answer` in `results.jsonl`. So the packs can be
regenerated offline and for free, byte-identical to what a fresh run would
have written -- rather than re-running a two-hour, real-money sweep to pick
up a field.

```
python -m evals.reassemble runs/sweep-01
```

This rewrites `packs/`. It never touches `results.jsonl`, `spans/` or the
saved reports, so it can be run again after any later `evidence.py` change.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from evals import evidence


class ReassembleError(Exception):
    """The named directory is not a completed run -- no `results.jsonl`."""


def reassemble_run(run_dir: str | Path) -> int:
    """Rewrite every pack under `run_dir/packs` from `run_dir/spans`.

    Parameters
    ----------
    run_dir : str or Path
        A run directory produced by `evals.run_eval.run_dataset`.

    Returns
    -------
    int
        How many packs were written. A result line with no `trace_id` (a
        case that crashed before producing one) is skipped rather than
        written as an empty pack -- an evidence-free pack would later read
        as a real run that simply did nothing.

    Raises
    ------
    ReassembleError
        `run_dir/results.jsonl` does not exist.
    """
    run_path = Path(run_dir)
    results_path = run_path / "results.jsonl"
    if not results_path.exists():
        raise ReassembleError(
            f"no results.jsonl under {run_path} -- not a run directory"
        )

    span_dump_dir = run_path / "spans"
    packs_dir = run_path / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        trace_id = result.get("trace_id")
        if not trace_id:
            continue
        pack = evidence.assemble(
            span_dump_dir, trace_id, final_answer=result.get("final_answer", "")
        )
        run_index = int(result.get("run_index", 1))
        target = packs_dir / f"{run_index:02d}__{result['case_id']}.json"
        target.write_text(json.dumps(asdict(pack)), encoding="utf-8")
        written += 1
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a run's evidence packs from its own span dump"
    )
    parser.add_argument("run_dir", help="Path to the run directory, e.g. runs/sweep-01")
    args = parser.parse_args(argv)

    written = reassemble_run(args.run_dir)
    print(f"[system] {written} pack(s) rebuilt under {Path(args.run_dir) / 'packs'}")


if __name__ == "__main__":
    main(sys.argv[1:])
