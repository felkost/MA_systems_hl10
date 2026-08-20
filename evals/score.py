"""The offline scorer (spec Sec5.5, decision D1): reads packs
`evals/run_eval.py` already persisted to disk, judges each, and writes
`scores-<version>.jsonl` + `summary-<version>.json` -- no servers boot, no
system-under-test spend, the judge model is the only thing reached.

This is the seam D1 exists to create: running (`evals/run_eval.py`) and
judging are separate passes, so a run-set's 36 packs are a *fixed corpus*
that the rubric revision `evals/judge.py`'s `JUDGE_RUBRIC_VERSION` names
(spec Sec5.4) can be applied to more than once, without paying for another
live sweep.

```
python -m evals.score --run-set runs/<name>
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel

from config import Settings, load_settings
from evals import judge as judge_module
from evals import statistics
from evals import worksheet as worksheet_module
from evals.dataset import GoldenCase, load_all
from evals.evidence import parse_pack_filename, pack_from_dict
from evals.labels import (
    HumanLabelSet,
    load_labels,
    validate_against_run,
    verify_worksheet,
)
from evals.metrics import metrics_of
from evals import pricing
from paths import PROJECT_ROOT

_RUBRIC_ITEMS = (
    "answer_relevancy",
    "requirement_compliance",
    "citation_reliability",
    "injection_resistance",
    "harmful_compliance",
)
_DEFAULT_LABEL_PATH = PROJECT_ROOT / "evals" / "labels" / "human_labels.yaml"


class ScoringError(Exception):
    """A run-set cannot be scored as given -- no packs, or a pack names a
    case the supplied dataset does not contain."""


class RubricVersionMismatchError(ScoringError):
    """`--rubric-version` names a version that is not the one currently
    active in `evals/judge.py` -- there is exactly one live rubric at a
    time (spec Sec5.4); switching versions is a code change to `judge.py`,
    citing the taxonomy category that motivated it, never a runtime flag
    selecting a stored variant."""


@dataclass
class CaseScore:
    """One case's judged outcome -- one line of `scores-<version>.jsonl`
    (or `scores-<version>-p<N>.jsonl` under `passes > 1`, stage 10c)."""

    run_index: int
    case_id: str
    verdict: dict[str, Any]
    metrics: dict[str, Any]
    pass_index: int = 1

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class CaseScoreError:
    """One (run_index, case_id) pair that could not be judged in one pass --
    recorded instead of aborting the batch (live-found 2026-08-20:
    gemini-2.5-pro returned an empty response for a pack whose saved report
    was itself jailbreak/prompt-disclosure content, and the JSON parse
    failure that followed crashed the whole run before any of the other 35
    cases had a chance to be scored)."""

    run_index: int
    case_id: str
    error: str
    pass_index: int = 1


@dataclass
class ScoringResult:
    scores: list[CaseScore]
    errors: list[CaseScoreError]
    summary: dict[str, Any]
    scores_path: Path
    """The first (or only) pass's scores file -- kept for callers written
    before `passes` existed. See `scores_paths` for every pass."""
    summary_path: Path
    scores_paths: list[Path] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_aggregates(scores: list[CaseScore]) -> dict[str, Any]:
    metrics_list = [score.metrics for score in scores]
    costs = [m["cost_usd"] for m in metrics_list]
    return {
        "n_cases": len(metrics_list),
        "mean_latency_ms": _mean([m["latency_ms"] for m in metrics_list]),
        "mean_tokens_in": _mean([m["tokens_in"] for m in metrics_list]),
        "mean_tokens_out": _mean([m["tokens_out"] for m in metrics_list]),
        "total_cost_usd": None if any(cost is None for cost in costs) else sum(costs),
        "mean_tool_calls": _mean([m["tool_calls"] for m in metrics_list]),
        "mean_delegations": _mean([m["delegations"] for m in metrics_list]),
        "mean_critic_revisions": _mean([m["critic_revisions"] for m in metrics_list]),
    }


def _load_matching_labels(
    label_path: Path,
    *,
    run_name: str,
    run_dir: Path,
    cases: Sequence[GoldenCase],
    prior_scores_mtime: float | None,
) -> tuple[HumanLabelSet | None, str | None]:
    """Resolve one human-label set to score agreement against, or the
    reason none applies.

    Two pre-registration checks, in order (stage 10c, decision D2 --
    replacing a single mtime check that a same-day comment edit to this
    project's own tracked label file proved could invalidate a label set by
    accident): a label file carrying `worksheet_digest` is verified against
    a fresh re-render of the exact worksheet it claims to be labelled from
    (`evals.labels.verify_worksheet` -- raises on a mismatch, propagated,
    not soft-skipped: a wrong digest is a structural problem with the file,
    the same class as a missing attestation). A label file with no digest
    (every file predating stage 10c) falls back to the original mtime
    check -- modified after a *prior* scores file for this run/version
    already existed, soft-skipped with a reason, since mtime alone cannot
    distinguish "labelled after reading the verdicts" from "someone edited
    a comment".

    Returns
    -------
    (label_set, skip_reason) : tuple
        Exactly one is non-`None`. `label_set` is only returned when the
        label file exists, names this exact run, passes whichever guard
        applies, and validates against the packs actually on disk.
    """
    if not label_path.exists():
        return None, None
    label_set = load_labels(label_path)  # raises LabelValidationError, not caught here
    if label_set.run_name != run_name:
        return None, f"label file labels run {label_set.run_name!r}, not {run_name!r}"
    if label_set.worksheet_digest is not None:
        pairs = [(label.run_index, label.case_id) for label in label_set.labels]
        entries = worksheet_module.load_entries(run_dir, cases, pairs)
        text = worksheet_module.render_worksheet(entries)
        verify_worksheet(label_set, text)  # raises LabelValidationError, propagated
    elif (
        prior_scores_mtime is not None
        and label_path.stat().st_mtime > prior_scores_mtime
    ):
        return None, (
            "label file was modified after a scores file for this run/version "
            "already existed, and carries no worksheet_digest to check instead "
            "-- labels must be written before scoring, not after"
        )
    validate_against_run(label_set, run_dir)
    return label_set, None


def _kappa_inputs(
    label_set: HumanLabelSet, scores: list[CaseScore], item: str
) -> tuple[list[str | None], list[str | None]]:
    scores_by_key = {(score.run_index, score.case_id): score for score in scores}
    judge_values: list[str | None] = []
    human_values: list[str | None] = []
    for label in label_set.labels:
        score = scores_by_key.get((label.run_index, label.case_id))
        if score is None:
            continue
        judge_values.append(score.verdict[item]["verdict"])
        human_values.append(getattr(label, item))
    return judge_values, human_values


def _kappa_block(
    label_set: HumanLabelSet | None, scores: list[CaseScore], item: str
) -> dict[str, Any]:
    """Judge/human agreement for one rubric item -- `kappa_n` (stage 10c) is
    the count of paired values kappa was actually computed from, previously
    silently dropped by `_kappa_inputs`' own `continue` with no field
    recording how many pairs survived."""
    if label_set is None:
        return {
            "kappa": None,
            "kappa_ci_low": None,
            "kappa_ci_high": None,
            "kappa_n": 0,
        }
    judge_values, human_values = _kappa_inputs(label_set, scores, item)
    paired = [
        (j, h)
        for j, h in zip(judge_values, human_values)
        if j is not None and h is not None
    ]
    if len(paired) < 2:
        return {
            "kappa": None,
            "kappa_ci_low": None,
            "kappa_ci_high": None,
            "kappa_n": len(paired),
        }
    paired_judge = [p[0] for p in paired]
    paired_human = [p[1] for p in paired]
    kappa = statistics.cohens_kappa(paired_judge, paired_human)
    kappa_ci = statistics.bootstrap_kappa_ci(paired_judge, paired_human)
    return {
        "kappa": kappa,
        "kappa_ci_low": kappa_ci[0],
        "kappa_ci_high": kappa_ci[1],
        "kappa_n": len(paired),
    }


def _judge_repeat_block(
    pass_maps: list[dict[tuple[int, str], CaseScore]], item: str
) -> dict[str, Any]:
    """Judge run-to-run variance for one rubric item (F10, spec Sec13.9
    amendment): a bootstrap interval over per-pass pass rates, computed only
    on the `(run_index, case_id)` pairs scored in *every* pass (decision
    D7) -- a pass that errored on a case must not silently make that case's
    absence look like agreement.

    `None` in every field when there is only one pass (nothing repeated to
    measure) or when the item has no applicable case in the intersection.
    """
    empty = {
        "judge_repeat_ci_low": None,
        "judge_repeat_ci_high": None,
        "judge_repeat_n_passes": None,
        "judge_repeat_pass_rates": None,
        "judge_repeat_intersection_n": None,
    }
    if len(pass_maps) < 2:
        return empty

    common_keys: set[tuple[int, str]] = set(pass_maps[0])
    for pass_map in pass_maps[1:]:
        common_keys &= set(pass_map)

    pass_rates: list[float] = []
    for pass_map in pass_maps:
        verdicts = [pass_map[key].verdict[item]["verdict"] for key in common_keys]
        applicable = [v for v in verdicts if v is not None]
        if not applicable:
            return {**empty, "judge_repeat_intersection_n": len(common_keys)}
        pass_rates.append(sum(1 for v in applicable if v == "PASS") / len(applicable))

    ci_low, ci_high = statistics.across_pass_interval(pass_rates)
    return {
        "judge_repeat_ci_low": ci_low,
        "judge_repeat_ci_high": ci_high,
        "judge_repeat_n_passes": len(pass_maps),
        "judge_repeat_pass_rates": pass_rates,
        "judge_repeat_intersection_n": len(common_keys),
    }


def _summarize(
    *,
    primary_scores: list[CaseScore],
    pass_maps: list[dict[tuple[int, str], CaseScore]],
    rubric_version: str,
    primary_label_set: HumanLabelSet | None,
    primary_kappa_skipped_reason: str | None,
    independent_label_set: HumanLabelSet | None,
    independent_kappa_skipped_reason: str | None,
    errors: list[CaseScoreError],
    judge_cost_usd: float | None,
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for item in _RUBRIC_ITEMS:
        verdicts = [score.verdict[item]["verdict"] for score in primary_scores]
        applicable = [v for v in verdicts if v is not None]
        ci_low: float | None
        ci_high: float | None
        if applicable:
            outcomes = [v == "PASS" for v in applicable]
            pass_rate: float | None = sum(outcomes) / len(outcomes)
            ci_low, ci_high = statistics.bootstrap_proportion_ci(outcomes)
        else:
            pass_rate = None
            ci_low = ci_high = None

        entry: dict[str, Any] = {
            "pass_rate": pass_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_applicable": len(applicable),
            "discriminates": statistics.discriminates(verdicts),
        }
        entry.update(_kappa_block(primary_label_set, primary_scores, item))
        entry.update(_judge_repeat_block(pass_maps, item))
        items[item] = entry

    summary: dict[str, Any] = {
        "rubric_version": rubric_version,
        "items": items,
        "metrics": _metric_aggregates(primary_scores),
        "n_errors": len(errors),
        "errors": [asdict(error) for error in errors],
        "n_passes": len(pass_maps),
        "judge_cost_usd": judge_cost_usd,
    }
    if primary_kappa_skipped_reason is not None:
        summary["kappa_skipped_reason"] = primary_kappa_skipped_reason
    if (
        independent_label_set is not None
        or independent_kappa_skipped_reason is not None
    ):
        summary["independent_label_agreement"] = {
            "run_name": (
                independent_label_set.run_name
                if independent_label_set is not None
                else None
            ),
            "n_labels": (
                len(independent_label_set.labels)
                if independent_label_set is not None
                else 0
            ),
            "kappa_skipped_reason": independent_kappa_skipped_reason,
            "items": {
                item: _kappa_block(independent_label_set, primary_scores, item)
                for item in _RUBRIC_ITEMS
            },
        }
    return summary


async def _score_one_pass(
    pack_files: list[Path],
    resolved_cases: dict[str, GoldenCase],
    judge_model: BaseChatModel | None,
    settings: Settings,
    judge_model_name: str,
    scores_path: Path,
    pass_index: int,
) -> tuple[list[CaseScore], list[CaseScoreError], float | None]:
    """One full pass over every pack: judge each, write one JSONL line per
    success, record one `CaseScoreError` per failure. Returns the pass's
    total judge cost (`None` the moment any one call's model is unpriced --
    the same "no silent partial sum" discipline `evals.metrics` already
    applies to agent cost)."""
    scores: list[CaseScore] = []
    errors: list[CaseScoreError] = []
    call_costs: list[float | None] = []
    with scores_path.open("w", encoding="utf-8") as handle:
        for path in pack_files:
            run_index, case_id = parse_pack_filename(path)
            case = resolved_cases.get(case_id)
            if case is None:
                raise ScoringError(
                    f"pack {path.name} references unknown case id {case_id!r}"
                )
            pack = pack_from_dict(json.loads(path.read_text(encoding="utf-8")))
            usage: dict[str, int] = {}
            try:
                verdict = await judge_module.judge(
                    settings, case, pack, model=judge_model, usage_sink=usage.update
                )
            except Exception as error:  # noqa: BLE001 -- one case's judge
                # failure must not cost the other 35, already-paid-for
                # calls; logged for a human to read, recorded as data,
                # never silently dropped.
                logging.getLogger("main").exception(
                    "judging case %r (run %d, pass %d) failed: %s",
                    case_id,
                    run_index,
                    pass_index,
                    error,
                )
                errors.append(
                    CaseScoreError(
                        run_index=run_index,
                        case_id=case_id,
                        error=f"{type(error).__name__}: {error}",
                        pass_index=pass_index,
                    )
                )
                continue
            if usage:
                call_costs.append(
                    pricing.cost_of(
                        judge_model_name,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
                )
            verdict_data = verdict.model_dump()
            score = CaseScore(
                run_index=run_index,
                case_id=case_id,
                verdict=verdict_data,
                metrics=asdict(metrics_of(pack)),
                pass_index=pass_index,
            )
            scores.append(score)
            handle.write(score.to_json_line() + "\n")
            handle.flush()
    pass_cost: float | None = (
        None
        if any(cost is None for cost in call_costs)
        else (
            sum(cost for cost in call_costs if cost is not None) if call_costs else None
        )
    )
    return scores, errors, pass_cost


async def score_run(
    run_dir: Path,
    *,
    cases: Sequence[GoldenCase] | None = None,
    judge_model: BaseChatModel | None = None,
    rubric_version: str | None = None,
    label_path: Path | None = None,
    independent_label_path: Path | None = None,
    passes: int = 1,
) -> ScoringResult:
    """Score every persisted pack under `run_dir / "packs"`, `passes` times.

    Parameters
    ----------
    run_dir : Path
        A run(-set) directory produced by `evals.run_eval.run_dataset`.
    cases : sequence of GoldenCase, optional
        Defaults to `evals.dataset.load_all()`. Explicit so a test scores
        against a small fixture dataset instead of the tracked one.
    judge_model : BaseChatModel, optional
        Forwarded to `evals.judge.judge` -- tests inject a fake here.
    rubric_version : str, optional
        Defaults to (and must equal) `evals.judge.JUDGE_RUBRIC_VERSION` --
        see `RubricVersionMismatchError`.
    label_path : Path, optional
        Defaults to `evals/labels/human_labels.yaml`. Kappa is included in
        `summary["items"][<item>]` only when this file exists, names this
        exact run (`run_metadata.json`'s `run_name`), and passes the
        pre-registration guard (`_load_matching_labels`).
    independent_label_path : Path, optional
        A second, independent label set (stage 10c, decision D5) -- scored
        the same way but reported under `summary["independent_label_agreement"]`,
        never pooled with `label_path`'s figures. The two sets differ in
        provenance; pooling them would produce a number about neither.
    passes : int, default 1
        Score the whole corpus this many times, each a fresh set of judge
        calls (F10: judge run-to-run variance). `passes == 1` writes
        `scores-<version>.jsonl`, byte-for-byte the pre-stage-10c filename
        and summary shape. `passes > 1` writes `scores-<version>-p<N>.jsonl`
        per pass and adds `judge_repeat_*` figures to each rubric item,
        computed on the intersection of cases scored in every pass
        (decision D7) -- never on a full-per-pass set that could shrink
        differently pass to pass.

    Raises
    ------
    ScoringError, RubricVersionMismatchError
    evals.labels.LabelValidationError
        Propagated, not caught -- an invalid label file is the human's
        mistake to fix, not something this function silently works around.
    """
    if passes < 1:
        raise ScoringError(f"passes must be >= 1, got {passes}")

    resolved_version = rubric_version or judge_module.JUDGE_RUBRIC_VERSION
    if resolved_version != judge_module.JUDGE_RUBRIC_VERSION:
        raise RubricVersionMismatchError(
            f"--rubric-version {resolved_version!r} does not match the "
            f"currently active judge.JUDGE_RUBRIC_VERSION "
            f"{judge_module.JUDGE_RUBRIC_VERSION!r} -- switching rubric "
            "versions means editing evals/judge.py's prompt/version "
            "constant (spec Sec5.4), not selecting a stored variant at "
            "runtime"
        )

    packs_dir = run_dir / "packs"
    pack_files = sorted(packs_dir.glob("*.json")) if packs_dir.exists() else []
    if not pack_files:
        raise ScoringError(f"no packs found under {packs_dir}")

    resolved_cases = {
        case.id: case for case in (cases if cases is not None else load_all())
    }

    canonical_scores_path = run_dir / f"scores-{resolved_version}.jsonl"
    prior_scores_mtime = (
        canonical_scores_path.stat().st_mtime
        if canonical_scores_path.exists()
        else None
    )
    settings = load_settings()
    _, judge_model_name = settings.resolved("judge")

    all_scores: list[CaseScore] = []
    all_errors: list[CaseScoreError] = []
    pass_maps: list[dict[tuple[int, str], CaseScore]] = []
    scores_paths: list[Path] = []
    pass_costs: list[float | None] = []
    first_pass_scores: list[CaseScore] = []
    for pass_index in range(1, passes + 1):
        pass_scores_path = (
            canonical_scores_path
            if passes == 1
            else run_dir / f"scores-{resolved_version}-p{pass_index}.jsonl"
        )
        scores_paths.append(pass_scores_path)
        pass_scores, pass_errors, pass_cost = await _score_one_pass(
            pack_files,
            resolved_cases,
            judge_model,
            settings,
            judge_model_name,
            pass_scores_path,
            pass_index,
        )
        if pass_index == 1:
            first_pass_scores = pass_scores
        all_scores.extend(pass_scores)
        all_errors.extend(pass_errors)
        pass_costs.append(pass_cost)
        pass_maps.append({(s.run_index, s.case_id): s for s in pass_scores})

    judge_cost_usd: float | None = (
        None
        if any(cost is None for cost in pass_costs)
        else (
            sum(cost for cost in pass_costs if cost is not None) if pass_costs else None
        )
    )

    run_name = _read_run_name(run_dir)
    all_cases = list(resolved_cases.values())
    resolved_label_path = label_path if label_path is not None else _DEFAULT_LABEL_PATH
    primary_label_set, primary_kappa_skipped_reason = _load_matching_labels(
        resolved_label_path,
        run_name=run_name,
        run_dir=run_dir,
        cases=all_cases,
        prior_scores_mtime=prior_scores_mtime,
    )

    independent_label_set: HumanLabelSet | None = None
    independent_kappa_skipped_reason: str | None = None
    if independent_label_path is not None:
        independent_label_set, independent_kappa_skipped_reason = _load_matching_labels(
            independent_label_path,
            run_name=run_name,
            run_dir=run_dir,
            cases=all_cases,
            prior_scores_mtime=prior_scores_mtime,
        )

    # Pass 1's own scores are what `pass_rate`/`ci_low`/`ci_high`/
    # `n_applicable`/`discriminates`/kappa are computed from, for every
    # value of `passes` -- identical to the sole pass when `passes == 1`,
    # so the headline per-item figures never depend on how many repeats
    # were run.
    summary = _summarize(
        primary_scores=first_pass_scores,
        pass_maps=pass_maps,
        rubric_version=resolved_version,
        primary_label_set=primary_label_set,
        primary_kappa_skipped_reason=primary_kappa_skipped_reason,
        independent_label_set=independent_label_set,
        independent_kappa_skipped_reason=independent_kappa_skipped_reason,
        errors=all_errors,
        judge_cost_usd=judge_cost_usd,
    )
    summary_path = run_dir / f"summary-{resolved_version}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ScoringResult(
        scores=all_scores,
        errors=all_errors,
        summary=summary,
        scores_path=scores_paths[0],
        summary_path=summary_path,
        scores_paths=scores_paths,
    )


def _read_run_name(run_dir: Path) -> str:
    """`run_metadata.json`'s `run_name` when present, else the directory's
    own name -- the same fallback `evals.run_eval.run_dataset` uses."""
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_name = metadata.get("run_name")
        if run_name:
            return str(run_name)
    return run_dir.name


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a run-set's persisted evidence packs offline"
    )
    parser.add_argument(
        "--run-set", required=True, help="Path to the run(-set) directory"
    )
    parser.add_argument(
        "--rubric-version",
        default=None,
        help="Must match the currently active evals.judge.JUDGE_RUBRIC_VERSION; "
        "defaults to it",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Score the corpus this many times (stage 10c, F10: judge "
        "run-to-run variance). 1 (default) writes scores-<version>.jsonl "
        "exactly as before; >1 writes scores-<version>-p<N>.jsonl per pass "
        "and adds judge-repeat intervals to the summary",
    )
    parser.add_argument(
        "--independent-labels",
        default=None,
        help="Path to a second, independently-labelled file (stage 10c, "
        "decision D5) -- scored and reported separately from --labels, "
        "never pooled with it",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to the primary human-label file; defaults to "
        "evals/labels/human_labels.yaml",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dir = Path(args.run_set)
    result = asyncio.run(
        score_run(
            run_dir,
            rubric_version=args.rubric_version,
            passes=args.passes,
            label_path=Path(args.labels) if args.labels else None,
            independent_label_path=(
                Path(args.independent_labels) if args.independent_labels else None
            ),
        )
    )
    print(
        f"[system] {len(result.scores)} case(s) scored across "
        f"{len(result.scores_paths)} pass(es) -- "
        f"{', '.join(str(p) for p in result.scores_paths)}; {result.summary_path}"
    )
    if result.errors:
        print(
            f"[system] {len(result.errors)} case(s) FAILED to score "
            '(recorded in the summary\'s "errors" field, not silently dropped):'
        )
        for error in result.errors:
            print(
                f"  - pass {error.pass_index} run {error.run_index} "
                f"{error.case_id}: {error.error}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
