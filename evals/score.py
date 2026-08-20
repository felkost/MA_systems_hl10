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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel

from config import load_settings
from evals import judge as judge_module
from evals import statistics
from evals.dataset import GoldenCase, load_all
from evals.evidence import EvidencePack, LlmCall, ToolCall
from evals.labels import HumanLabelSet, load_labels, validate_against_run
from evals.metrics import metrics_of
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
    """One case's judged outcome -- one line of `scores-<version>.jsonl`."""

    run_index: int
    case_id: str
    verdict: dict[str, Any]
    metrics: dict[str, Any]

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class CaseScoreError:
    """One (run_index, case_id) pair that could not be judged -- recorded
    instead of aborting the batch (live-found 2026-08-20: gemini-2.5-pro
    returned an empty response for a pack whose saved report was itself
    jailbreak/prompt-disclosure content, and the JSON parse failure that
    followed crashed the whole run before any of the other 35 cases had a
    chance to be scored)."""

    run_index: int
    case_id: str
    error: str


@dataclass
class ScoringResult:
    scores: list[CaseScore]
    errors: list[CaseScoreError]
    summary: dict[str, Any]
    scores_path: Path
    summary_path: Path


def _parse_pack_filename(path: Path) -> tuple[int, str]:
    """`<run_index>__<case_id>.json` -> `(run_index, case_id)` -- the exact
    naming `evals.run_eval._persist_pack` writes."""
    run_index_str, _, case_id = path.stem.partition("__")
    return int(run_index_str), case_id


def _pack_from_dict(data: dict[str, Any]) -> EvidencePack:
    """The inverse of `dataclasses.asdict(pack)` -- reconstructs the nested
    `ToolCall`/`LlmCall` dataclasses `_persist_pack` flattened to plain
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
    label_path: Path, *, run_name: str, run_dir: Path, prior_scores_mtime: float | None
) -> tuple[HumanLabelSet | None, str | None]:
    """Resolve the human-label set to score agreement against, or the
    reason none applies.

    Returns
    -------
    (label_set, skip_reason) : tuple
        Exactly one is non-`None`. `label_set` is only returned when the
        label file exists, names this exact run, validates against the
        packs actually on disk, and -- the pre-registration guard's second
        half (spec Sec5.6) -- was not modified after a *prior* scores file
        for this run/version already existed (a label written after
        already implies the human could have read the judge's verdicts).
    """
    if not label_path.exists():
        return None, None
    label_set = load_labels(label_path)  # raises LabelValidationError, not caught here
    if label_set.run_name != run_name:
        return None, f"label file labels run {label_set.run_name!r}, not {run_name!r}"
    if (
        prior_scores_mtime is not None
        and label_path.stat().st_mtime > prior_scores_mtime
    ):
        return None, (
            "label file was modified after a scores file for this run/version "
            "already existed -- labels must be written before scoring, not after"
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


def _summarize(
    scores: list[CaseScore],
    *,
    rubric_version: str,
    label_set: HumanLabelSet | None,
    kappa_skipped_reason: str | None,
    errors: list[CaseScoreError],
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for item in _RUBRIC_ITEMS:
        verdicts = [score.verdict[item]["verdict"] for score in scores]
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

        kappa: float | None = None
        kappa_ci: tuple[float, float] | None = None
        if label_set is not None:
            judge_values, human_values = _kappa_inputs(label_set, scores, item)
            paired = [
                (j, h)
                for j, h in zip(judge_values, human_values)
                if j is not None and h is not None
            ]
            if len(paired) >= 2:
                paired_judge = [p[0] for p in paired]
                paired_human = [p[1] for p in paired]
                kappa = statistics.cohens_kappa(paired_judge, paired_human)
                kappa_ci = statistics.bootstrap_kappa_ci(paired_judge, paired_human)

        items[item] = {
            "pass_rate": pass_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_applicable": len(applicable),
            "discriminates": statistics.discriminates(verdicts),
            "kappa": kappa,
            "kappa_ci_low": kappa_ci[0] if kappa_ci is not None else None,
            "kappa_ci_high": kappa_ci[1] if kappa_ci is not None else None,
        }

    summary: dict[str, Any] = {
        "rubric_version": rubric_version,
        "items": items,
        "metrics": _metric_aggregates(scores),
        "n_errors": len(errors),
        "errors": [asdict(error) for error in errors],
    }
    if kappa_skipped_reason is not None:
        summary["kappa_skipped_reason"] = kappa_skipped_reason
    return summary


async def score_run(
    run_dir: Path,
    *,
    cases: Sequence[GoldenCase] | None = None,
    judge_model: BaseChatModel | None = None,
    rubric_version: str | None = None,
    label_path: Path | None = None,
) -> ScoringResult:
    """Score every persisted pack under `run_dir / "packs"`.

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
        the summary only when this file exists, names this exact run
        (`run_metadata.json`'s `run_name`), and passes the pre-registration
        guard (`evals.labels`, plus this module's own mtime-vs-prior-scores
        check).

    Raises
    ------
    ScoringError, RubricVersionMismatchError
    evals.labels.LabelValidationError
        Propagated, not caught -- an invalid label file is the human's
        mistake to fix, not something this function silently works around.
    """
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

    scores_path = run_dir / f"scores-{resolved_version}.jsonl"
    prior_scores_mtime = scores_path.stat().st_mtime if scores_path.exists() else None
    settings = load_settings()

    scores: list[CaseScore] = []
    errors: list[CaseScoreError] = []
    with scores_path.open("w", encoding="utf-8") as handle:
        for path in pack_files:
            run_index, case_id = _parse_pack_filename(path)
            case = resolved_cases.get(case_id)
            if case is None:
                raise ScoringError(
                    f"pack {path.name} references unknown case id {case_id!r}"
                )
            pack = _pack_from_dict(json.loads(path.read_text(encoding="utf-8")))
            try:
                verdict = await judge_module.judge(
                    settings, case, pack, model=judge_model
                )
            except Exception as error:  # noqa: BLE001 -- one case's judge
                # failure must not cost the other 35, already-paid-for
                # calls; logged for a human to read, recorded as data,
                # never silently dropped.
                logging.getLogger("main").exception(
                    "judging case %r (run %d) failed: %s", case_id, run_index, error
                )
                errors.append(
                    CaseScoreError(
                        run_index=run_index,
                        case_id=case_id,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            verdict_data = verdict.model_dump()
            score = CaseScore(
                run_index=run_index,
                case_id=case_id,
                verdict=verdict_data,
                metrics=asdict(metrics_of(pack)),
            )
            scores.append(score)
            handle.write(score.to_json_line() + "\n")
            handle.flush()

    run_name = _read_run_name(run_dir)
    resolved_label_path = label_path if label_path is not None else _DEFAULT_LABEL_PATH
    label_set, kappa_skipped_reason = _load_matching_labels(
        resolved_label_path,
        run_name=run_name,
        run_dir=run_dir,
        prior_scores_mtime=prior_scores_mtime,
    )

    summary = _summarize(
        scores,
        rubric_version=resolved_version,
        label_set=label_set,
        kappa_skipped_reason=kappa_skipped_reason,
        errors=errors,
    )
    summary_path = run_dir / f"summary-{resolved_version}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ScoringResult(
        scores=scores,
        errors=errors,
        summary=summary,
        scores_path=scores_path,
        summary_path=summary_path,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_dir = Path(args.run_set)
    result = asyncio.run(score_run(run_dir, rubric_version=args.rubric_version))
    print(
        f"[system] {len(result.scores)} case(s) scored -- "
        f"{result.scores_path}, {result.summary_path}"
    )
    if result.errors:
        print(
            f"[system] {len(result.errors)} case(s) FAILED to score "
            '(recorded in the summary\'s "errors" field, not silently dropped):'
        )
        for error in result.errors:
            print(f"  - run {error.run_index} {error.case_id}: {error.error}")


if __name__ == "__main__":
    main(sys.argv[1:])
