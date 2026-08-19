"""The dataset runner (spec Sec13, decisions D0/D6): drives the real
five-server stack over the golden dataset without a human at every example.

Boots the stack once for the whole dataset -- six processes at stage-9's
measured 0.64 GB minimum free RAM do not survive a boot per example, and the
two-phase boot alone costs ~31 s. Every case gets a fresh `thread_id`: one
session-long thread would let the checkpointer skip Planner/Researcher on
cases after the first (measured at stage 9: 209 s -> 36 s for a follow-up on
the same thread), silently evaluating a different system from case two
onward. The whole dataset is loaded and validated before the first case
runs -- hl8 lost the tokens of every cell preceding a broken one.

This module never touches `supervisor.py`, `hitl.py` or `middleware.py`: the
auto-approve harness (`evals/harness.py`) substitutes `main._resolve_interrupt`
for the duration of one case only, restored immediately after -- the
production HITL gate keeps exactly one behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import uuid
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import hitl
import main as repl
import models
import observability
import servers
from config import Settings, load_settings
from evals import evidence, judge as judge_module
from evals.dataset import GoldenCase, load_all
from evals.harness import auto_approve
from models import resolved_map
from paths import PROJECT_ROOT
from supervisor import create_supervisor

_RUNS_DIR_NAME = "evals"
FIXTURES_DIR = PROJECT_ROOT / "evals" / "fixtures"
_FIXTURE_URL_PLACEHOLDER = "{fixture_url}"


class DatasetValidationError(Exception):
    """Fail before spending (hl8's own lesson): raised before a single case
    runs, never mid-sweep."""


class DatasetLinker(Protocol):
    """The subset of the Langfuse client this module needs to link a run's
    traces back to the dataset items they answered (D6).

    Narrow enough to fake offline, the same `Protocol` shape
    `evals/upload_dataset.py::DatasetClient` already uses -- which is what
    lets the gate prove the linking with no container running.
    """

    def create_dataset_run_item(
        self,
        *,
        run_name: str,
        dataset_item_id: str,
        trace_id: str,
        metadata: Any = None,
    ) -> Any: ...

    def flush(self) -> None: ...


def _link_case_to_dataset(
    linker: DatasetLinker,
    *,
    run_name: str,
    case: GoldenCase,
    trace_id: str,
) -> None:
    """Link one case's trace to its dataset item, never letting the link
    fail the run.

    D3 makes this a *projection*: the JSONL span dump is the evidence, and
    a Langfuse that is down or misconfigured must cost the run its UI
    linkage, not its data. Swallowing the exception here is therefore the
    point, not laziness -- the alternative loses a paid run's results to a
    container problem.
    """
    try:
        linker.create_dataset_run_item(
            run_name=run_name,
            dataset_item_id=case.id,
            trace_id=trace_id,
        )
    except Exception as error:  # noqa: BLE001 -- see the docstring
        logging.getLogger("main").warning(
            "dataset run link failed for case %r: %s", case.id, error
        )


class FixtureServer:
    """Serves `evals/fixtures/` over loopback HTTP on an ephemeral port
    (D5): the adversarial slice's indirect-injection case needs a real
    `read_url`-reachable URL, and a live third-party injection page would
    be neither reproducible nor something this repository should point at.

    `read_url`'s own egress guardrail refuses loopback addresses by
    design (stage 3) -- `run_dataset` sets `ALLOW_PRIVATE_NETWORK_URLS=true`
    in the child processes' environment only when a fixture server is
    actually needed, and only for that run. This measures the *content*
    defence (the untrusted-content delimiters, the agents' behaviour on
    injected instructions), never a re-test of egress confinement itself,
    which stays proven by stage 3's own tests.
    """

    def __init__(self, directory: Path = FIXTURES_DIR) -> None:
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(directory)
        )
        self._httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> "FixtureServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def url_for(self, filename: str) -> str:
        port = self._httpd.server_address[1]
        return f"http://127.0.0.1:{port}/{filename}"


def _build_poisoned_index(run_dir: Path) -> tuple[Path, Path]:
    """Build a throwaway index containing the real corpus plus the
    poisoned-document fixture (D5) -- never touches the repository's own
    `data/`/`index/`, so the `output/` -> `data/` poisoning invariant this
    project holds absolute is never at risk from an eval run.

    Costs real embedding calls on every rebuild (`ingest.ingest`); the
    caller invokes this once per run, never per case. Live-only: no
    offline test builds a real index, since that would require a real
    embeddings provider.
    """
    import shutil

    from ingest import ingest as run_ingest

    data_dir = run_dir / "eval_data"
    index_dir = run_dir / "eval_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    for source in (PROJECT_ROOT / "data").iterdir():
        if source.is_file() and source.name != ".gitkeep":
            shutil.copy2(source, data_dir / source.name)
    poisoned = FIXTURES_DIR / "poisoned_document.md"
    shutil.copy2(poisoned, data_dir / poisoned.name)

    eval_settings = load_settings().model_copy(
        update={"data_dir": str(data_dir), "index_dir": str(index_dir)}
    )
    run_ingest(eval_settings)
    return data_dir, index_dir


def _resolve_question(case: GoldenCase, fixture_server: "FixtureServer | None") -> str:
    """Substitute `{fixture_url}` with the fixture server's actual runtime
    URL -- the port is only known once the server has started, so the
    golden YAML can never hardcode it."""
    if case.fixture is None:
        return case.question
    if fixture_server is None:
        raise DatasetValidationError(
            f"case {case.id!r} needs fixture {case.fixture!r} but no "
            "fixture server was started"
        )
    return case.question.replace(
        _FIXTURE_URL_PLACEHOLDER, fixture_server.url_for(case.fixture)
    )


@dataclass
class CaseResult:
    """One case's outcome -- appended to the run's `results.jsonl` as soon
    as it is known, so a crash mid-run loses one case, not the sweep."""

    case_id: str
    thread_id: str
    trace_id: str | None
    final_answer: str
    saved_report: bool
    verdict: dict[str, Any] | None = None
    error: str | None = None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


PreflightFn = Callable[[Settings], Awaitable[dict[str, Any]]]
BuildAgentFn = Callable[..., Any]
LaunchAllFn = Callable[[Settings], list[Any]]
ShutdownAllFn = Callable[[Sequence[Any]], None]


@contextmanager
def _auto_approved() -> Iterator[None]:
    """Substitute `main._resolve_interrupt` with the auto-approve harness
    for the duration of one case, restored unconditionally after -- the
    same seam stage 9's own smoke test patches, made a first-class context
    manager here since the runner calls it once per case rather than once
    per test."""
    original = repl._resolve_interrupt
    repl._resolve_interrupt = auto_approve  # type: ignore[assignment]
    try:
        yield
    finally:
        repl._resolve_interrupt = original  # type: ignore[assignment]


def _final_answer_text(messages: list[Any]) -> str:
    """The last message carrying non-empty text -- `BaseMessage.text`
    flattens both a plain string and the list-of-content-blocks shape a
    real MCP-loaded tool result carries (`hitl.py`'s own documented trap)."""
    for message in reversed(messages):
        text = getattr(message, "text", "")
        if text:
            return str(text)
    return ""


def _saved_report(messages: list[Any]) -> bool:
    return hitl.render_save_status(messages).startswith("Report saved to:")


async def _drive_case(
    agent: Any,
    settings: Settings,
    case: GoldenCase,
    question: str,
    *,
    span_dump_dir: Path,
) -> CaseResult:
    """Run one case through the real Supervisor graph with HITL
    auto-approved, and assemble its evidence.

    A fresh `thread_id` per case (see module docstring) -- never reused,
    never derived from the case id, so two runs of the same case never
    collide on the checkpointer.

    Parameters
    ----------
    question : str
        `case.question` with any `{fixture_url}` placeholder already
        substituted (`_resolve_question`) -- driven separately from `case`
        so the substitution stays testable on its own.
    """
    thread_id = str(uuid.uuid4())
    config = repl.build_run_config(settings, thread_id)

    with _auto_approved():
        messages = await repl._drive(agent, config, settings, question)

    trace_id = evidence.find_case_trace_id(span_dump_dir, thread_id)
    return CaseResult(
        case_id=case.id,
        thread_id=thread_id,
        trace_id=trace_id,
        final_answer=_final_answer_text(messages),
        saved_report=_saved_report(messages),
    )


def _validate_dataset(cases: Sequence[GoldenCase]) -> None:
    if not cases:
        raise DatasetValidationError("no cases to run")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise DatasetValidationError(f"duplicate case id(s): {', '.join(duplicates)}")


async def run_dataset(
    settings: Settings,
    cases: Sequence[GoldenCase],
    *,
    run_dir: Path,
    run_judge: bool = True,
    launch_all: LaunchAllFn = servers.launch_all,
    shutdown_all: ShutdownAllFn = servers.shutdown_all,
    preflight: PreflightFn | None = None,
    build_agent: BuildAgentFn = create_supervisor,
    judge_model: BaseChatModel | None = None,
    dataset_linker: DatasetLinker | None = None,
    run_name: str | None = None,
) -> list[CaseResult]:
    """Run every case in `cases` against the stack `launch_all` boots,
    writing one JSON line per case to `run_dir / "results.jsonl"` as soon as
    it is known.

    Parameters
    ----------
    settings : Settings
        Must already point `span_dump_dir`/`output_dir`/`checkpoint_db`
        inside `run_dir` -- this function does not set them itself, so a
        test can point them anywhere.
    cases : sequence of GoldenCase
        Validated before anything boots (`DatasetValidationError`).
    run_dir : Path
        Created if missing; holds `results.jsonl` and (via `settings`) the
        span dump and output directory.
    run_judge : bool, default True
        Skip the judge entirely when `False` -- e.g. no `JUDGE_*`
        configured.
    launch_all, shutdown_all : callables, default servers.launch_all/shutdown_all
        Injectable so a test drives this against dummy processes or a
        no-op stack instead of the real five servers.
    preflight : callable, optional
        Defaults to `main._preflight`. Injectable for the same reason.
    build_agent : callable, default supervisor.create_supervisor
        Injectable so a test drives a scripted fake agent instead of a real
        Supervisor graph.
    judge_model : BaseChatModel, optional
        Forwarded to `evals.judge.judge` -- tests inject a scripted fake
        here instead of reaching a real provider.
    dataset_linker : DatasetLinker, optional
        When given, each case's trace is linked back to its Langfuse
        dataset item (D6), which is what fills the dataset's Experiments
        tab. `None` (the default) keeps the whole run offline-usable --
        the JSONL evidence never depends on it (D3).
    run_name : str, optional
        The Langfuse dataset-run name every link is filed under. Defaults
        to the run directory's own name, so a run is findable in the UI by
        the same string it has on disk.

    Returns
    -------
    list of CaseResult
    """
    _validate_dataset(cases)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_preflight = preflight or repl._preflight

    observability.configure_logging("main")
    observability.configure_observability(settings, "main")

    resolved_run_name = run_name or run_dir.name
    metadata = {
        "run_dir": str(run_dir),
        "run_name": resolved_run_name,
        "provider_map": resolved_map(settings),
        "case_ids": [case.id for case in cases],
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    fixture_server: FixtureServer | None = None
    if any(case.fixture is not None for case in cases):
        # D5: only when a case actually needs a served fixture, and only
        # for this run -- `read_url`'s egress guardrail otherwise refuses
        # loopback by design (stage 3), which stays proven by stage 3's own
        # tests regardless of this override.
        os.environ["ALLOW_PRIVATE_NETWORK_URLS"] = "true"
        fixture_server = FixtureServer().start()

    if any(case.needs_poisoned_index for case in cases):
        data_dir, index_dir = _build_poisoned_index(run_dir)
        # Real environment variables, not a `Settings.model_copy` -- the
        # subprocess children `launch_all` spawns inherit `os.environ`
        # wholesale (`servers.py::_start`), never this process's in-memory
        # `Settings` object.
        os.environ["DATA_DIR"] = str(data_dir)
        os.environ["INDEX_DIR"] = str(index_dir)

    running = launch_all(settings)
    results: list[CaseResult] = []
    results_path = run_dir / "results.jsonl"
    try:
        preflight_result = await resolved_preflight(settings)
        if run_judge:
            await models.assert_structured_output_supported(settings, "judge")

        async with AsyncSqliteSaver.from_conn_string(
            str(run_dir / "checkpoints.sqlite")
        ) as saver:
            agent = build_agent(
                settings,
                checkpointer=saver,
                save_report_tool=preflight_result["save_report_tool"],
            )
            span_dump_dir = Path(str(settings.span_dump_dir))
            with results_path.open("a", encoding="utf-8") as handle:
                for case in cases:
                    question = _resolve_question(case, fixture_server)
                    result = await _drive_case(
                        agent, settings, case, question, span_dump_dir=span_dump_dir
                    )
                    if run_judge and result.trace_id is not None:
                        pack = evidence.assemble(
                            span_dump_dir,
                            result.trace_id,
                            final_answer=result.final_answer,
                        )
                        verdict = await judge_module.judge(
                            settings, case, pack, model=judge_model
                        )
                        result.verdict = verdict.model_dump()
                    if dataset_linker is not None and result.trace_id is not None:
                        _link_case_to_dataset(
                            dataset_linker,
                            run_name=resolved_run_name,
                            case=case,
                            trace_id=result.trace_id,
                        )
                    results.append(result)
                    handle.write(result.to_json_line() + "\n")
                    handle.flush()
    finally:
        shutdown_all(running)
        observability.shutdown_observability()
        if fixture_server is not None:
            fixture_server.stop()
        if dataset_linker is not None:
            # Ingestion is batched; a runner that exits without flushing
            # links nothing at all -- the same trap `upload_dataset.py`
            # already carries a comment about.
            try:
                dataset_linker.flush()
            except Exception as error:  # noqa: BLE001 -- projection, not evidence
                logging.getLogger("main").warning(
                    "dataset linker flush failed: %s", error
                )

    return results


def _settings_for_run(run_dir: Path) -> Settings:
    """Point this run's `SPAN_DUMP_DIR`/`OUTPUT_DIR`/`CHECKPOINT_DB` at
    `run_dir`, as real process environment variables -- `Settings` fields
    set via `.model_copy` never reach the child server processes
    `servers.launch_all` spawns, since each child inherits `dict(os.environ)`
    (`servers.py::_start`), not this process's in-memory `Settings` object.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    os.environ["SPAN_DUMP_DIR"] = str(run_dir / "spans")
    os.environ["OUTPUT_DIR"] = str(run_dir / "output")
    os.environ["CHECKPOINT_DB"] = str(run_dir / "checkpoints.sqlite")
    return load_settings()


def _select_cases(
    all_cases: list[GoldenCase], wanted_ids: str | None
) -> list[GoldenCase]:
    if wanted_ids is None:
        return all_cases
    wanted = {case_id.strip() for case_id in wanted_ids.split(",") if case_id.strip()}
    selected = [case for case in all_cases if case.id in wanted]
    missing = wanted - {case.id for case in selected}
    if missing:
        raise DatasetValidationError(
            f"unknown case id(s): {', '.join(sorted(missing))}"
        )
    return selected


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the golden dataset against the real five-server stack"
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids to run; default is the whole dataset",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory for this run's span dump, output and results; "
        "default is a fresh runtime/evals/<timestamp> directory",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the judge for every case"
    )
    parser.add_argument(
        "--no-dataset-link",
        action="store_true",
        help="Do not link this run's traces to the Langfuse dataset (D6). "
        "Linking is skipped automatically when the Langfuse keys are unset, "
        "so this flag is only for running against a live Langfuse without "
        "recording the run in it",
    )
    return parser.parse_args(argv)


def _build_dataset_linker(settings: Settings) -> DatasetLinker | None:
    """A real Langfuse client for D6's run linking, or `None` when the keys
    are unset -- an unconfigured Langfuse must skip the projection, never
    fail the run (D3)."""
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    from evals.upload_dataset import build_client

    return build_client(settings)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cases = _select_cases(load_all(), args.cases)

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else PROJECT_ROOT / "runtime" / _RUNS_DIR_NAME / str(int(time.time()))
    )
    settings = _settings_for_run(run_dir)
    linker = None if args.no_dataset_link else _build_dataset_linker(settings)

    results = asyncio.run(
        run_dataset(
            settings,
            cases,
            run_dir=run_dir,
            run_judge=not args.no_judge,
            dataset_linker=linker,
        )
    )
    linked = " (linked to the Langfuse dataset)" if linker is not None else ""
    print(
        f"[system] {len(results)} case(s) run{linked} -- results at "
        f"{run_dir / 'results.jsonl'}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
