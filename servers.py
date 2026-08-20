"""One-command launcher for all five servers (stage 8): two MCP + three A2A,
each an independent OS process, a readiness wait, Ctrl+C shutdown.

**Process model: `subprocess.Popen`, not `multiprocessing`.** Not really an
open choice -- `a2a_servers.py`'s own `main()` docstring already commits to
"launches all three by name through this same contract" (a CLI invocation),
predating this module. Confirmed on the merits too: reuses the
`Popen(stdout=PIPE)` + stdout-poll-for-readiness pattern already proven 3x in
this project's test suite rather than inventing one; avoids
`multiprocessing.spawn`'s Windows re-import cost for `agents/*`'s
langchain/langgraph imports and its picklability constraints; gives each
server its own stdout pipe for readiness polling without interleaving, which
`multiprocessing.Process` does not provide for free.

**Two-phase boot, not five-at-once.** The first draft started all five
simultaneously on the premise that no startup-order dependency exists --
wrong: this same stage adds a fail-fast SearchMCP-reachability preflight to
`a2a_servers.py`, which *creates* that dependency. Booted together, the
three A2A servers would race SearchMCP's cold start (3-10s of imports alone,
measured at stage 2) and die with `sys.exit(1)`. Two phases -- (SearchMCP,
ReportMCP), then (Planner, Researcher, Critic) -- resolves it, and matches
the brief's own documented launch order ("Порядок запуску"). Stated cost:
boot wall-clock is the sum of the two phases' readiness waits, not their max.

No auth token handling lives here: each child's own `main()` calls
`auth.build_token_verifier(settings)` itself (search_mcp.py, report_mcp.py,
a2a_servers.py) and refuses on its own if unset -- this module only chooses
*what* to run and *when*, inheriting the parent's environment (and, for
ports, an explicit per-spec override -- see `default_phases`) exactly as
`python main.py`/`python a2a_servers.py <name>` already would run standalone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import paths
from config import Settings, load_settings

_READY_SIGNAL = "Uvicorn running"  # true for both FastMCP's http_app() and
# a2a_servers.py's own uvicorn.run() -- one constant covers all five.
_DEFAULT_READY_TIMEOUT_SECONDS = 60.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_LIVENESS_POLL_SECONDS = 1.0
_READY_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class _ServerSpec:
    """One process to boot: `argv[0]` is the script, the rest its own CLI
    args (e.g. `a2a_servers.py`'s agent name). `env` is merged over the
    parent's environment for this one subprocess -- `None` inherits it
    unchanged."""

    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str] | None = None


class ServerLaunchError(Exception):
    """A server did not reach readiness within the timeout, or exited
    early. Structural (spec Sec5): names the server, same message
    convention as `main.py`'s own `PreflightError`.

    `running` carries every `_RunningServer` `launch_all` had already
    started (across every phase) at the moment of failure -- `launch_all`
    stops all of them via `shutdown_all` before re-raising with this
    attached, so a caller (or a test) can confirm no orphan survives.
    """

    running: tuple["_RunningServer", ...] = ()


class _RunningServer:
    """One live subprocess plus its stdout reader thread and captured
    lines, for readiness polling and, on failure, diagnostic output."""

    def __init__(self, spec: _ServerSpec, process: "subprocess.Popen[str]") -> None:
        self.spec = spec
        self.process = process
        self.lines: list[str] = []
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.append(line)

    def is_ready(self) -> bool:
        return any(_READY_SIGNAL in line for line in self.lines)

    def exited_early(self) -> bool:
        """True once the process has exited, whether or not it ever
        printed the ready line -- `main()`'s post-readiness liveness check
        reuses this same method for a crash *after* boot succeeded."""
        return self.process.poll() is not None


def _port_env(settings: Settings) -> dict[str, str]:
    """Every server's port, explicit in every spec's environment. Each
    process resolves its dependencies' locations via its own `Settings`,
    so a spec's env must carry the whole picture, not just its own port --
    e.g. the A2A servers' own preflight (stage 8) needs the *actual*
    SearchMCP port, which may differ from the default under ephemeral
    ports (the smoke test's own scenario)."""
    return {
        "SEARCH_MCP_PORT": str(settings.search_mcp_port),
        "REPORT_MCP_PORT": str(settings.report_mcp_port),
        "PLANNER_A2A_PORT": str(settings.planner_a2a_port),
        "RESEARCHER_A2A_PORT": str(settings.researcher_a2a_port),
        "CRITIC_A2A_PORT": str(settings.critic_a2a_port),
    }


def default_phases(settings: Settings) -> tuple[tuple[_ServerSpec, ...], ...]:
    """The five servers in this project's own port-table order, in two boot
    phases: (SearchMCP, ReportMCP), then (Planner, Researcher, Critic)."""
    env = _port_env(settings)
    search_mcp = paths.PROJECT_ROOT / "mcp_servers" / "search_mcp.py"
    report_mcp = paths.PROJECT_ROOT / "mcp_servers" / "report_mcp.py"
    a2a = paths.PROJECT_ROOT / "a2a_servers.py"
    return (
        (
            _ServerSpec("SearchMCP", (str(search_mcp),), env=env),
            _ServerSpec("ReportMCP", (str(report_mcp),), env=env),
        ),
        (
            _ServerSpec("Planner", (str(a2a), "planner"), env=env),
            _ServerSpec("Researcher", (str(a2a), "researcher"), env=env),
            _ServerSpec("Critic", (str(a2a), "critic"), env=env),
        ),
    )


def _start(spec: _ServerSpec) -> _RunningServer:
    env = dict(os.environ)
    if spec.env:
        env.update(spec.env)
    process = subprocess.Popen(
        [sys.executable, *spec.argv],
        cwd=paths.PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return _RunningServer(spec, process)


def _wait_until_ready(phase: Sequence[_RunningServer], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for server in phase:
            if server.exited_early() and not server.is_ready():
                raise ServerLaunchError(
                    f"{server.spec.name} exited before becoming ready "
                    f"(exit code {server.process.returncode}):\n"
                    + "".join(server.lines)
                )
        if all(server.is_ready() for server in phase):
            return
        time.sleep(_READY_POLL_SECONDS)
    not_ready = ", ".join(s.spec.name for s in phase if not s.is_ready())
    raise ServerLaunchError(f"{not_ready} did not become ready within {timeout}s")


def launch_all(
    settings: Settings | None = None,
    *,
    phases: Sequence[Sequence[_ServerSpec]] | None = None,
    ready_timeout: float = _DEFAULT_READY_TIMEOUT_SECONDS,
) -> list[_RunningServer]:
    """Boot every phase in order, waiting for each phase's servers to all
    become ready under one shared deadline before starting the next phase
    -- not five (or per-phase N) sequential per-server timeouts, since a
    stuck server should fail the whole boot promptly.

    Parameters
    ----------
    settings : Settings, optional
        Defaults to `load_settings()` -- but only read at all if `phases`
        is not given, so a hermetic test that injects `phases` never needs
        a valid `Settings`/`.env`/`OPENAI_API_KEY` either.
    phases : sequence of sequence of _ServerSpec, optional
        Defaults to `default_phases(settings)`. Injectable so tests can
        substitute dummy scripts instead of paying real
        FastMCP/uvicorn/langchain import cost.
    ready_timeout : float
        Per-phase readiness deadline, in seconds.

    Raises
    ------
    ServerLaunchError
        A server exited before becoming ready, or the deadline passed with
        at least one server still not ready. Every server already started
        (across every phase) is terminated before this is raised -- never
        an orphaned partial boot.
    """
    boot_phases = (
        phases if phases is not None else default_phases(settings or load_settings())
    )
    running: list[_RunningServer] = []
    try:
        for phase in boot_phases:
            started = [_start(spec) for spec in phase]
            running.extend(started)
            _wait_until_ready(started, ready_timeout)
    except ServerLaunchError as error:
        shutdown_all(running)
        error.running = tuple(running)
        raise
    return running


def shutdown_all(servers: Sequence[_RunningServer]) -> None:
    """`terminate()` -> `wait(10)` -> `kill()` per process, generalizing
    the pattern already proven in this project's test suite to N
    processes. Never raises -- a shutdown that can itself fail is a
    launcher that can hang on Ctrl+C, which defeats the point."""
    for server in servers:
        try:
            server.process.terminate()
        except Exception:
            pass
    for server in servers:
        try:
            server.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                server.process.kill()
                server.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except Exception:
                pass
        except Exception:
            pass


def main() -> None:
    settings = load_settings()
    try:
        running = launch_all(settings)
    except ServerLaunchError as error:
        print(f"[system] {error}", flush=True)
        sys.exit(1)

    # flush=True on every line below: live verification (2026-08-18) found
    # the readiness banner sitting unflushed in Python's default block
    # buffering whenever stdout is redirected to a file (`servers.py >
    # log.txt`, the exact shape an operator running this detached would
    # use) -- the process was demonstrably up (all five ports listening)
    # long before the banner became visible in the log.
    print("[system] all five servers ready. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(_LIVENESS_POLL_SECONDS)
            dead = [server for server in running if server.exited_early()]
            if dead:
                names = ", ".join(server.spec.name for server in dead)
                print(
                    f"[system] {names} exited unexpectedly -- stopping the stack",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        print("\n[system] stopping...", flush=True)
    finally:
        shutdown_all(running)


if __name__ == "__main__":
    main()
