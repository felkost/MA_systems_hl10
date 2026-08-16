"""ReportMCP :8902 -- the only write in the whole system.

`save_report` and `resource://output-dir` behind their own HTTP MCP server.
No module-level *mutable* state (spec Sec6 rule 2): `Settings` is loaded
fresh inside every callable, the same discipline `search_mcp.py` already
uses -- an eager module-level load would also need `OPENAI_API_KEY` just to
import this module, since `Settings` validates all five chat roles' keys
unconditionally even though nothing here calls a chat model.

The path-confinement guardrail (`_resolve_inside_output`) is the safety
property every downstream stage's HITL story depends on: `save_report`'s
`filename` argument is untrusted (it can be shaped by a page `read_url`
fetched, forwarded through the Researcher's and Critic's output), and this
function is what stands between that string and the filesystem. A refusal
here is a **policy** failure (spec Sec5) -- typed as `ReportPathError`, never
a bare string, so stage 9's spans and stage 10b's failure taxonomy can count
"guardrail tripped" apart from "the disk was full".
"""

from __future__ import annotations

import logging
import os.path
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    # `python mcp_servers/report_mcp.py` (the invocation docs/task-hl10.md
    # prescribes) puts this file's own directory on sys.path[0], not the
    # project root -- same fix search_mcp.py already carries, and the same
    # reason: the project-local imports below would not resolve otherwise.
    sys.path.insert(0, str(_PROJECT_ROOT))

import observability  # noqa: E402
import paths  # noqa: E402
from config import load_settings  # noqa: E402

# First character must not be a dot: a naive [A-Za-z0-9._-]{1,120} class
# admits ".", "..", "..." and ".md" outright, since every character in each
# of them is individually allowed (insights.md, 2026-08-16). Because "/",
# "\" and ":" are all outside this class, any string that matches is always
# exactly one path component -- traversal is not spellable at all, not
# merely blocked incidentally.
_ALLOWED_FILENAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,119}")

_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{digit}" for digit in range(1, 10)}
    | {f"LPT{digit}" for digit in range(1, 10)}
)

_logger = logging.getLogger("report_mcp")

mcp = FastMCP("ReportMCP")


class ReportPathError(Exception):
    """A filename refused by `_resolve_inside_output` -- policy class, per
    spec Sec5: the caller never lets this reach the model as a raise."""


def _resolve_inside_output(filename: str, output_root: Path) -> Path:
    """Confine `filename` to `output_root`, appending `.md` if it is bare.

    Two independent layers -- see insights.md 2026-08-16 for the corrected
    account of which case each one actually catches. Layer 1 (the character
    allowlist) closes every *spellable* traversal shape, because none of
    them can be written without a forbidden character. Layer 2 (resolve
    then compare parents) is what catches a symlink: a name that clears
    layer 1 outright but resolves, once followed, outside `output_root`.

    Raises
    ------
    ReportPathError
        `filename` fails the character allowlist, names a Windows reserved
        device, carries a suffix other than `.md`, or resolves outside
        `output_root`.
    """
    if not _ALLOWED_FILENAME.fullmatch(filename):
        raise ReportPathError(
            f"filename {filename!r} is not allowed -- only letters, digits, "
            "'_', '-' and '.' are permitted, and the name must not start "
            "with '.'"
        )

    candidate = Path(filename)
    stem_upper = candidate.stem.upper()
    if stem_upper in _WINDOWS_RESERVED_STEMS:
        raise ReportPathError(f"filename {filename!r} names a Windows reserved device")

    if candidate.suffix == "":
        candidate = candidate.with_suffix(".md")
    elif candidate.suffix != ".md":
        raise ReportPathError(
            f"filename {filename!r} must have no suffix or '.md', not "
            f"{candidate.suffix!r}"
        )

    resolved = (output_root / candidate.name).resolve()
    # os.path.normcase, not a bare `==`: on Windows, "F:/X/OUTPUT" ==
    # "F:/X/output" is False even though the two name one directory
    # (measured, insights.md 2026-08-16) -- a raw comparison here would
    # refuse legitimate writes whenever case differs.
    if os.path.normcase(str(resolved.parent)) != os.path.normcase(
        str(output_root.resolve())
    ):
        raise ReportPathError(
            f"filename {filename!r} resolves outside the output directory"
        )
    return resolved


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def save_report(filename: str, content: str) -> str:
    """Save `content` as a Markdown report named `filename` (`.md` appended
    if omitted). Refuses to overwrite an existing report -- pick a
    different name if this one is taken."""
    _logger.info("save_report filename=%r", filename)
    settings = load_settings()
    output_root = paths.resolve(settings.output_dir)

    try:
        target = _resolve_inside_output(filename, output_root)
    except ReportPathError as error:
        _logger.warning("save_report filename=%r refused: %s", filename, error)
        return f"ERROR: {error}"

    if len(content) > settings.max_report_content_length:
        _logger.warning(
            "save_report filename=%r refused: content too long (%d > %d)",
            filename,
            len(content),
            settings.max_report_content_length,
        )
        return (
            f"ERROR: content is {len(content)} characters, exceeding the "
            f"{settings.max_report_content_length}-character limit"
        )

    if target.exists():
        _logger.warning(
            "save_report filename=%r refused: %s already exists", filename, target.name
        )
        return f"ERROR: {target.name!r} already exists -- choose a different name"

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as error:
        _logger.warning("save_report filename=%r failed to write: %s", filename, error)
        return f"ERROR: could not write {target.name!r}: {error}"

    # Joined against settings.output_dir as configured, not against
    # PROJECT_ROOT: in the default configuration ("output") the two
    # coincide, but a test (or an operator) pointing OUTPUT_DIR at an
    # absolute path outside the project makes target.relative_to
    # (PROJECT_ROOT) raise ValueError -- this stays correct either way.
    return str(Path(settings.output_dir) / target.name)


@mcp.resource("resource://output-dir")
def output_dir() -> dict[str, Any]:
    """The output directory's path and the reports saved in it so far --
    never creates the directory, so a fresh checkout with no report yet
    still answers instead of failing."""
    _logger.info("output_dir requested")
    settings = load_settings()
    output_root = paths.resolve(settings.output_dir)

    if not output_root.is_dir():
        return {"path": str(output_root), "count": 0, "reports": []}

    reports = [
        {
            "name": entry.name,
            "size_bytes": entry.stat().st_size,
            "modified": datetime.fromtimestamp(
                entry.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        for entry in sorted(output_root.glob("*.md"))
    ]
    return {"path": str(output_root), "count": len(reports), "reports": reports}


def main() -> None:
    settings = load_settings()
    observability.configure_logging("report_mcp")
    mcp.run(transport="http", host="127.0.0.1", port=settings.report_mcp_port)


if __name__ == "__main__":
    main()
