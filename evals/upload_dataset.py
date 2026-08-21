"""Upload the golden dataset to Langfuse (spec Sec13, decisions D6/D7):
`create_dataset` then `create_dataset_item` per case, upserted by the case's
own stable id -- idempotent re-upload after editing a slice, the one place
this improves on an earlier iteration's own additive uploader (which
required clearing the dataset by hand before a second run).

The tracked `evals/golden/*.yaml` files are the source of truth; this
dataset is a projection (spec Sec13's own words for the relationship, "the
local file is the source of truth, Langfuse is a projection").
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from config import Settings, load_settings
from evals.dataset import GoldenCase, load_all

DATASET_NAME = "ma-systems-hl10-golden"
_DATASET_DESCRIPTION = (
    "Golden dataset for MA_systems_hl10 (spec Sec13): core, edge and "
    "adversarial slices. evals/golden/*.yaml is the source of truth; this "
    "dataset is a projection of it."
)


class DatasetUploadError(Exception):
    """Langfuse is not reachable. Structural (spec Sec5): the fix is to
    start the container stack, never a retry -- same message convention as
    `main.py::PreflightError`, which names the down dependency *and* the
    command that starts it.

    Found live 2026-08-19: with the stack stopped, `upload()` surfaced a
    raw `httpx.ReadTimeout` chain over a hundred lines long, naming
    neither Langfuse nor `docker compose`. Every other entry point in this
    project already refuses clearly; this one did not.
    """


class DatasetClient(Protocol):
    """The subset of the Langfuse client's surface this module needs --
    narrow enough to fake offline, the same `Protocol` shape an earlier
    iteration's own uploader used for its LangSmith client."""

    def create_dataset(self, *, name: str, description: str | None = None) -> Any: ...

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        id: str,
        input: Any = None,
        expected_output: Any = None,
        metadata: Any = None,
    ) -> Any: ...

    def flush(self) -> None: ...


def build_client(settings: Settings) -> DatasetClient:
    """A real `langfuse.Langfuse` client, constructed from `Settings`
    explicitly.

    Never from `os.environ`: `pydantic-settings` does not populate the
    process environment from `.env` on its own, and an earlier iteration's
    own uploader failed a real run on exactly this -- a client that reads
    its key from `os.environ` raised an auth error even though `.env` held
    a valid one.
    """
    from langfuse import Langfuse

    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )


def _item_payload(case: GoldenCase) -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "id": case.id,
        "input": {"question": case.question},
        "expected_output": case.expects.model_dump(),
        "metadata": {"rationale": case.rationale},
    }


def upload(
    cases: list[GoldenCase] | None = None,
    *,
    client: DatasetClient | None = None,
    settings: Settings | None = None,
    host: str | None = None,
) -> None:
    """Upload every golden case as one Langfuse dataset item, upserted by
    the case's own stable id.

    Parameters
    ----------
    cases : list of GoldenCase, optional
        Defaults to `load_all()` -- every case under `evals/golden/`.
    client : DatasetClient, optional
        Defaults to `build_client(settings)`. Injected so the gate tests
        this against a fake, with no container and no network.
    settings : Settings, optional
        Defaults to `load_settings()`; only consulted when `client` is not
        given.
    host : str, optional
        The Langfuse host to name in a `DatasetUploadError`. Defaults to
        `settings.langfuse_host`; passed explicitly only when `client` is
        injected and there is no `Settings` to read it from.

    Raises
    ------
    DatasetUploadError
        Langfuse is not reachable -- one named refusal naming the host and
        the command that starts the stack, never a raw `httpx` traceback.

    Notes
    -----
    `create_dataset_item(id=...)` upserts (Langfuse 4.14.4, measured): a
    second run after editing a slice updates existing items in place
    instead of duplicating them.
    """
    resolved_cases = cases if cases is not None else load_all()
    resolved_settings = settings
    if client is not None:
        resolved_client = client
    else:
        if resolved_settings is None:
            resolved_settings = load_settings()
        resolved_client = build_client(resolved_settings)
    resolved_host = host or (
        resolved_settings.langfuse_host if resolved_settings else "the Langfuse host"
    )

    try:
        resolved_client.create_dataset(
            name=DATASET_NAME, description=_DATASET_DESCRIPTION
        )
        for case in resolved_cases:
            resolved_client.create_dataset_item(**_item_payload(case))
        resolved_client.flush()
    except httpx.HTTPError as error:
        raise DatasetUploadError(
            f"Langfuse at {resolved_host} is not reachable ({type(error).__name__})"
            " -- start the container stack with `docker compose up -d`, wait"
            " for all six services to report healthy, then re-run this"
            " command"
        ) from error


def main() -> None:
    """CLI entry point. A structural failure prints one `[system]` line and
    exits non-zero -- the same shape `main.py` and `servers.py` already use,
    rather than letting a traceback reach the operator."""
    try:
        upload()
    except DatasetUploadError as error:
        print(f"[system] {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
