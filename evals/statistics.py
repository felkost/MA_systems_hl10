"""Statistics for the sweep: bootstrap confidence intervals for a pass rate
and Cohen's kappa for judge/human agreement (spec Sec13.6/13.7/13.9, decision
D7).

Thin wrappers, deliberately -- `scipy.stats.bootstrap` and
`sklearn.metrics.cohen_kappa_score` do the actual statistics; what these
functions add is the project's own reporting discipline: a degenerate input
(a constant sample, a single-label confusion matrix) must never leak `NaN`
into a report table, and every random draw is seeded so a published interval
is reproducible from the same run data.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy import stats

# scikit-learn ships no py.typed marker and no PyPI stub package exists
# (unlike scipy-stubs, pinned alongside it) -- every call site below passes
# plain lists and reads a plain float back, so the missing stub costs
# nothing beyond this one import.
from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]


def bootstrap_proportion_ci(
    outcomes: Sequence[bool],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    random_state: int = 0,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval for the proportion of `True` values
    in `outcomes` (spec Sec13.9: "n >= 3 runs, confidence intervals, no
    single number in a report").

    `random_state` fixes the resampling draw -- two calls on the same
    `outcomes` with the same seed return byte-identical bounds, which is
    what makes a published interval reproducible from the run data alone.

    Returns
    -------
    (low, high) : tuple of float
        A degenerate sample (every outcome identical) returns the point
        value twice, e.g. `(1.0, 1.0)` for all-`True` -- `scipy`'s BCa
        method returns `NaN` bounds there (division by a zero-variance
        term), and a `NaN` interval is worse than an honest point interval.

    Raises
    ------
    ValueError
        `outcomes` is empty -- there is no interval to compute.
    """
    if not outcomes:
        raise ValueError("cannot compute a confidence interval over an empty sequence")

    values = np.asarray([1.0 if outcome else 0.0 for outcome in outcomes])
    if bool(np.all(values == values[0])):
        point = float(values[0])
        return point, point

    result = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="BCa",
        random_state=random_state,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def cohens_kappa(
    labels_a: Sequence[str | None], labels_b: Sequence[str | None]
) -> float:
    """Cohen's kappa for agreement between two raters (spec Sec13.6: judge
    agreement against a human-labelled subset).

    Returns
    -------
    float
        `sklearn.metrics.cohen_kappa_score` returns `NaN` (with a warning)
        when both raters used exactly one, shared label throughout --
        agreement is total but the statistic is undefined by its own
        formula (division by a zero expected-agreement-beyond-chance term).
        This wrapper reports `0.0` there: no agreement is *measurable*
        beyond chance, which is the honest reading for a report table that
        must never carry `NaN`.

    Raises
    ------
    ValueError
        `labels_a` and `labels_b` differ in length -- a rater pair must
        rate the same set of cases.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"labels_a and labels_b must have the same length, got "
            f"{len(labels_a)} and {len(labels_b)}"
        )
    # The single-shared-label case is handled explicitly below (-> 0.0), so
    # sklearn's own "undefined, set to NaN" warning is expected noise here,
    # not a signal -- suppressed only for this one, already-handled call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = cohen_kappa_score(list(labels_a), list(labels_b))
    return 0.0 if np.isnan(kappa) else float(kappa)


def bootstrap_kappa_ci(
    labels_a: Sequence[str | None],
    labels_b: Sequence[str | None],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    random_state: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for `cohens_kappa(labels_a,
    labels_b)` (spec Sec13.6: judge agreement "with its interval").

    `n_resamples` defaults far lower than `bootstrap_proportion_ci`'s 9999:
    each resample here is a pure-Python `cohen_kappa_score` call rather
    than `scipy`'s vectorised C statistic, so the same resample count would
    cost orders of magnitude more wall-clock for the (very small, n~12)
    label sets this function is ever called on.

    A hand-rolled paired resample, not `scipy.stats.bootstrap` directly:
    `cohen_kappa_score` warns and returns `NaN` on a resample that happens
    to draw only one label, which `scipy`'s own BCa machinery would
    propagate into a `NaN` bound (the same failure mode
    `bootstrap_proportion_ci` avoids for the degenerate whole-sample case,
    here showing up per-resample instead). Routing every resample through
    this module's own `cohens_kappa` reuses its already-NaN-safe handling
    instead of re-solving the same problem twice.

    Parameters
    ----------
    labels_a, labels_b : sequence
        Paired ratings over the same cases -- resampled together (the same
        drawn indices apply to both), never independently, or the pairing
        that kappa measures would be destroyed.

    Raises
    ------
    ValueError
        The two sequences differ in length, or are empty.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"labels_a and labels_b must have the same length, got "
            f"{len(labels_a)} and {len(labels_b)}"
        )
    if not labels_a:
        raise ValueError("cannot compute a confidence interval over an empty sequence")

    rng = np.random.default_rng(random_state)
    n = len(labels_a)
    a = list(labels_a)
    b = list(labels_b)
    resampled_kappas = np.empty(n_resamples)
    for i in range(n_resamples):
        draw = rng.integers(0, n, size=n)
        resampled_kappas[i] = cohens_kappa([a[j] for j in draw], [b[j] for j in draw])

    if bool(np.all(resampled_kappas == resampled_kappas[0])):
        point = float(resampled_kappas[0])
        return point, point

    lower_pct = (1 - confidence_level) / 2 * 100
    upper_pct = (1 + confidence_level) / 2 * 100
    low, high = np.percentile(resampled_kappas, [lower_pct, upper_pct])
    return float(low), float(high)


def discriminates(verdicts: Sequence[str | None]) -> bool:
    """Whether `verdicts` shows any spread at all (spec Sec13.7: evaluator
    discriminating power; decision D6's saturation check).

    Abstentions (`None`) are excluded before judging spread -- an item that
    abstains on most cases and splits PASS/FAIL on the rest is
    discriminating on the cases it actually applies to.

    Returns
    -------
    bool
        `False` when fewer than two non-abstaining values remain, or every
        remaining value is identical (the exact hl8 defect this check
        exists to catch: an evaluator whose verdict never varies looks
        exactly like a passing gate).
    """
    non_abstaining = [verdict for verdict in verdicts if verdict is not None]
    if len(non_abstaining) < 2:
        return False
    return len(set(non_abstaining)) > 1
