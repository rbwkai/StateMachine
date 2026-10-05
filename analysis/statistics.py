"""
analysis/statistics.py
=====================
Interval estimation, paired significance testing and small-cell reporting for the
analysis layer (research-plan checklist 10, "Analysis").

Three helpers, all implemented on the standard library so `analysis` keeps
importing without the inference stack (AGENTS.md §4):

1. `wilson_interval`   - binomial interval for one cell accuracy, so a cell
                         accuracy is reported with an error bar and not as a bare
                         point estimate.
2. `mcnemar_test`      - exact paired test for a direct-vs-CoT comparison on the
                         same items (the two arms are paired by instance id, so an
                         unpaired test would be wrong).
3. `flag_low_success_cells` - cells whose success count is too small for the
                         reported accuracy to be meaningful.

The exact (binomial) McNemar p-value is used rather than the chi-square
approximation because the discordant-pair count in a prompt ablation can be
small, which is exactly where the chi-square approximation is worst.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ==== Constants ====
# Every threshold here is analysis-owned; they belong next to
# generator/constants.py::FAILURE_THRESHOLD_TAU once that module is importable
# from analysis without a cycle (generator.probes imports analysis today).

# Two-sided 95% coverage: z_{0.975} for the normal quantile.
CONFIDENCE_LEVEL: float = 0.95
Z_TWO_SIDED_95: float = 1.959963984540054

# Checklist 10: report cells with fewer than 50 successes. Below this the
# accuracy point estimate moves by more than the interval half-width under a
# single extra success, so the cell is reported but flagged as unreliable.
MIN_SUCCESSES_FOR_REPORTING: int = 50


# ==== Interval estimation ====


def wilson_interval(
    successes: int,
    trials: int,
    z: float = Z_TWO_SIDED_95,
) -> Tuple[float, float]:
    """
    Return the (low, high) bounds of a two-sided Wilson score interval.

    `successes` of `trials` returns the interval for that cell's accuracy, so a
    reported accuracy always carries its error bar.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")

    p_hat = successes / trials
    z_sq = z ** 2
    denominator = 1.0 + z_sq / trials
    center = (p_hat + z_sq / (2.0 * trials)) / denominator
    margin = (
        z / denominator
        * math.sqrt(p_hat * (1.0 - p_hat) / trials + z_sq / (4.0 * trials * trials))
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


# ==== Paired significance testing ====


@dataclass(frozen=True)
class McNemarResult:
    """Exact paired test on the discordant pairs of two arms."""

    b: int
    c: int
    n_pairs: int
    n_discordant: int
    statistic: float
    p_value: float
    method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "b": self.b,
            "c": self.c,
            "n_pairs": self.n_pairs,
            "n_discordant": self.n_discordant,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method": self.method,
        }


def _binomial_two_sided_p_value(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value under H0: p = 0.5 on the discordant pairs."""
    if trials == 0:
        return 1.0

    tail = min(successes, trials - successes)
    cumulative = sum(math.comb(trials, i) for i in range(tail + 1))
    p_value = 2.0 * cumulative / (2 ** trials)
    return min(1.0, p_value)


def mcnemar_test(b: int, c: int, n_pairs: Optional[int] = None) -> Dict[str, Any]:
    """
    McNemar's exact test for paired binary outcomes.

    `b` is the number of items the first arm got right and the second arm wrong;
    `c` is the reverse. Items both arms answer alike carry no information and are
    excluded, so only the discordant pairs enter the test. `n_pairs` is the total
    number of paired items, recorded for reporting only.

    Returns a dict with the exact two-sided p-value, which is the quantity a
    direct-vs-CoT comparison reports. The chi-square statistic with continuity
    correction is included as a descriptive number only; it is not used for the
    decision because it is unreliable at small discordant counts.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")

    n_discordant = b + c
    statistic = (abs(b - c) - 1.0) ** 2 / n_discordant if n_discordant > 0 else 0.0
    return McNemarResult(
        b=b,
        c=c,
        n_pairs=n_pairs if n_pairs is not None else n_discordant,
        n_discordant=n_discordant,
        statistic=statistic,
        p_value=_binomial_two_sided_p_value(b, n_discordant),
        method="exact_binomial",
    ).to_dict()


# ==== Small-cell reporting ====


def flag_low_success_cells(
    cells: Sequence[Dict[str, Any]],
    min_successes: int = MIN_SUCCESSES_FOR_REPORTING,
) -> List[Dict[str, Any]]:
    """
    Return the cells whose success count is below the reporting threshold.

    A cell is one (condition, factor) row with a `correct` count and a `total`.
    Flagged cells are returned as copies carrying `low_success` and
    `min_successes`, so the caller's own rows are not mutated.
    """
    flagged: List[Dict[str, Any]] = []
    for cell in cells:
        successes = int(cell.get("correct", 0))
        if successes < min_successes:
            entry = dict(cell)
            entry["low_success"] = True
            entry["min_successes"] = min_successes
            flagged.append(entry)
    return flagged
