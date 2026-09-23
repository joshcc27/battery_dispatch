"""Full-period perfect-foresight revenue ceiling for a rolling-horizon run."""

from __future__ import annotations

import pandas as pd

from .config import BatteryConfig
from .horizon import _normalise_prices
from .optimiser import solve_window
from .settlement import RevenueResult


def perfect_foresight_ceiling(
    prices: pd.DataFrame | pd.Series,
    asset: BatteryConfig,
    *,
    soc_initial: float,
    deg_cost: float = 0.0,
    terminal_value_per_mwh: float = 0.0,
) -> dict[str, float]:
    """Solve the whole period as one LP with the exclusivity binary removed.

    Removing the rolling look-ahead limit and the exclusivity constraint can
    only raise the optimum, and the throughput tie-breaker is excluded, so the
    value bounds every feasible dispatch of the same asset, prices, starting SOC
    and terminal valuation from above. The binaries are what make the MILP
    expensive; the LP solves a full financial year in seconds.
    """
    frame = _normalise_prices(prices)
    result = solve_window(
        frame["rrp"].to_numpy(dtype=float),
        asset,
        soc_initial,
        deg_cost=deg_cost,
        settlementdate=frame["settlementdate"].to_numpy(),
        terminal_value_per_mwh=terminal_value_per_mwh,
        enforce_exclusivity=False,
        throughput_tie_breaker=0.0,
    )
    return {
        "perfect_foresight_ceiling_aud": result.objective_value,
        "ceiling_solve_seconds": result.solve_seconds,
    }


def audit_ceiling(
    ceiling: dict[str, float],
    revenue: RevenueResult,
    *,
    terminal_value_per_mwh: float = 0.0,
) -> dict[str, float]:
    """Compare a settled run with its ceiling on the ceiling's own objective.

    The ceiling values closing SOC at ``terminal_value_per_mwh``, so the
    achieved side adds the same valuation to settled cash revenue.
    """
    terminal_soc = float(revenue.intervals["soc_mwh"].iloc[-1])
    achieved = revenue.total_revenue + terminal_value_per_mwh * terminal_soc
    bound = float(ceiling["perfect_foresight_ceiling_aud"])
    gap = bound - achieved
    return {
        **ceiling,
        "achieved_objective_aud": achieved,
        "horizon_gap_aud": gap,
        "horizon_gap_fraction": gap / bound if bound > 0 else 0.0,
    }
