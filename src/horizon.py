"""Rolling-horizon orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery import validate_dispatch_trace
from .config import BatteryConfig
from .optimiser import solve_window


def _normalise_prices(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        frame = prices.rename("rrp").rename_axis("settlementdate").reset_index()
    else:
        frame = prices.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        if "settlementdate" not in frame and isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.rename_axis("settlementdate").reset_index()
    if not {"settlementdate", "rrp"}.issubset(frame.columns):
        raise ValueError("prices must contain settlementdate and rrp")
    frame = frame[["settlementdate", "rrp"]].sort_values("settlementdate").reset_index(drop=True)
    if frame["settlementdate"].duplicated().any():
        raise ValueError("prices contain duplicate timestamps")
    if frame["rrp"].isna().any():
        raise ValueError("prices contain NaNs")
    return frame


def run_rolling_horizon(
    prices: pd.DataFrame | pd.Series,
    asset: BatteryConfig,
    *,
    soc_initial: float | None = None,
    window_hours: int = 48,
    step_hours: int = 24,
    deg_cost: float = 0.0,
) -> pd.DataFrame:
    """Solve 48 hours, implement 24 hours, carry SOC, and repeat.

    This physical dispatch policy can cross financial-year boundaries. Dated
    MLFs are applied later by settlement and do not enter the core optimiser.
    """
    frame = _normalise_prices(prices)
    intervals_per_hour = 60 // asset.interval_minutes
    if 60 % asset.interval_minutes:
        raise ValueError("interval_minutes must divide one hour")
    window_size = window_hours * intervals_per_hour
    step_size = step_hours * intervals_per_hour
    if window_size < step_size or step_size <= 0:
        raise ValueError("Require window_hours >= step_hours > 0")

    current_soc = asset.initial_soc_mwh if soc_initial is None else float(soc_initial)
    implemented: list[pd.DataFrame] = []
    start = 0
    while start < len(frame):
        stop = min(start + window_size, len(frame))
        window = frame.iloc[start:stop]
        result = solve_window(
            window["rrp"].to_numpy(dtype=float),
            asset,
            current_soc,
            deg_cost=deg_cost,
            settlementdate=window["settlementdate"].to_numpy(),
        )
        take = min(step_size, len(window))
        part = result.to_frame(window["settlementdate"].to_numpy()).iloc[:take].copy()
        implemented.append(part)
        current_soc = float(part["soc_mwh"].iloc[-1])
        start += take

    trace = pd.concat(implemented, ignore_index=True)
    starting_soc = asset.initial_soc_mwh if soc_initial is None else float(soc_initial)
    validate_dispatch_trace(trace, asset, soc_initial=starting_soc)
    return trace
