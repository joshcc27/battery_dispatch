"""Battery physics and dispatch feasibility checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BatteryConfig


REQUIRED_TRACE_COLUMNS = {
    "settlementdate",
    "charge_mw",
    "discharge_mw",
    "soc_mwh",
}


def soc_path(
    charge_mw: np.ndarray,
    discharge_mw: np.ndarray,
    soc_initial: float,
    asset: BatteryConfig,
) -> np.ndarray:
    """Return end-of-interval stored energy for a meter-side dispatch path."""
    charge = np.asarray(charge_mw, dtype=float)
    discharge = np.asarray(discharge_mw, dtype=float)
    if charge.shape != discharge.shape:
        raise ValueError("Charge and discharge arrays must have the same shape")
    changes = asset.interval_hours * (
        asset.charge_efficiency * charge
        - discharge / asset.discharge_efficiency
    )
    return soc_initial + np.cumsum(changes)


def energy_balance_residual(
    charge_mw: np.ndarray,
    discharge_mw: np.ndarray,
    soc_initial: float,
    soc_final: float,
    asset: BatteryConfig,
) -> float:
    """Return input less output stored energy minus the observed SOC change."""
    charge = np.asarray(charge_mw, dtype=float)
    discharge = np.asarray(discharge_mw, dtype=float)
    physical_change = asset.interval_hours * (
        asset.charge_efficiency * charge.sum()
        - discharge.sum() / asset.discharge_efficiency
    )
    return float(physical_change - (soc_final - soc_initial))


def validate_dispatch_trace(
    trace: pd.DataFrame,
    asset: BatteryConfig,
    *,
    soc_initial: float | None = None,
    tolerance: float = 1e-6,
) -> None:
    """Fail if a dispatch trace violates its schema or battery physics."""
    missing = REQUIRED_TRACE_COLUMNS.difference(trace.columns)
    if missing:
        raise ValueError(f"Dispatch trace is missing columns: {sorted(missing)}")
    if trace.empty:
        raise ValueError("Dispatch trace is empty")
    numeric = trace[["charge_mw", "discharge_mw", "soc_mwh"]]
    if numeric.isna().any().any():
        raise ValueError("Dispatch trace contains NaNs")
    charge = trace["charge_mw"].to_numpy(dtype=float)
    discharge = trace["discharge_mw"].to_numpy(dtype=float)
    soc = trace["soc_mwh"].to_numpy(dtype=float)
    if np.any(charge < -tolerance) or np.any(discharge < -tolerance):
        raise ValueError("Charge and discharge must be non-negative")
    if np.any(charge > asset.power_mw + tolerance) or np.any(
        discharge > asset.power_mw + tolerance
    ):
        raise ValueError("Dispatch exceeds the asset power rating")
    if np.any(charge * discharge > tolerance):
        raise ValueError("Simultaneous charging and discharging detected")
    if np.any(soc < asset.soc_min_mwh - tolerance) or np.any(
        soc > asset.soc_max_mwh + tolerance
    ):
        raise ValueError("SOC is outside the configured operating envelope")
    if soc_initial is not None:
        expected = soc_path(charge, discharge, soc_initial, asset)
        if not np.allclose(soc, expected, rtol=0.0, atol=tolerance):
            raise ValueError("Dispatch trace fails the interval energy balance")
