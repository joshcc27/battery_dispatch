"""Single-window battery MILP built with linopy and solved by HiGHS."""

from __future__ import annotations

from dataclasses import dataclass

import linopy
import numpy as np
import pandas as pd

from .battery import energy_balance_residual
from .config import BatteryConfig
from .settlement import settlement_coefficients


@dataclass(frozen=True)
class DispatchWindow:
    """Optimised meter-side dispatch and end-of-interval SOC arrays."""

    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    soc_initial_mwh: float
    objective_value: float

    @property
    def soc_final_mwh(self) -> float:
        return float(self.soc_mwh[-1]) if len(self.soc_mwh) else self.soc_initial_mwh

    def to_frame(self, settlementdate: pd.Index | np.ndarray) -> pd.DataFrame:
        timestamps = pd.Index(settlementdate)
        if len(timestamps) != len(self.charge_mw):
            raise ValueError("Timestamp and dispatch lengths differ")
        return pd.DataFrame(
            {
                "settlementdate": timestamps,
                "charge_mw": self.charge_mw,
                "discharge_mw": self.discharge_mw,
                "soc_mwh": self.soc_mwh,
            }
        )


def solve_window(
    prices: np.ndarray,
    asset: BatteryConfig,
    soc_initial: float,
    deg_cost: float = 0.0,
    *,
    settlementdate: object | None = None,
) -> DispatchWindow:
    """Solve one perfect-foresight energy-arbitrage window.

    The objective uses the same dated loss factors and degradation convention
    as :func:`battery_dispatch.settlement.settle`.
    """
    price = np.asarray(prices, dtype=float)
    if price.ndim != 1 or len(price) == 0:
        raise ValueError("prices must be a non-empty one-dimensional array")
    if not np.isfinite(price).all():
        raise ValueError("prices contains NaN or infinite values")
    if deg_cost < 0:
        raise ValueError("deg_cost must be non-negative")
    if not asset.soc_min_mwh <= soc_initial <= asset.soc_max_mwh:
        raise ValueError("soc_initial is outside the configured SOC envelope")

    intervals = pd.RangeIndex(len(price), name="interval")
    model = linopy.Model()
    charge = model.add_variables(
        lower=0.0, upper=asset.power_mw, coords=[intervals], name="charge_mw"
    )
    discharge = model.add_variables(
        lower=0.0, upper=asset.power_mw, coords=[intervals], name="discharge_mw"
    )
    soc = model.add_variables(
        lower=asset.soc_min_mwh,
        upper=asset.soc_max_mwh,
        coords=[intervals],
        name="soc_mwh",
    )
    mode = model.add_variables(binary=True, coords=[intervals], name="charge_mode")

    model.add_constraints(charge <= asset.power_mw * mode, name="charge_mode_limit")
    model.add_constraints(
        discharge <= asset.power_mw * (1 - mode), name="discharge_mode_limit"
    )
    eta_c = asset.charge_efficiency
    eta_d = asset.discharge_efficiency
    dt = asset.interval_hours
    stored_energy_change = dt * (eta_c * charge - discharge / eta_d)
    model.add_constraints(
        soc.sel(interval=0) == soc_initial + stored_energy_change.sel(interval=0),
        name="soc_balance_initial",
    )
    if len(intervals) > 1:
        model.add_constraints(
            soc.diff("interval").isel(interval=slice(1, None))
            == stored_energy_change.isel(interval=slice(1, None)),
            name="soc_balance",
        )

    cashflows = settlement_coefficients(
        price,
        asset,
        settlementdate=settlementdate,
        deg_cost=deg_cost,
    )
    objective = (
        cashflows.discharge_value_per_mw * discharge
        + cashflows.charge_value_per_mw * charge
    ).sum()
    model.add_objective(objective, sense="max")
    status, termination = model.solve(solver_name="highs", output_flag=False)
    if str(status).lower() != "ok" or str(termination).lower() != "optimal":
        raise RuntimeError(f"HiGHS failed: status={status}, termination={termination}")

    def clean(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        result[np.abs(result) < 1e-8] = 0.0
        return result

    charge_values = clean(charge.solution.to_numpy())
    discharge_values = clean(discharge.solution.to_numpy())
    soc_values = clean(soc.solution.to_numpy())
    residual = energy_balance_residual(
        charge_values,
        discharge_values,
        soc_initial,
        float(soc_values[-1]),
        asset,
    )
    if abs(residual) > 1e-5:
        raise RuntimeError(f"Solver result fails energy balance: residual={residual}")
    return DispatchWindow(
        charge_mw=charge_values,
        discharge_mw=discharge_values,
        soc_mwh=soc_values,
        soc_initial_mwh=float(soc_initial),
        objective_value=float(model.objective.value),
    )
