"""Single-window battery MILP built with linopy and solved by HiGHS."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import linopy
import numpy as np
import pandas as pd

from .battery import energy_balance_residual
from .config import BatteryConfig
from .settlement import settlement_coefficients


# A secondary numerical preference, not a reported degradation or settlement
# cost. It removes economically indifferent cycling/branching while being less
# than or equal to one cent per 100 MWh of throughput.
THROUGHPUT_TIE_BREAKER_AUD_PER_MWH = 1e-4
SOLVER_RELATIVE_MIP_GAP = 1e-6


@dataclass(frozen=True)
class SolverPolicy:
    """Per-scenario solver tolerances and what the audit will accept.

    The default is the strict policy used for every reported result except the
    A$0/MWh degradation scenario, where exclusivity binds at 23% of intervals
    and proving a 1e-6 relative gap is not tractable. See docs/VALIDATION.md.
    """

    mip_rel_gap: float = SOLVER_RELATIVE_MIP_GAP
    time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.mip_rel_gap < 1:
            raise ValueError("mip_rel_gap must be in (0, 1)")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive when set")

    @property
    def allows_time_limited_windows(self) -> bool:
        return self.time_limit_seconds is not None

    @property
    def gap_tolerance(self) -> float:
        """The largest achieved relative gap the audit will accept.

        A window stopped by the time limit can report a gap wider than the
        requested tolerance, so a time-limited policy is audited on the gap
        actually achieved and recorded, not on the one requested.
        """
        return float("inf") if self.allows_time_limited_windows else 1.01 * self.mip_rel_gap


STRICT_SOLVER_POLICY = SolverPolicy()


@dataclass(frozen=True)
class DispatchWindow:
    """Optimised meter-side dispatch and end-of-interval SOC arrays."""

    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    soc_initial_mwh: float
    objective_value: float
    solver_status: str
    solver_termination: str
    solve_seconds: float
    solver_mip_gap: float
    # The objective HiGHS actually optimised, including the throughput
    # tie-breaker. ``objective_value`` above excludes it and is the economic
    # figure; the solver bounds below are only comparable to this one.
    solver_objective_value: float
    solver_dual_bound: float
    solver_absolute_gap: float
    solver_time_limited: bool

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
    terminal_value_per_mwh: float = 0.0,
    policy: SolverPolicy = STRICT_SOLVER_POLICY,
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
    if terminal_value_per_mwh < 0:
        raise ValueError("terminal_value_per_mwh must be non-negative")
    soc_tolerance = 1e-8
    if (
        soc_initial < asset.soc_min_mwh - soc_tolerance
        or soc_initial > asset.soc_max_mwh + soc_tolerance
    ):
        raise ValueError("soc_initial is outside the configured SOC envelope")
    # HiGHS solutions can land a few machine epsilons outside a bound. Do not
    # let harmless floating-point drift break the next rolling window.
    soc_initial = float(np.clip(soc_initial, asset.soc_min_mwh, asset.soc_max_mwh))

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
        - asset.interval_hours
        * THROUGHPUT_TIE_BREAKER_AUD_PER_MWH
        * (charge + discharge)
    ).sum() + terminal_value_per_mwh * soc.sel(interval=intervals[-1])
    model.add_objective(objective, sense="max")
    solver_options: dict[str, object] = {
        "solver_name": "highs",
        "output_flag": False,
        "mip_rel_gap": policy.mip_rel_gap,
    }
    if policy.time_limit_seconds is not None:
        solver_options["time_limit"] = float(policy.time_limit_seconds)
    solve_started = perf_counter()
    status, termination = model.solve(**solver_options)
    solve_seconds = perf_counter() - solve_started
    termination_text = str(termination).lower()
    # A time-limited window is accepted only when the policy asked for a time
    # limit. Everything else, including a silently truncated solve, still fails.
    accepted = {"optimal"}
    if policy.allows_time_limited_windows:
        accepted.add("time_limit")
    if str(status).lower() != "ok" or termination_text not in accepted:
        raise RuntimeError(f"HiGHS failed: status={status}, termination={termination}")
    time_limited = termination_text != "optimal"
    info = model.solver_model.getInfo()
    solver_mip_gap = float(info.mip_gap)
    # HiGHS reports both in its own objective sense, so the difference is a
    # valid absolute optimality gap regardless of how linopy flipped the sign.
    solver_dual_bound = float(info.mip_dual_bound)
    solver_objective_value = float(info.objective_function_value)
    solver_absolute_gap = abs(solver_dual_bound - solver_objective_value)
    if not np.isfinite([solver_mip_gap, solver_dual_bound, solver_absolute_gap]).all():
        raise RuntimeError("HiGHS returned a non-finite optimality bound")

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
    economic_objective = float(
        np.dot(cashflows.discharge_value_per_mw, discharge_values)
        + np.dot(cashflows.charge_value_per_mw, charge_values)
        + terminal_value_per_mwh * soc_values[-1]
    )
    return DispatchWindow(
        charge_mw=charge_values,
        discharge_mw=discharge_values,
        soc_mwh=soc_values,
        soc_initial_mwh=float(soc_initial),
        objective_value=economic_objective,
        solver_status=str(status),
        solver_termination=str(termination),
        solve_seconds=solve_seconds,
        solver_mip_gap=solver_mip_gap,
        solver_objective_value=solver_objective_value,
        solver_dual_bound=solver_dual_bound,
        solver_absolute_gap=solver_absolute_gap,
        solver_time_limited=time_limited,
    )
