"""Single-window battery MILP built with linopy and solved by HiGHS."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import linopy
import numpy as np
import pandas as pd

from .battery import energy_balance_residual
from .config import BatteryConfig
from .settlement import SettlementCoefficients, settlement_coefficients

# A secondary numerical preference, not a reported degradation or settlement
# cost. It removes economically indifferent cycling/branching while being less
# than or equal to one cent per 100 MWh of throughput.
THROUGHPUT_TIE_BREAKER_AUD_PER_MWH = 1e-4


SOLVER_RELATIVE_MIP_GAP = 1e-6


@dataclass(frozen=True)
class SolverPolicy:
    """The solver tolerance a scenario declares and the audit enforces.

    Every reported scenario uses the strict default. The exclusivity reduction
    in :func:`exclusivity_binding_mask` keeps the reference month tractable at
    this gap; see docs/VALIDATION.md.
    """

    mip_rel_gap: float = SOLVER_RELATIVE_MIP_GAP

    def __post_init__(self) -> None:
        if not 0 < self.mip_rel_gap < 1:
            raise ValueError("mip_rel_gap must be in (0, 1)")

    @property
    def gap_tolerance(self) -> float:
        """The largest achieved relative gap the audit will accept."""
        return 1.01 * self.mip_rel_gap

    @property
    def label(self) -> str:
        return f"fixed {self.mip_rel_gap:g}"

    def describe(self) -> dict[str, object]:
        """The policy's identity for a run's config hash."""
        return {"kind": "fixed", "mip_rel_gap": self.mip_rel_gap}


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
    # How many intervals kept an exclusivity binary. The rest were proven
    # redundant by ``exclusivity_binding_mask`` and dropped from the MILP.
    exclusivity_binary_count: int

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


def exclusivity_binding_mask(
    cashflows: SettlementCoefficients,
    asset: BatteryConfig,
    *,
    margin: float = 1e-9,
) -> np.ndarray:
    """Return which intervals still need an exclusivity binary.

    Shifting a solution that charges ``c`` and discharges ``d`` in one interval
    by ``-a`` on charge and ``-a * eta_c * eta_d`` on discharge leaves the whole
    SOC path unchanged, so the objective moves by ``-a * (A_c + eta_rt * A_d)``
    using only this interval's settled coefficients. Where that bracket is
    strictly negative, simultaneous operation is strictly dominated in every
    optimal solution, so the binary is redundant rather than merely slack and
    can be dropped without changing the optimum.

    The bracket is evaluated directly instead of via the equivalent price
    threshold ``p < -6.776 * deg``, so dated loss factors and an adverse sign of
    ``L - G * eta_rt`` are both handled without rearrangement.
    See docs/VALIDATION.md.
    """
    eta_rt = asset.charge_efficiency * asset.discharge_efficiency
    burn_value = (
        cashflows.charge_value_per_mw + eta_rt * cashflows.discharge_value_per_mw
    )
    return np.asarray(burn_value >= -margin)


def solve_window(
    prices: np.ndarray,
    asset: BatteryConfig,
    soc_initial: float,
    deg_cost: float = 0.0,
    *,
    settlementdate: object | None = None,
    terminal_value_per_mwh: float = 0.0,
    policy: SolverPolicy = STRICT_SOLVER_POLICY,
    enforce_exclusivity: bool = True,
    throughput_tie_breaker: float = THROUGHPUT_TIE_BREAKER_AUD_PER_MWH,
) -> DispatchWindow:
    """Solve one perfect-foresight energy-arbitrage window.

    The objective uses the same dated loss factors and degradation convention
    as :func:`battery_dispatch.settlement.settle`.

    ``enforce_exclusivity=False`` with a zero tie-breaker solves the pure LP
    relaxation used as a revenue ceiling; see :mod:`battery_dispatch.ceiling`.
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
    if throughput_tie_breaker < 0:
        raise ValueError("throughput_tie_breaker must be non-negative")
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
    cashflows = settlement_coefficients(
        price,
        asset,
        settlementdate=settlementdate,
        deg_cost=deg_cost,
    )
    # One binary per interval is redundant: simultaneous charge and discharge is
    # strictly dominated wherever burning energy does not pay, which is all but
    # 0.55% of FY2022-23 intervals at the headline A$25/MWh degradation cost.
    binding = (
        exclusivity_binding_mask(cashflows, asset)
        if enforce_exclusivity
        else np.zeros(len(price), dtype=bool)
    )
    binding_positions = np.flatnonzero(binding)
    if len(binding_positions):
        binding_index = pd.Index(binding_positions, name="interval")
        mode = model.add_variables(
            binary=True, coords=[binding_index], name="charge_mode"
        )
        model.add_constraints(
            charge.sel(interval=binding_index) <= asset.power_mw * mode,
            name="charge_mode_limit",
        )
        model.add_constraints(
            discharge.sel(interval=binding_index) <= asset.power_mw * (1 - mode),
            name="discharge_mode_limit",
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

    objective = (
        cashflows.discharge_value_per_mw * discharge
        + cashflows.charge_value_per_mw * charge
        - asset.interval_hours
        * throughput_tie_breaker
        * (charge + discharge)
    ).sum() + terminal_value_per_mwh * soc.sel(interval=intervals[-1])
    model.add_objective(objective, sense="max")
    solver_options: dict[str, object] = {
        "solver_name": "highs",
        "output_flag": False,
        "mip_rel_gap": policy.mip_rel_gap,
    }
    solve_started = perf_counter()
    status, termination = model.solve(**solver_options)
    solve_seconds = perf_counter() - solve_started
    termination_text = str(termination).lower()
    if str(status).lower() != "ok" or termination_text != "optimal":
        raise RuntimeError(f"HiGHS failed: status={status}, termination={termination}")
    info = model.solver_model.getInfo()
    solver_objective_value = float(info.objective_function_value)
    if len(binding_positions):
        solver_mip_gap = float(info.mip_gap)
        # HiGHS reports both in its own objective sense, so the difference is a
        # valid absolute optimality gap regardless of how linopy flipped the sign.
        solver_dual_bound = float(info.mip_dual_bound)
        solver_absolute_gap = abs(solver_dual_bound - solver_objective_value)
    else:
        # Every binary was proven redundant, so HiGHS solved a pure LP. Its
        # simplex optimum is exact and there is no branch-and-bound gap; the
        # mip_* fields are undefined for an LP and must not be read.
        solver_mip_gap = 0.0
        solver_dual_bound = solver_objective_value
        solver_absolute_gap = 0.0
    if not np.isfinite(
        [solver_mip_gap, solver_dual_bound, solver_objective_value, solver_absolute_gap]
    ).all():
        raise RuntimeError("HiGHS returned a non-finite optimality bound")

    def clean(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        result[np.abs(result) < 1e-8] = 0.0
        return result

    charge_values = clean(charge.solution.to_numpy())
    discharge_values = clean(discharge.solution.to_numpy())
    soc_values = clean(soc.solution.to_numpy())
    if not (
        np.isfinite(charge_values).all()
        and np.isfinite(discharge_values).all()
        and np.isfinite(soc_values).all()
    ):
        raise RuntimeError("HiGHS returned a non-finite dispatch solution")
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
        exclusivity_binary_count=int(len(binding_positions)),
    )

