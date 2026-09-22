from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_validation import passing_audit

from battery_dispatch.config import AEST, BatteryConfig
from battery_dispatch.horizon import run_rolling_horizon
from battery_dispatch.optimiser import (
    STRICT_SOLVER_POLICY,
    THROUGHPUT_TIE_BREAKER_AUD_PER_MWH,
    SolverPolicy,
    solve_window,
)
from battery_dispatch.validation import assert_audit_passes, audit_dispatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse_exclusivity_binding import binding_price_threshold  # noqa: E402


def make_asset(**changes: float) -> BatteryConfig:
    values = {
        "generation_loss_factor": 0.9652,
        "load_loss_factor": 0.9681,
        "power_mw": 1.0,
        "energy_mwh": 1.0,
        "round_trip_efficiency": 0.87,
        "soc_min_fraction": 0.0,
        "soc_max_fraction": 1.0,
        "interval_minutes": 60,
    }
    values.update(changes)
    return BatteryConfig(**values)


def test_policy_rejects_impossible_tolerances() -> None:
    with pytest.raises(ValueError, match="mip_rel_gap"):
        SolverPolicy(mip_rel_gap=0.0)
    with pytest.raises(ValueError, match="mip_rel_gap"):
        SolverPolicy(mip_rel_gap=1.0)


def test_strict_policy_audits_against_the_gap_it_asked_for() -> None:
    assert STRICT_SOLVER_POLICY.gap_tolerance == pytest.approx(1.01e-6)
    assert SolverPolicy(mip_rel_gap=1e-3).gap_tolerance == pytest.approx(1.01e-3)


def test_solve_window_reports_a_valid_optimality_bound() -> None:
    battery = make_asset()
    prices = np.array([10.0, 20.0, 100.0, 30.0], dtype=float)
    result = solve_window(prices, battery, soc_initial=0.0)
    # Maximisation: the dual bound can never sit below the objective HiGHS
    # optimised, and the recorded absolute gap is exactly that distance.
    assert result.solver_dual_bound >= result.solver_objective_value - 1e-6
    assert result.solver_absolute_gap == pytest.approx(
        abs(result.solver_dual_bound - result.solver_objective_value)
    )

    # The reported economic objective excludes the throughput tie-breaker, so
    # it sits above the solved objective by exactly the throughput penalty.
    throughput = float(result.charge_mw.sum() + result.discharge_mw.sum())
    penalty = battery.interval_hours * THROUGHPUT_TIE_BREAKER_AUD_PER_MWH * throughput
    assert result.objective_value - result.solver_objective_value == pytest.approx(
        penalty, abs=1e-6
    )


def test_binding_threshold_predicts_when_exclusivity_pays() -> None:
    """Simultaneous operation is only attractive below the derived threshold."""
    battery = make_asset()
    for deg_cost in (0.0, 25.0, 50.0):
        threshold = binding_price_threshold(battery, deg_cost)
        round_trip = battery.charge_efficiency * battery.discharge_efficiency
        dt = battery.interval_hours

        def simultaneous_value(
            price: float,
            *,
            dt: float = dt,
            deg_cost: float = deg_cost,
            round_trip: float = round_trip,
        ) -> float:
            charge = -dt * battery.load_loss_factor * price
            discharge = dt * (battery.generation_loss_factor * price - deg_cost)
            return charge + discharge * round_trip

        assert simultaneous_value(threshold - 1.0) > 0
        assert simultaneous_value(threshold + 1.0) < 0
    assert binding_price_threshold(make_asset(), 25.0) == pytest.approx(-169.42, abs=0.01)


def test_a_declared_gap_tolerance_is_never_implied() -> None:
    """A scenario may declare a looser gap, but only explicitly."""
    audit = passing_audit(max_solver_mip_gap=8e-4)
    with pytest.raises(ValueError, match="MIP gap"):
        assert_audit_passes(audit)
    assert_audit_passes(audit, mip_gap_tolerance=1e-3)


def test_rolling_horizon_records_the_policy_it_was_given() -> None:
    timestamps = pd.date_range("2023-01-01 00:05", periods=4, freq="5min", tz=AEST)
    prices = pd.DataFrame({"settlementdate": timestamps, "rrp": [0.0, 90.0, 0.0, 90.0]})
    battery = make_asset(interval_minutes=5, round_trip_efficiency=1.0)
    trace = run_rolling_horizon(
        prices,
        battery,
        soc_initial=0.0,
        window_hours=1,
        step_hours=1,
        policy=SolverPolicy(mip_rel_gap=1e-3),
    )
    for column in ("solver_dual_bound", "solver_absolute_gap"):
        assert column in trace
    audit = audit_dispatch(trace, battery, soc_initial=0.0)
    assert audit["total_solver_absolute_gap_aud"] >= 0.0
    assert_audit_passes(
        {**passing_audit(), **audit}, mip_gap_tolerance=1e-3
    )
