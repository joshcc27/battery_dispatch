from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_validation import passing_audit

from battery_dispatch.ceiling import audit_ceiling, perfect_foresight_ceiling
from battery_dispatch.config import AEST, BatteryConfig
from battery_dispatch.horizon import run_rolling_horizon
from battery_dispatch.optimiser import solve_window
from battery_dispatch.settlement import settle
from battery_dispatch.validation import assert_audit_passes


def asset(**changes: float) -> BatteryConfig:
    values = {
        "generation_loss_factor": 1.0,
        "load_loss_factor": 1.0,
        "power_mw": 1.0,
        "energy_mwh": 1.0,
        "round_trip_efficiency": 1.0,
        "soc_min_fraction": 0.0,
        "soc_max_fraction": 1.0,
        "interval_minutes": 60,
    }
    values.update(changes)
    return BatteryConfig(**values)


def prices(values: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01 01:00", periods=len(values), freq="h", tz=AEST)
    return pd.DataFrame({"settlementdate": timestamps, "rrp": values})


def test_ceiling_measures_what_a_myopic_horizon_leaves_behind() -> None:
    """A one-hour window never sees a later price, so it cannot trade at all."""
    battery = asset()
    frame = prices([0.0, 100.0, 0.0, 200.0])
    trace = run_rolling_horizon(frame, battery, soc_initial=0, window_hours=1, step_hours=1)
    revenue = settle(trace, frame, battery)
    audit = audit_ceiling(perfect_foresight_ceiling(frame, battery, soc_initial=0), revenue)
    # Hand-derived: charge 1 MWh at 0, sell at 100, charge at 0, sell at 200.
    assert audit["perfect_foresight_ceiling_aud"] == pytest.approx(300.0)
    assert revenue.total_revenue == pytest.approx(0.0)
    assert audit["horizon_gap_aud"] == pytest.approx(300.0)
    assert audit["horizon_gap_fraction"] == pytest.approx(1.0)


def test_ceiling_matches_the_exact_optimum_when_the_horizon_sees_everything() -> None:
    battery = asset(round_trip_efficiency=0.81)
    values = [10.0, 20.0, 100.0, 30.0, 20.0, 10.0]
    exact = solve_window(np.array(values), battery, 0.0)
    ceiling = perfect_foresight_ceiling(prices(values), battery, soc_initial=0.0)
    assert ceiling["perfect_foresight_ceiling_aud"] == pytest.approx(
        exact.objective_value, abs=1e-6
    )


def test_ceiling_values_terminal_soc_on_both_sides() -> None:
    """With salvage above every price, holding energy is the optimum and gap is 0."""
    battery = asset()
    frame = prices([10.0, 20.0])
    trace = run_rolling_horizon(
        frame, battery, soc_initial=0, window_hours=2, step_hours=2,
        terminal_value_per_mwh=500.0,
    )
    revenue = settle(trace, frame, battery)
    audit = audit_ceiling(
        perfect_foresight_ceiling(frame, battery, soc_initial=0, terminal_value_per_mwh=500.0),
        revenue,
        terminal_value_per_mwh=500.0,
    )
    assert audit["perfect_foresight_ceiling_aud"] == pytest.approx(490.0)
    assert audit["horizon_gap_aud"] == pytest.approx(0.0, abs=1e-6)


def test_result_above_the_ceiling_fails_the_audit() -> None:
    with pytest.raises(ValueError, match="ceiling"):
        assert_audit_passes(
            passing_audit(perfect_foresight_ceiling_aud=1_000.0, horizon_gap_aud=-5.0)
        )
    # Solver-tolerance noise below the ceiling is not a failure.
    assert_audit_passes(
        passing_audit(perfect_foresight_ceiling_aud=1_000.0, horizon_gap_aud=-0.5)
    )
