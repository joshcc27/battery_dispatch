from __future__ import annotations

import pandas as pd
import pytest

from battery_dispatch.config import AEST, BatteryConfig
from battery_dispatch.horizon import run_rolling_horizon
from battery_dispatch.settlement import settle
from battery_dispatch.validation import (
    assert_audit_passes,
    audit_completed_cycles,
    audit_dispatch,
    audit_prices,
)


def asset() -> BatteryConfig:
    return BatteryConfig(
        generation_loss_factor=1,
        load_loss_factor=1,
        power_mw=1,
        energy_mwh=1,
        round_trip_efficiency=1,
        soc_min_fraction=0,
        soc_max_fraction=1,
        interval_minutes=5,
    )


def passing_audit(**overrides: object) -> dict[str, object]:
    """A complete, green dispatch audit that a test can spoil one key at a time."""
    audit: dict[str, object] = {
        "energy_balance_residual_mwh": 0.0,
        "max_interval_energy_balance_error_mwh": 0.0,
        "max_simultaneous_dispatch_fraction": 0.0,
        "solver_all_optimal": True,
        "max_solver_mip_gap": 0.0,
        "completed_cycle_break_even_violation_count": 0,
    }
    audit.update(overrides)
    return audit


def test_price_and_dispatch_audits() -> None:
    timestamps = pd.date_range("2023-01-01 00:05", periods=4, freq="5min", tz=AEST)
    prices = pd.DataFrame(
        {
            "settlementdate": timestamps,
            "regionid": "SA1",
            "rrp": [0.0, 100.0, 0.0, 100.0],
        }
    )
    trace = run_rolling_horizon(
        prices, asset(), soc_initial=0, window_hours=1, step_hours=1
    )
    audit = {
        **audit_prices(
            prices,
            expected_start="2023-01-01 00:00",
            expected_end="2023-01-01 00:20",
        ),
        **audit_dispatch(trace, asset(), soc_initial=0),
        **audit_completed_cycles(
            settle(trace, prices, asset()).intervals,
            asset(),
            deg_cost=0.0,
            initial_soc_mwh=0.0,
        ),
    }
    assert audit["price_missing_interval_count"] == 0
    assert audit["solver_window_count"] == 1
    assert audit["solver_all_optimal"] is True
    assert_audit_passes(audit, require_price_audit=True)


def test_audit_rejects_non_optimal_status() -> None:
    with pytest.raises(ValueError, match="non-optimal"):
        assert_audit_passes(passing_audit(solver_all_optimal=False))


def test_a_missing_gate_is_a_failure_not_a_pass() -> None:
    """An absent key means the check never ran, so it must not be treated as green."""
    for dropped in (
        "energy_balance_residual_mwh",
        "max_simultaneous_dispatch_fraction",
        "completed_cycle_break_even_violation_count",
    ):
        audit = passing_audit()
        del audit[dropped]
        with pytest.raises(ValueError, match="incomplete"):
            assert_audit_passes(audit)
    # Price gates are only demanded when the caller merged the price audit.
    assert_audit_passes(passing_audit())
    with pytest.raises(ValueError, match="incomplete"):
        assert_audit_passes(passing_audit(), require_price_audit=True)


def test_completed_cycle_audit_rejects_loss_making_cycle() -> None:
    battery = BatteryConfig(
        generation_loss_factor=0.7,
        load_loss_factor=1.2,
        power_mw=1,
        energy_mwh=1,
        round_trip_efficiency=1,
        soc_min_fraction=0,
        soc_max_fraction=1,
        interval_minutes=60,
    )
    timestamps = pd.date_range("2023-01-01 01:00", periods=2, freq="h", tz=AEST)
    prices = pd.DataFrame({"settlementdate": timestamps, "rrp": [10.0, 12.0]})
    trace = pd.DataFrame(
        {
            "settlementdate": timestamps,
            "charge_mw": [1.0, 0.0],
            "discharge_mw": [0.0, 1.0],
            "soc_mwh": [1.0, 0.0],
        }
    )
    revenue = settle(trace, prices, battery)
    audit = audit_completed_cycles(
        revenue.intervals, battery, deg_cost=0, initial_soc_mwh=0
    )
    assert audit["completed_cycle_break_even_violation_count"] == 1
    with pytest.raises(ValueError, match="break-even"):
        assert_audit_passes(passing_audit(**audit))
