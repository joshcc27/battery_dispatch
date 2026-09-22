from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_dispatch.battery import energy_balance_residual
from battery_dispatch.config import AEST, BatteryConfig
from battery_dispatch.optimiser import solve_window
from battery_dispatch.settlement import settle


def make_asset(**changes: float) -> BatteryConfig:
    values = {
        "generation_loss_factor": 1.0,
        "load_loss_factor": 1.0,
        "power_mw": 1.0,
        "energy_mwh": 1.0,
        "round_trip_efficiency": 0.81,
        "soc_min_fraction": 0.0,
        "soc_max_fraction": 1.0,
        "interval_minutes": 60,
    }
    values.update(changes)
    return BatteryConfig(**values)


def objective_revenue(window, prices, asset, deg_cost=0.0) -> float:
    timestamps = pd.date_range(
        "2023-01-01 01:00",
        periods=len(prices),
        freq=f"{asset.interval_minutes}min",
        tz=AEST,
    )
    price_frame = pd.DataFrame({"settlementdate": timestamps, "rrp": prices})
    return settle(
        window.to_frame(timestamps), price_frame, asset, deg_cost=deg_cost
    ).total_revenue


def test_known_six_interval_solution() -> None:
    battery = make_asset(round_trip_efficiency=1.0)
    prices = np.array([10, 20, 100, 30, 20, 10], dtype=float)
    result = solve_window(prices, battery, soc_initial=0.0)
    assert result.charge_mw[0] == pytest.approx(1)
    assert result.discharge_mw[2] == pytest.approx(1)
    assert result.objective_value == pytest.approx(90)


def test_energy_balance_soc_bounds_and_negative_price_binary() -> None:
    battery = make_asset()
    prices = np.array([-100, -100, -100, -100, 150, 200], dtype=float)
    result = solve_window(prices, battery, soc_initial=0.0)
    assert np.all(result.charge_mw * result.discharge_mw <= 1e-8)
    assert np.all(result.soc_mwh >= battery.soc_min_mwh - 1e-7)
    assert np.all(result.soc_mwh <= battery.soc_max_mwh + 1e-7)
    assert abs(
        energy_balance_residual(
            result.charge_mw,
            result.discharge_mw,
            result.soc_initial_mwh,
            result.soc_final_mwh,
            battery,
        )
    ) < 1e-6


def test_flat_price_does_not_cycle_from_minimum_soc() -> None:
    battery = make_asset()
    result = solve_window(np.full(6, 50.0), battery, soc_initial=battery.initial_soc_mwh)
    assert np.allclose(result.charge_mw, 0)
    assert np.allclose(result.discharge_mw, 0)
    assert result.objective_value == pytest.approx(0)


def test_revenue_monotonic_in_energy_and_power() -> None:
    prices = np.array([0, 0, 100, 100], dtype=float)
    small_energy = make_asset(energy_mwh=0.5, round_trip_efficiency=1.0)
    large_energy = make_asset(energy_mwh=1.0, round_trip_efficiency=1.0)
    small_power = make_asset(power_mw=0.5, round_trip_efficiency=1.0)
    large_power = make_asset(power_mw=1.0, round_trip_efficiency=1.0)
    e1_window = solve_window(prices, small_energy, 0)
    e2_window = solve_window(prices, large_energy, 0)
    p1_window = solve_window(prices, small_power, 0)
    p2_window = solve_window(prices, large_power, 0)
    e1 = objective_revenue(e1_window, prices, small_energy)
    e2 = objective_revenue(e2_window, prices, large_energy)
    p1 = objective_revenue(p1_window, prices, small_power)
    p2 = objective_revenue(p2_window, prices, large_power)
    assert e2 >= e1
    assert p2 >= p1


def test_realised_revenue_is_non_increasing_in_degradation_cost() -> None:
    battery = make_asset(
        round_trip_efficiency=1.0,
        generation_loss_factor=0.7,
        load_loss_factor=1.2,
    )
    prices = np.array([10, 12], dtype=float)
    low = solve_window(prices, battery, 0, deg_cost=0)
    high = solve_window(prices, battery, 0, deg_cost=5)
    assert objective_revenue(high, prices, battery, 5) <= objective_revenue(
        low, prices, battery, 0
    )
    assert high.discharge_mw.sum() <= low.discharge_mw.sum()


def test_soc_initial_clips_solver_scale_roundoff_only() -> None:
    battery = BatteryConfig(
        generation_loss_factor=1,
        load_loss_factor=1,
        power_mw=1,
        energy_mwh=1,
        round_trip_efficiency=1,
        soc_min_fraction=0,
        soc_max_fraction=1,
        interval_minutes=60,
    )
    clipped = solve_window(np.array([0.0]), battery, 1.0 + 1e-12)
    assert clipped.soc_initial_mwh == 1.0
    with pytest.raises(ValueError, match="outside"):
        solve_window(np.array([0.0]), battery, 1.0 + 1e-4)


def test_core_policy_optimises_settled_loss_factor_cashflows() -> None:
    prices = np.array([10, 12], dtype=float)
    unit = make_asset(
        generation_loss_factor=1.0,
        load_loss_factor=1.0,
        round_trip_efficiency=1.0,
    )
    asymmetric = make_asset(
        generation_loss_factor=0.7,
        load_loss_factor=1.2,
        round_trip_efficiency=1.0,
    )
    first = solve_window(prices, unit, 0)
    second = solve_window(prices, asymmetric, 0)
    assert first.discharge_mw.sum() > 0
    np.testing.assert_allclose(second.charge_mw, 0, atol=1e-8)
    np.testing.assert_allclose(second.discharge_mw, 0, atol=1e-8)
    assert objective_revenue(second, prices, asymmetric) == pytest.approx(0)


def test_completed_cycle_respects_loss_factor_break_even() -> None:
    battery = make_asset(
        round_trip_efficiency=1.0,
        generation_loss_factor=0.7,
        load_loss_factor=1.2,
    )
    unprofitable = solve_window(np.array([10.0, 17.0]), battery, 0)
    profitable = solve_window(np.array([10.0, 18.0]), battery, 0)
    np.testing.assert_allclose(unprofitable.charge_mw, 0, atol=1e-8)
    np.testing.assert_allclose(unprofitable.discharge_mw, 0, atol=1e-8)
    assert profitable.charge_mw[0] == pytest.approx(1)
    assert profitable.discharge_mw[1] == pytest.approx(1)
