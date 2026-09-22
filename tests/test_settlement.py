from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from battery_dispatch.battery import energy_balance_residual, soc_path
from battery_dispatch.config import (
    AEST,
    BatteryConfig,
    LossFactors,
    LossFactorTable,
    financial_year,
)
from battery_dispatch.settlement import break_even_discharge_price, settle, settlement_coefficients


def asset(**changes: float) -> BatteryConfig:
    values = {
        "generation_loss_factor": 0.98,
        "load_loss_factor": 1.02,
        "power_mw": 1.0,
        "energy_mwh": 1.0,
        "round_trip_efficiency": 1.0,
        "soc_min_fraction": 0.0,
        "soc_max_fraction": 1.0,
        "interval_minutes": 60,
    }
    values.update(changes)
    return BatteryConfig(**values)


def test_hand_computed_six_interval_settlement() -> None:
    timestamps = pd.date_range("2023-01-01 01:00", periods=6, freq="h", tz=AEST)
    trace = pd.DataFrame(
        {
            "settlementdate": timestamps,
            "charge_mw": [1, 0, 0, 0, 0, 0],
            "discharge_mw": [0, 0, 1, 0, 0, 0],
            "soc_mwh": [1, 1, 0, 0, 0, 0],
        }
    )
    prices = pd.DataFrame(
        {"settlementdate": timestamps, "rrp": [10, 20, 100, 30, 20, 10]}
    )
    result = settle(trace, prices, asset(), deg_cost=2)
    assert result.energy_revenue == pytest.approx(0.98 * 100 - 1.02 * 10)
    assert result.degradation_cost == pytest.approx(2)
    assert result.total_revenue == pytest.approx(85.8)


def test_efficiency_is_in_soc_not_applied_again_at_settlement() -> None:
    battery = asset(round_trip_efficiency=0.81)
    charge = np.array([1.0, 0.0])
    discharge = np.array([0.0, 0.81])
    soc = soc_path(charge, discharge, 0.0, battery)
    assert soc[-1] == pytest.approx(0.0)
    assert energy_balance_residual(charge, discharge, 0.0, soc[-1], battery) == pytest.approx(0)


def test_break_even_includes_efficiency_loss_factors_and_degradation() -> None:
    battery = asset(round_trip_efficiency=0.81)
    price = break_even_discharge_price(50.0, battery, deg_cost=4.0)
    expected = 1.02 * 50 / (0.98 * 0.81) + 4 / 0.98
    assert price == pytest.approx(expected)


def test_settlement_selects_loss_factors_by_financial_year() -> None:
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp("2023-07-01 00:00", tz=AEST), pd.Timestamp("2023-07-01 00:05", tz=AEST)]
    )
    battery = asset()
    battery = replace(
        battery,
        loss_factor_table=LossFactorTable(
            {
                "2022-23": LossFactors(0.9, 1.0),
                "2023-24": LossFactors(1.1, 1.0),
            }
        ),
    )
    trace = pd.DataFrame(
        {
            "settlementdate": timestamps,
            "charge_mw": [0, 0],
            "discharge_mw": [1, 1],
            "soc_mwh": [1, 0],
        }
    )
    prices = pd.DataFrame({"settlementdate": timestamps, "rrp": [100, 100]})
    result = settle(trace, prices, battery)
    assert result.total_revenue == pytest.approx(200)
    assert result.intervals["generation_loss_factor"].tolist() == [0.9, 1.1]


def test_financial_year_respects_interval_ending_midnight() -> None:
    assert financial_year(pd.Timestamp("2023-07-01 00:00", tz=AEST)) == "2022-23"
    assert financial_year(pd.Timestamp("2023-07-01 00:05", tz=AEST)) == "2023-24"


def test_optimizer_and_settlement_share_dated_coefficients() -> None:
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp("2023-07-01 00:00", tz=AEST), pd.Timestamp("2023-07-01 00:05", tz=AEST)]
    )
    battery = replace(
        asset(interval_minutes=5),
        loss_factor_table=LossFactorTable(
            {
                "2022-23": LossFactors(0.9, 1.0, "published FY22-23"),
                "2023-24": LossFactors(1.1, 1.0, "published FY23-24"),
            }
        ),
    )
    coefficients = settlement_coefficients(
        np.array([100.0, 100.0]), battery, settlementdate=timestamps
    )
    assert coefficients.generation_loss_factor.tolist() == [0.9, 1.1]


def test_cross_financial_year_requires_dated_loss_factors() -> None:
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp("2023-07-01 00:00", tz=AEST), pd.Timestamp("2023-07-01 00:05", tz=AEST)]
    )
    with pytest.raises(ValueError, match="dated loss-factor table"):
        settlement_coefficients(
            np.array([100.0, 100.0]), asset(interval_minutes=5), settlementdate=timestamps
        )
