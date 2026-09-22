from __future__ import annotations

import pandas as pd

from battery_dispatch.config import AEST, BatteryConfig
from battery_dispatch.horizon import run_rolling_horizon
from battery_dispatch.metrics import summarise_run, write_run_outputs
from battery_dispatch.settlement import settle


def test_rolling_horizon_settlement_metrics_and_outputs(tmp_path) -> None:
    battery = BatteryConfig(
        generation_loss_factor=1.0,
        load_loss_factor=1.0,
        power_mw=1,
        energy_mwh=2,
        round_trip_efficiency=0.81,
        soc_min_fraction=0,
        soc_max_fraction=1,
        interval_minutes=60,
    )
    timestamps = pd.date_range("2022-07-01 01:00", periods=30, freq="h", tz=AEST)
    prices = pd.DataFrame(
        {
            "settlementdate": timestamps,
            "rrp": ([0.0] * 6 + [100.0] * 6 + [10.0] * 6 + [200.0] * 6 + [20.0] * 6),
        }
    )
    trace = run_rolling_horizon(
        prices, battery, soc_initial=0, window_hours=12, step_hours=6
    )
    assert len(trace) == len(prices)
    assert trace["settlementdate"].is_unique
    revenue = settle(trace, prices, battery)
    run_hash = battery.config_hash(extra={"case": "test"})
    summary = summarise_run(revenue, battery, config_hash=run_hash)
    assert summary["total_revenue"] >= 0
    assert summary["equivalent_full_cycles"] >= 0
    assert summary["revenue_per_mw_year"] == (
        summary["revenue_per_mw_period"] * summary["annualisation_factor"]
    )
    interval_path, result_path = write_run_outputs(revenue, summary, tmp_path)
    assert interval_path.exists()
    assert result_path.exists()


