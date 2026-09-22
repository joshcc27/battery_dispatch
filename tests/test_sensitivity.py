from __future__ import annotations

import pandas as pd

from battery_dispatch.config import AEST, BatteryConfig
from battery_dispatch.sensitivity import EndpointCase, run_sensitivities


def test_endpoint_and_degradation_sensitivity() -> None:
    asset = BatteryConfig(
        generation_loss_factor=1,
        load_loss_factor=1,
        power_mw=1,
        energy_mwh=1,
        round_trip_efficiency=1,
        soc_min_fraction=0,
        soc_max_fraction=1,
        interval_minutes=60,
    )
    prices = pd.DataFrame(
        {
            "settlementdate": pd.date_range(
                "2023-01-01 01:00", periods=4, freq="h", tz=AEST
            ),
            "rrp": [0.0, 100.0, 0.0, 100.0],
        }
    )
    result = run_sensitivities(
        prices,
        asset,
        endpoint_cases=[
            EndpointCase("minimum", 0.0, 0.0),
            EndpointCase("valued_terminal", 0.0, 150.0),
        ],
        degradation_costs=[0.0, 95.0],
        window_hours=4,
        step_hours=4,
    )
    assert len(result) == 4
    minimum = result.loc[result["endpoint_case"] == "minimum"].sort_values(
        "degradation_cost_per_mwh"
    )
    assert minimum["total_revenue"].is_monotonic_decreasing
    assert minimum["equivalent_full_cycles"].is_monotonic_decreasing
    valued = result.loc[
        (result["endpoint_case"] == "valued_terminal")
        & (result["degradation_cost_per_mwh"] == 0)
    ].iloc[0]
    assert valued["terminal_soc_mwh"] == 1.0
    assert valued["inventory_value_change"] == 150.0
