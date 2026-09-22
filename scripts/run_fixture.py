"""Run a deterministic end-to-end fixture and emit its audited summary."""

from __future__ import annotations

import json

import pandas as pd

from battery_dispatch.config import AEST, BatteryConfig
from battery_dispatch.horizon import run_rolling_horizon
from battery_dispatch.metrics import summarise_run
from battery_dispatch.settlement import settle
from battery_dispatch.validation import (
    assert_audit_passes,
    audit_completed_cycles,
    audit_dispatch,
    audit_prices,
)


def main() -> None:
    start = pd.Timestamp("2023-01-01 00:00", tz=AEST)
    end = pd.Timestamp("2023-01-02 00:00", tz=AEST)
    timestamps = pd.date_range(start + pd.Timedelta(minutes=5), end, freq="5min")
    daily = [0.0] * 96 + [100.0] * 96 + [20.0] * 96
    prices = pd.DataFrame(
        {"settlementdate": timestamps, "regionid": "SA1", "rrp": daily}
    )
    asset = BatteryConfig(
        generation_loss_factor=0.9652,
        load_loss_factor=0.9681,
        loss_factor_source="deterministic CI fixture",
        connection_point="fixture",
    )
    trace = run_rolling_horizon(prices, asset)
    revenue = settle(trace, prices, asset)
    audit = {
        **audit_prices(prices, expected_start=start, expected_end=end),
        **audit_dispatch(trace, asset, soc_initial=asset.initial_soc_mwh),
        **audit_completed_cycles(
            revenue.intervals, asset, deg_cost=0, initial_soc_mwh=asset.initial_soc_mwh
        ),
    }
    assert_audit_passes(audit)
    summary = summarise_run(revenue, asset, config_hash="ci-fixture")
    print(json.dumps({**summary, **audit}, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
