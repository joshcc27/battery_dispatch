"""Run-level metrics and durable result outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BatteryConfig, financial_year
from .settlement import RevenueResult


def summarise_run(
    result: RevenueResult,
    asset: BatteryConfig,
    *,
    config_hash: str,
    region: str | None = None,
    initial_soc_mwh: float | None = None,
    terminal_value_per_mwh: float = 0.0,
) -> dict[str, object]:
    """Calculate industry-standard revenue, cycling, and concentration metrics."""
    data = result.intervals
    if data.empty:
        raise ValueError("Cannot summarise an empty result")
    charge_mwh = float(data["charge_mwh"].sum())
    discharge_mwh = float(data["discharge_mwh"].sum())
    charge_price = (
        float(np.average(data["rrp"], weights=data["charge_mwh"]))
        if charge_mwh > 0
        else np.nan
    )
    discharge_price = (
        float(np.average(data["rrp"], weights=data["discharge_mwh"]))
        if discharge_mwh > 0
        else np.nan
    )
    hours = len(data) * asset.interval_hours
    annualisation_factor = (365.25 * 24.0) / hours
    total = result.total_revenue
    sorted_revenue = data["total_revenue"].sort_values(ascending=False)

    def concentration(n: int) -> float:
        return float(sorted_revenue.head(n).sum() / total) if total != 0 else np.nan

    top_point_one_n = max(1, int(np.ceil(len(data) * 0.001)))
    minimum = asset.soc_min_mwh
    width = asset.usable_energy_mwh
    decile = np.floor((data["soc_mwh"] - minimum) / width * 10).clip(0, 9).astype(int)
    decile_shares = decile.value_counts(normalize=True)
    initial_soc = asset.initial_soc_mwh if initial_soc_mwh is None else float(initial_soc_mwh)
    terminal_soc = float(data["soc_mwh"].iloc[-1])
    inventory_value_change = (terminal_soc - initial_soc) * terminal_value_per_mwh
    summary: dict[str, object] = {
        "config_hash": config_hash,
        "region": region or asset.region,
        "financial_year": financial_year(data["settlementdate"].iloc[0]),
        "total_revenue": total,
        "energy_revenue": result.energy_revenue,
        "degradation_cost": result.degradation_cost,
        "power_mw": asset.power_mw,
        "energy_mwh": asset.energy_mwh,
        "period_hours": hours,
        "annualisation_factor": annualisation_factor,
        "revenue_per_mw_period": total / asset.power_mw,
        "revenue_per_mw_year": total / asset.power_mw * annualisation_factor,
        "energy_revenue_per_mw_year": (
            result.energy_revenue / asset.power_mw * annualisation_factor
        ),
        "net_revenue_per_mw_year": total / asset.power_mw * annualisation_factor,
        "equivalent_full_cycles": discharge_mwh / asset.energy_mwh,
        "equivalent_full_cycles_per_year": (
            discharge_mwh / asset.energy_mwh * annualisation_factor
        ),
        "initial_soc_mwh": initial_soc,
        "terminal_soc_mwh": terminal_soc,
        "net_inventory_change_mwh": terminal_soc - initial_soc,
        "terminal_value_per_mwh": terminal_value_per_mwh,
        "inventory_value_change": inventory_value_change,
        "cash_plus_inventory_value_change": total + inventory_value_change,
        "volume_weighted_charge_price": charge_price,
        "volume_weighted_discharge_price": discharge_price,
        "realised_spread": discharge_price - charge_price,
        "top_10_revenue_share": concentration(10),
        "top_50_revenue_share": concentration(50),
        "top_0_1pct_revenue_share": concentration(top_point_one_n),
        "negative_price_charge_share": (
            float(data.loc[data["rrp"] < 0, "charge_mwh"].sum() / charge_mwh)
            if charge_mwh > 0
            else np.nan
        ),
        "capacity_factor": discharge_mwh / (asset.power_mw * hours),
        "loss_factor_source": asset.loss_factor_source,
    }
    for bucket in range(10):
        summary[f"soc_decile_{bucket}_share"] = float(decile_shares.get(bucket, 0.0))
    return summary


def write_run_outputs(
    result: RevenueResult,
    summary: Mapping[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write one interval CSV and upsert one config-keyed summary row."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = str(summary["config_hash"])
    interval_path = directory / f"dispatch_{run_id}.csv"
    summary_path = directory / "results.csv"
    result.intervals.to_csv(interval_path, index=False)
    row = pd.DataFrame([dict(summary)])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing.loc[existing["config_hash"].astype(str) != run_id]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(summary_path, index=False)
    return interval_path, summary_path
