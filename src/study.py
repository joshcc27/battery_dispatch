"""Parameter-sweep orchestration over regions, financial years, and degradation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .config import BatteryConfig, LossFactorTable, financial_year
from .horizon import run_rolling_horizon
from .metrics import summarise_run, write_run_outputs
from .settlement import settle


def run_parameter_sweep(
    prices: pd.DataFrame,
    base_asset: BatteryConfig,
    loss_factors: Mapping[str, LossFactorTable],
    degradation_costs: Iterable[float],
    *,
    output_dir: str | Path | None = None,
    regions: Iterable[str] | None = None,
    window_hours: int = 48,
    step_hours: int = 24,
) -> pd.DataFrame:
    """Run every requested region/FY/degradation combination.

    ``prices`` must contain ``settlementdate``, ``regionid``, and ``rrp``. Loss
    factors are asset-specific schedules keyed first by region and then by FY.
    Each annual run starts at the configured minimum SOC, making annual and
    regional comparisons independent and reproducible.
    """
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"settlementdate", "regionid", "rrp"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"prices are missing columns: {sorted(missing)}")
    frame["financial_year"] = frame["settlementdate"].map(financial_year)
    region_list = list(regions) if regions is not None else sorted(frame["regionid"].unique())
    costs = [float(value) for value in degradation_costs]
    if not costs or any(value < 0 for value in costs):
        raise ValueError("degradation_costs must contain non-negative values")

    summaries: list[dict[str, object]] = []
    for region in region_list:
        if region not in loss_factors:
            raise KeyError(f"No loss-factor table configured for region {region}")
        regional = frame.loc[frame["regionid"] == region]
        if regional.empty:
            raise ValueError(f"No prices supplied for region {region}")
        for fy, annual in regional.groupby("financial_year", sort=True):
            factors = loss_factors[region].for_financial_year(str(fy))
            if factors.source is None:
                raise ValueError(
                    f"Loss factors for {region} {fy} require published source provenance"
                )
            asset = replace(base_asset, region=region).with_loss_factors(factors)
            price_columns = ["settlementdate", "rrp"]
            if "june_2022_event" in annual:
                price_columns.append("june_2022_event")
            annual_prices = annual[price_columns].sort_values("settlementdate")
            for deg_cost in costs:
                trace = run_rolling_horizon(
                    annual_prices,
                    asset,
                    window_hours=window_hours,
                    step_hours=step_hours,
                    deg_cost=deg_cost,
                )
                revenue = settle(trace, annual_prices, asset, deg_cost=deg_cost)
                run_hash = asset.config_hash(
                    extra={"financial_year": fy, "degradation_cost": deg_cost}
                )
                summary = summarise_run(
                    revenue, asset, config_hash=run_hash, region=region
                )
                summary["degradation_cost_per_mwh"] = deg_cost
                summaries.append(summary)
                if output_dir is not None:
                    write_run_outputs(revenue, summary, output_dir)
    return pd.DataFrame(summaries)
