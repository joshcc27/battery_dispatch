"""Report where the charge/discharge exclusivity binary actually binds.

The rolling-horizon MILP carries one binary per interval to forbid simultaneous
charging and discharging. That binary is only needed where simultaneous
operation would otherwise be profitable, and this script measures how often
that is true at each reported degradation cost. It is the evidence behind the
A$0/MWh scenario's documented solver tolerance in docs/VALIDATION.md.

Derivation. Take any solution with both charge c and discharge d positive in
one interval, and shift it by -a on charge and -a*eta_c*eta_d on discharge.
That leaves the stored-energy change, and therefore the whole SOC path,
exactly unchanged. The objective moves by -a * (A_c + A_d * eta_c * eta_d),
where A_c = -dt * L * p and A_d = dt * (G * p - deg) are the settled per-MW
coefficients. Removing simultaneous operation is therefore never costly unless

    A_c + A_d * eta_c * eta_d > 0
    <=> p * (G * eta_c * eta_d - L) - deg * eta_c * eta_d > 0

so with G * eta_c * eta_d < L the binary can only bind when

    p < -deg * eta_c * eta_d / (L - G * eta_c * eta_d)

The throughput tie-breaker in the objective is strictly positive on both legs,
so where the binary does not bind the optimum still carries no simultaneous
operation and the exclusivity audit still holds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_dispatch.config import BatteryConfig
from battery_dispatch.data import MarketDataCache
from battery_dispatch.mlf import load_loss_factor_schedule

REPORTED_DEGRADATION_COSTS = (0.0, 25.0, 50.0)


def binding_price_threshold(asset: BatteryConfig, deg_cost: float) -> float:
    """Return the price below which exclusivity can bind at ``deg_cost``."""
    round_trip = asset.charge_efficiency * asset.discharge_efficiency
    margin = asset.load_loss_factor - asset.generation_loss_factor * round_trip
    if margin <= 0:
        # Discharging into the network out-earns the charge leg even before
        # arbitrage, so simultaneous operation never pays at any price.
        return float("-inf")
    # Normalise the signed zero at deg_cost == 0 so it prints as A$0.00.
    return (-deg_cost * round_trip / margin) or 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--start", default="2022-07-01 00:00")
    parser.add_argument("--end", default="2023-07-01 00:00")
    parser.add_argument("--region", default="SA1")
    parser.add_argument("--financial-year", default="2022-23")
    parser.add_argument("--worst-days", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    schedule = load_loss_factor_schedule(
        "config/loss_factors.csv",
        region=args.region,
        connection_point="Hornsdale 275 kV",
    )
    factors = schedule.table.for_financial_year(args.financial_year)
    asset = BatteryConfig(
        generation_loss_factor=factors.generation,
        load_loss_factor=factors.load,
        loss_factor_source=schedule.provenance,
        loss_factor_table=schedule.table,
        connection_point=schedule.connection_point,
        region=args.region,
    )
    prices = MarketDataCache(args.cache).fetch_prices(
        args.start, args.end, [args.region]
    )
    price = prices["rrp"].astype(float).to_numpy()
    # Interval-ending timestamps: an interval belongs to the trading day of its
    # start, so attribute it to the day of (timestamp - one interval).
    day = (
        pd.DatetimeIndex(prices["settlementdate"])
        - pd.Timedelta(minutes=asset.interval_minutes)
    ).strftime("%Y-%m-%d")

    round_trip = asset.charge_efficiency * asset.discharge_efficiency
    print(f"Region {args.region}, financial year {args.financial_year}")
    print(
        f"Generation MLF {asset.generation_loss_factor}, "
        f"load MLF {asset.load_loss_factor}, round trip {round_trip:.4f}"
    )
    print(f"Intervals: {len(price)}\n")
    print("| Degradation cost | Binary binds when | Intervals | Share | Days | Worst day |")
    print("|---|---:|---:|---:|---:|---:|")
    for deg_cost in REPORTED_DEGRADATION_COSTS:
        threshold = binding_price_threshold(asset, deg_cost)
        binding = pd.Series(price < threshold)
        per_day = binding.groupby(day).sum()
        print(
            f"| A${deg_cost:,.0f}/MWh | p < A${threshold:,.2f} | {int(binding.sum()):,} "
            f"| {binding.mean():.2%} | {int((per_day > 0).sum())} / {per_day.size} "
            f"| {int(per_day.max())} of {int(per_day.size and 288)} |"
        )

    print(f"\nWorst {args.worst_days} trading days by negative-price intervals:")
    negative = pd.Series(price < 0).groupby(day).sum().sort_values(ascending=False)
    for trading_day, count in negative.head(args.worst_days).items():
        print(f"  {trading_day}  {int(count):3d} of 288")


if __name__ == "__main__":
    main()
