"""Small command-line interface for fetching and running one financial year."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import BatteryConfig
from .data import MarketDataCache
from .horizon import run_rolling_horizon
from .metrics import summarise_run, write_run_outputs
from .settlement import settle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive AEST physical period boundary")
    parser.add_argument("--end", required=True, help="Exclusive AEST physical period boundary")
    parser.add_argument("--region", default="SA1")
    parser.add_argument("--generation-mlf", type=float, required=True)
    parser.add_argument("--load-mlf", type=float, required=True)
    parser.add_argument(
        "--mlf-source",
        required=True,
        help="Published source/provenance for the asset and financial-year MLFs",
    )
    parser.add_argument("--degradation-cost", type=float, default=0.0)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asset = BatteryConfig(
        generation_loss_factor=args.generation_mlf,
        load_loss_factor=args.load_mlf,
        loss_factor_source=args.mlf_source,
        region=args.region,
    )
    prices = MarketDataCache(args.cache).fetch_prices(args.start, args.end, [args.region])
    trace = run_rolling_horizon(prices, asset, deg_cost=args.degradation_cost)
    revenue = settle(trace, prices, asset, deg_cost=args.degradation_cost)
    run_hash = asset.config_hash(
        extra={"start": args.start, "end": args.end, "deg_cost": args.degradation_cost}
    )
    summary = summarise_run(revenue, asset, config_hash=run_hash)
    interval_path, summary_path = write_run_outputs(revenue, summary, args.output)
    print(f"Wrote {interval_path} and {summary_path}")


if __name__ == "__main__":
    main()
