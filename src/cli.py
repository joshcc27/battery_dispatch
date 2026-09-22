"""Small command-line interface for fetching and running one financial year."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import BatteryConfig
from .data import MarketDataCache
from .horizon import run_rolling_horizon
from .metrics import summarise_run, write_run_outputs
from .mlf import load_loss_factor_schedule
from .settlement import settle
from .validation import (
    assert_audit_passes,
    audit_completed_cycles,
    audit_dispatch,
    audit_prices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive AEST physical period boundary")
    parser.add_argument("--end", required=True, help="Exclusive AEST physical period boundary")
    parser.add_argument("--region", default="SA1")
    parser.add_argument("--generation-mlf", type=float)
    parser.add_argument("--load-mlf", type=float)
    parser.add_argument(
        "--mlf-source",
        help="Published source/provenance for the asset and financial-year MLFs",
    )
    parser.add_argument("--loss-factor-file", type=Path)
    parser.add_argument("--connection-point")
    parser.add_argument("--degradation-cost", type=float, default=0.0)
    parser.add_argument("--terminal-value-per-mwh", type=float, default=0.0)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser


def asset_from_args(args: argparse.Namespace) -> BatteryConfig:
    if args.loss_factor_file is not None:
        if any(
            value is not None
            for value in (args.generation_mlf, args.load_mlf, args.mlf_source)
        ):
            raise ValueError(
                "--loss-factor-file cannot be combined with scalar MLF arguments"
            )
        schedule = load_loss_factor_schedule(
            args.loss_factor_file,
            region=args.region,
            connection_point=args.connection_point,
        )
        first = next(iter(schedule.table.by_financial_year.values()))
        return BatteryConfig(
            generation_loss_factor=first.generation,
            load_loss_factor=first.load,
            loss_factor_source=schedule.provenance,
            loss_factor_table=schedule.table,
            connection_point=schedule.connection_point,
            region=args.region,
        )
    required = {
        "--generation-mlf": args.generation_mlf,
        "--load-mlf": args.load_mlf,
        "--mlf-source": args.mlf_source,
        "--connection-point": args.connection_point,
    }
    if missing := [name for name, value in required.items() if value is None]:
        raise ValueError(
            "Scalar MLF mode requires " + ", ".join(missing)
        )
    return BatteryConfig(
        generation_loss_factor=args.generation_mlf,
        load_loss_factor=args.load_mlf,
        loss_factor_source=args.mlf_source,
        connection_point=args.connection_point,
        region=args.region,
    )


def main() -> None:
    args = build_parser().parse_args()
    asset = asset_from_args(args)
    prices = MarketDataCache(args.cache).fetch_prices(args.start, args.end, [args.region])
    trace = run_rolling_horizon(
        prices,
        asset,
        deg_cost=args.degradation_cost,
        terminal_value_per_mwh=args.terminal_value_per_mwh,
    )
    revenue = settle(trace, prices, asset, deg_cost=args.degradation_cost)
    run_hash = asset.config_hash(
        extra={
            "start": args.start,
            "end": args.end,
            "deg_cost": args.degradation_cost,
            "terminal_value_per_mwh": args.terminal_value_per_mwh,
        }
    )
    summary = summarise_run(
        revenue,
        asset,
        config_hash=run_hash,
        terminal_value_per_mwh=args.terminal_value_per_mwh,
    )
    audit = {
        **audit_prices(prices, expected_start=args.start, expected_end=args.end),
        **audit_dispatch(trace, asset, soc_initial=asset.initial_soc_mwh),
        **audit_completed_cycles(
            revenue.intervals,
            asset,
            deg_cost=args.degradation_cost,
            initial_soc_mwh=asset.initial_soc_mwh,
        ),
    }
    assert_audit_passes(audit, require_price_audit=True)
    summary.update(audit)
    interval_path, summary_path = write_run_outputs(revenue, summary, args.output)
    print(f"Wrote {interval_path} and {summary_path}")


if __name__ == "__main__":
    main()
