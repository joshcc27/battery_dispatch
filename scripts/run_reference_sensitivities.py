"""Run and persist the declared full-year endpoint/degradation grid."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from battery_dispatch.config import BatteryConfig
from battery_dispatch.data import MarketDataCache
from battery_dispatch.mlf import load_loss_factor_schedule
from battery_dispatch.optimiser import STRICT_SOLVER_POLICY, SolverPolicy
from battery_dispatch.sensitivity import EndpointCase, run_sensitivities

# At A$0/MWh the exclusivity binary binds at 23% of intervals instead of 0.55%,
# and a 1e-6 relative gap is not tractable: one 48-hour window alone runs past
# 900 seconds. The tractability knee on the hardest window sits between 5e-4
# (63s) and 7e-4 (2.9s), so this scenario is run at 1e-3 with a time-limit
# backstop and reports the optimality gap it actually achieved.
# See docs/VALIDATION.md.
DEGRADATION_0_POLICY = SolverPolicy(mip_rel_gap=1e-3, time_limit_seconds=120.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/reference_sensitivities.csv")
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Report every Nth window; time-limited windows always report",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run only the named scenario; repeat to select more than one",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.getLogger("linopy").setLevel(logging.WARNING)
    schedule = load_loss_factor_schedule(
        "config/loss_factors.csv",
        region="SA1",
        connection_point="Hornsdale 275 kV",
    )
    factors = schedule.table.for_financial_year("2022-23")
    asset = BatteryConfig(
        generation_loss_factor=factors.generation,
        load_loss_factor=factors.load,
        loss_factor_source=schedule.provenance,
        loss_factor_table=schedule.table,
        connection_point=schedule.connection_point,
        region="SA1",
    )
    prices = MarketDataCache(args.cache).fetch_prices(
        "2022-07-01 00:00", "2023-07-01 00:00", ["SA1"]
    )
    scenarios = [
        (
            "degradation_0",
            EndpointCase("minimum_no_salvage", asset.soc_min_mwh, 0.0),
            0.0,
            DEGRADATION_0_POLICY,
        ),
        (
            "headline",
            EndpointCase("minimum_no_salvage", asset.soc_min_mwh, 0.0),
            25.0,
            STRICT_SOLVER_POLICY,
        ),
        (
            "degradation_50",
            EndpointCase("minimum_no_salvage", asset.soc_min_mwh, 0.0),
            50.0,
            STRICT_SOLVER_POLICY,
        ),
        (
            "midpoint_initial_soc",
            EndpointCase("midpoint_no_salvage", asset.energy_mwh / 2.0, 0.0),
            25.0,
            STRICT_SOLVER_POLICY,
        ),
        (
            "lookahead_salvage_100",
            EndpointCase("minimum_with_100_aud_salvage", asset.soc_min_mwh, 100.0),
            25.0,
            STRICT_SOLVER_POLICY,
        ),
        (
            "lookahead_salvage_400",
            EndpointCase("minimum_with_400_aud_salvage", asset.soc_min_mwh, 400.0),
            25.0,
            STRICT_SOLVER_POLICY,
        ),
    ]
    known_scenarios = {scenario_id for scenario_id, _, _, _ in scenarios}
    if args.scenario:
        unknown = set(args.scenario).difference(known_scenarios)
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
        selected = set(args.scenario)
        scenarios = [row for row in scenarios if row[0] in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.output) if args.output.exists() else pd.DataFrame()
    completed = set(results.get("scenario_id", pd.Series(dtype=str)).astype(str))
    for scenario_id, endpoint, degradation_cost, policy in scenarios:
        if scenario_id in completed:
            print(f"Skipping completed {scenario_id}", flush=True)
            continue
        print(f"Running {scenario_id}", flush=True)
        started = time.perf_counter()

        def report(window_id: int, window_start, window, _started=started) -> None:
            if window_id % args.progress_every and not window.solver_time_limited:
                return
            flag = " TIME-LIMITED" if window.solver_time_limited else ""
            print(
                f"  window {window_id:>3} {window_start:%Y-%m-%d}"
                f" {window.solve_seconds:7.1f}s"
                f" gap={window.solver_mip_gap:.2e}"
                f" elapsed={time.perf_counter() - _started:7.1f}s{flag}",
                flush=True,
            )

        result = run_sensitivities(
            prices,
            asset,
            endpoint_cases=[endpoint],
            degradation_costs=[degradation_cost],
            policy=policy,
            progress=report,
        )
        result["wall_clock_seconds"] = time.perf_counter() - started
        result.insert(0, "scenario_id", scenario_id)
        results = pd.concat([results, result], ignore_index=True)
        results.to_csv(args.output, index=False)
        print(
            f"Completed {scenario_id} in "
            f"{result['wall_clock_seconds'].iloc[0]:.1f}s",
            flush=True,
        )
    print(results[
        [
            "scenario_id",
            "endpoint_case",
            "degradation_cost_per_mwh",
            "total_revenue",
            "energy_revenue",
            "equivalent_full_cycles",
            "terminal_soc_mwh",
            "solver_all_optimal",
            "max_solver_mip_gap",
            "total_solver_absolute_gap_aud",
            "wall_clock_seconds",
        ]
    ].to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
