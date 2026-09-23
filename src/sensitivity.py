"""Endpoint and degradation sensitivity runners."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd

from .ceiling import audit_ceiling, perfect_foresight_ceiling
from .config import BatteryConfig
from .horizon import run_rolling_horizon
from .metrics import summarise_run
from .optimiser import STRICT_SOLVER_POLICY, SolverPolicy
from .settlement import settle
from .validation import (
    assert_audit_passes,
    audit_completed_cycles,
    audit_dispatch,
)


@dataclass(frozen=True)
class EndpointCase:
    name: str
    initial_soc_mwh: float
    terminal_value_per_mwh: float


def run_sensitivities(
    prices: pd.DataFrame,
    asset: BatteryConfig,
    *,
    endpoint_cases: Iterable[EndpointCase],
    degradation_costs: Iterable[float],
    window_hours: int = 48,
    step_hours: int = 24,
    policy: SolverPolicy = STRICT_SOLVER_POLICY,
    progress: Callable[..., None] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for endpoint in endpoint_cases:
        if not asset.soc_min_mwh <= endpoint.initial_soc_mwh <= asset.soc_max_mwh:
            raise ValueError(f"Initial SOC for {endpoint.name} is outside the envelope")
        for degradation_cost in degradation_costs:
            trace = run_rolling_horizon(
                prices,
                asset,
                soc_initial=endpoint.initial_soc_mwh,
                window_hours=window_hours,
                step_hours=step_hours,
                deg_cost=float(degradation_cost),
                terminal_value_per_mwh=endpoint.terminal_value_per_mwh,
                policy=policy,
                progress=progress,
            )
            revenue = settle(
                trace, prices, asset, deg_cost=float(degradation_cost)
            )
            run_hash = asset.config_hash(
                extra={
                    "endpoint_case": endpoint.name,
                    "initial_soc_mwh": endpoint.initial_soc_mwh,
                    "terminal_value_per_mwh": endpoint.terminal_value_per_mwh,
                    "degradation_cost": float(degradation_cost),
                    "solver_policy": policy.describe(),
                }
            )
            summary = summarise_run(
                revenue,
                asset,
                config_hash=run_hash,
                initial_soc_mwh=endpoint.initial_soc_mwh,
                terminal_value_per_mwh=endpoint.terminal_value_per_mwh,
            )
            summary["endpoint_case"] = endpoint.name
            summary["degradation_cost_per_mwh"] = float(degradation_cost)
            summary["solver_policy"] = policy.label
            audit = {
                **audit_dispatch(trace, asset, soc_initial=endpoint.initial_soc_mwh),
                **audit_completed_cycles(
                    revenue.intervals,
                    asset,
                    deg_cost=float(degradation_cost),
                    initial_soc_mwh=endpoint.initial_soc_mwh,
                ),
                **audit_ceiling(
                    perfect_foresight_ceiling(
                        prices,
                        asset,
                        soc_initial=endpoint.initial_soc_mwh,
                        deg_cost=float(degradation_cost),
                        terminal_value_per_mwh=endpoint.terminal_value_per_mwh,
                    ),
                    revenue,
                    terminal_value_per_mwh=endpoint.terminal_value_per_mwh,
                ),
            }
            assert_audit_passes(audit, mip_gap_tolerance=policy.gap_tolerance)
            summary.update(audit)
            rows.append(summary)
    return pd.DataFrame(rows)
