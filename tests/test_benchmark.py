from __future__ import annotations

import math

from battery_dispatch.benchmark import compare_to_benchmark, load_benchmarks


def test_benchmark_loader_and_scope_warning() -> None:
    benchmarks = load_benchmarks("config/benchmarks.csv")
    row = benchmarks.loc[benchmarks["benchmark_id"] == "AEMO_NEM_Q4_2024"].iloc[0]
    comparison = compare_to_benchmark(
        {
            "energy_revenue_per_mw_year": 100_000.0,
            "net_revenue_per_mw_year": 90_000.0,
            "region": "SA1",
        },
        row,
        model_period_start="2024-10-01",
        model_period_end="2025-01-01",
    )
    assert comparison["period_aligned"] is True
    assert comparison["comparison_is_like_for_like"] is False
    assert comparison["model_less_than_benchmark_total"] is True
    assert math.isfinite(comparison["benchmark_energy_revenue_per_mw_year"])
    assert comparison["capacity_basis_name"] == "average availability MW"


def test_missing_capacity_does_not_invent_per_mw_value() -> None:
    benchmarks = load_benchmarks("config/benchmarks.csv")
    row = benchmarks.loc[benchmarks["benchmark_id"] == "AEMO_SA_Q1_2024"].iloc[0]
    comparison = compare_to_benchmark(
        {"energy_revenue_per_mw_year": 100_000.0, "region": "SA1"},
        row,
        model_period_start="2024-01-01",
        model_period_end="2024-04-01",
    )
    assert math.isnan(comparison["benchmark_energy_revenue_per_mw_year"])
    assert comparison["model_less_than_benchmark_total"] is None
