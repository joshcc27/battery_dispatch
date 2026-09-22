"""Scope-aware comparison with published battery revenue benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "benchmark_id",
    "period_start",
    "period_end",
    "region",
    "fleet_scope",
    "energy_revenue_aud_million",
    "fcas_revenue_aud_million",
    "total_revenue_aud_million",
    "capacity_basis_mw",
    "capacity_basis_name",
    "source_title",
    "source_url",
    "source_page",
}


def load_benchmarks(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).lower() for column in frame.columns]
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"Benchmark file is missing columns: {sorted(missing)}")
    if frame["benchmark_id"].duplicated().any():
        raise ValueError("benchmark_id values must be unique")
    if not frame["source_url"].str.startswith("https://").all():
        raise ValueError("Benchmark source URLs must use HTTPS")
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    if (frame["period_end"] <= frame["period_start"]).any():
        raise ValueError("Benchmark periods must have positive duration")
    for column in (
        "energy_revenue_aud_million",
        "fcas_revenue_aud_million",
        "total_revenue_aud_million",
        "capacity_basis_mw",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def compare_to_benchmark(
    model_summary: Mapping[str, object],
    benchmark: pd.Series,
    *,
    model_period_start: object,
    model_period_end: object,
) -> dict[str, object]:
    start = pd.Timestamp(model_period_start).tz_localize(None)
    end = pd.Timestamp(model_period_end).tz_localize(None)
    benchmark_start = pd.Timestamp(benchmark["period_start"]).tz_localize(None)
    benchmark_end = pd.Timestamp(benchmark["period_end"]).tz_localize(None)
    aligned = start == benchmark_start and end == benchmark_end
    period_days = (benchmark_end - benchmark_start).total_seconds() / 86_400
    annualisation = 365.25 / period_days
    capacity = float(benchmark["capacity_basis_mw"])
    energy_per_mw_year = (
        float(benchmark["energy_revenue_aud_million"])
        * 1_000_000
        / capacity
        * annualisation
        if np.isfinite(capacity) and capacity > 0
        else np.nan
    )
    total_per_mw_year = (
        float(benchmark["total_revenue_aud_million"])
        * 1_000_000
        / capacity
        * annualisation
        if np.isfinite(capacity) and capacity > 0
        else np.nan
    )
    model_energy_value = float(model_summary["energy_revenue_per_mw_year"])
    model_net_value = float(
        model_summary.get("net_revenue_per_mw_year", model_energy_value)
    )
    return {
        "benchmark_id": benchmark["benchmark_id"],
        "period_aligned": aligned,
        "model_energy_revenue_per_mw_year": model_energy_value,
        "model_net_revenue_per_mw_year": model_net_value,
        "benchmark_energy_revenue_per_mw_year": energy_per_mw_year,
        "benchmark_total_revenue_per_mw_year": total_per_mw_year,
        "model_less_than_benchmark_total": (
            bool(model_net_value < total_per_mw_year)
            if np.isfinite(total_per_mw_year)
            else None
        ),
        "comparison_is_like_for_like": bool(
            aligned
            and benchmark["region"] == model_summary.get("region")
            and benchmark["fleet_scope"] == "single_reference_asset"
        ),
        "scope_warning": (
            "Published figures include FCAS and/or a fleet with different capacity; "
            "do not interpret the difference as asset outperformance."
        ),
        "source_url": benchmark["source_url"],
        "capacity_basis_name": benchmark["capacity_basis_name"],
    }
