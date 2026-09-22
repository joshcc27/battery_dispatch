"""Independent run-audit calculations used by CLI and regression tests."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import pandas as pd

from .battery import energy_balance_residual, soc_path
from .config import AEST, BatteryConfig


def _aest(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(AEST) if timestamp.tz is None else timestamp.tz_convert(AEST)


def dataframe_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    canonical = frame[columns].sort_values(columns[:2]).reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def audit_prices(
    prices: pd.DataFrame,
    *,
    expected_start: object,
    expected_end: object,
) -> dict[str, object]:
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"settlementdate", "regionid", "rrp"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"prices are missing audit columns: {sorted(missing)}")
    start = _aest(expected_start)
    end = _aest(expected_end)
    expected = pd.date_range(start + pd.Timedelta(minutes=5), end, freq="5min")
    duplicate_count = int(frame.duplicated(["settlementdate", "regionid"]).sum())
    missing_count = 0
    unexpected_count = 0
    for _, group in frame.groupby("regionid"):
        actual = pd.DatetimeIndex(group["settlementdate"])
        missing_count += len(expected.difference(actual))
        unexpected_count += len(actual.difference(expected))
    event_count = (
        int(frame["june_2022_event"].fillna(False).astype(bool).sum())
        if "june_2022_event" in frame
        else 0
    )
    return {
        "price_row_count": len(frame),
        "expected_price_row_count_per_region": len(expected),
        "price_region_count": int(frame["regionid"].nunique()),
        "price_duplicate_count": duplicate_count,
        "price_missing_interval_count": missing_count,
        "price_unexpected_interval_count": unexpected_count,
        "price_min_timestamp": frame["settlementdate"].min().isoformat(),
        "price_max_timestamp": frame["settlementdate"].max().isoformat(),
        "rrp_min": float(frame["rrp"].min()),
        "rrp_max": float(frame["rrp"].max()),
        "june_2022_event_interval_count": event_count,
        "price_sha256": dataframe_sha256(
            frame, ["settlementdate", "regionid", "rrp"]
        ),
    }


def audit_dispatch(
    trace: pd.DataFrame,
    asset: BatteryConfig,
    *,
    soc_initial: float,
) -> dict[str, object]:
    charge = trace["charge_mw"].to_numpy(dtype=float)
    discharge = trace["discharge_mw"].to_numpy(dtype=float)
    soc = trace["soc_mwh"].to_numpy(dtype=float)
    expected_soc = soc_path(charge, discharge, soc_initial, asset)
    interval_residual = soc - expected_soc
    terminal = float(soc[-1])
    window_statuses = (
        trace[["window_id", "solver_status", "solver_termination"]]
        .drop_duplicates("window_id")
        if "window_id" in trace
        else pd.DataFrame()
    )
    return {
        "energy_balance_residual_mwh": energy_balance_residual(
            charge, discharge, soc_initial, terminal, asset
        ),
        "max_interval_energy_balance_error_mwh": float(
            np.max(np.abs(interval_residual))
        ),
        "soc_min_observed_mwh": float(soc.min()),
        "soc_max_observed_mwh": float(soc.max()),
        "max_simultaneous_charge_discharge_mw2": float(
            np.max(charge * discharge)
        ),
        "max_charge_mw": float(charge.max()),
        "max_discharge_mw": float(discharge.max()),
        "initial_soc_mwh": float(soc_initial),
        "terminal_soc_mwh": terminal,
        "solver_window_count": len(window_statuses),
        "solver_all_optimal": bool(
            not window_statuses.empty
            and window_statuses["solver_status"].str.lower().eq("ok").all()
            and window_statuses["solver_termination"].str.lower().eq("optimal").all()
        ),
        "max_window_solve_seconds": (
            float(trace.groupby("window_id")["solve_seconds"].first().max())
            if "window_id" in trace
            else np.nan
        ),
        "max_solver_mip_gap": (
            float(trace.groupby("window_id")["solver_mip_gap"].first().max())
            if "solver_mip_gap" in trace
            else np.nan
        ),
        "max_solver_absolute_gap_aud": (
            float(trace.groupby("window_id")["solver_absolute_gap"].first().max())
            if "solver_absolute_gap" in trace
            else np.nan
        ),
        # Each window was solved to within its own absolute gap, so the summed
        # gap bounds how much window objective the dispatch decisions left on
        # the table. It is not a bound on the annual settled revenue, which is
        # the exact settlement of the dispatch that was actually implemented.
        "total_solver_absolute_gap_aud": (
            float(trace.groupby("window_id")["solver_absolute_gap"].first().sum())
            if "solver_absolute_gap" in trace
            else np.nan
        ),
        "solver_time_limited_window_count": (
            int(trace.groupby("window_id")["solver_time_limited"].first().sum())
            if "solver_time_limited" in trace
            else 0
        ),
        "solver_non_optimal_window_count": (
            int(
                (~window_statuses["solver_termination"].str.lower().eq("optimal")).sum()
            )
            if not window_statuses.empty
            else 0
        ),
    }


def assert_audit_passes(
    audit: Mapping[str, object],
    *,
    energy_tolerance: float = 1e-6,
    exclusivity_tolerance: float = 1e-8,
    mip_gap_tolerance: float = 1.01e-6,
    allow_time_limited: bool = False,
) -> None:
    """Fail a run that misses any declared gate.

    ``allow_time_limited`` accepts windows that HiGHS stopped at a requested
    time limit, and only those; a window that ended non-optimal for any other
    reason still fails. It is used by the documented A$0/MWh degradation
    scenario and must be paired with an explicit ``mip_gap_tolerance``.
    """
    failures: list[str] = []
    for field in (
        "price_duplicate_count",
        "price_missing_interval_count",
        "price_unexpected_interval_count",
    ):
        if field in audit and int(audit[field]) != 0:
            failures.append(f"{field}={audit[field]}")
    if abs(float(audit.get("energy_balance_residual_mwh", 0.0))) > energy_tolerance:
        failures.append("aggregate energy balance")
    if float(audit.get("max_interval_energy_balance_error_mwh", 0.0)) > energy_tolerance:
        failures.append("interval energy balance")
    if (
        float(audit.get("max_simultaneous_charge_discharge_mw2", 0.0))
        > exclusivity_tolerance
    ):
        failures.append("charge/discharge exclusivity")
    if "solver_all_optimal" in audit and not bool(audit["solver_all_optimal"]):
        non_optimal = int(audit.get("solver_non_optimal_window_count", 0))
        time_limited = int(audit.get("solver_time_limited_window_count", 0))
        if not allow_time_limited or non_optimal != time_limited or time_limited == 0:
            failures.append("non-optimal solver window")
    if (
        "max_solver_mip_gap" in audit
        and float(audit["max_solver_mip_gap"]) > mip_gap_tolerance
    ):
        failures.append("solver MIP gap")
    if int(audit.get("completed_cycle_break_even_violation_count", 0)) != 0:
        failures.append("completed-cycle break-even")
    if failures:
        raise ValueError("Run audit failed: " + ", ".join(failures))


def audit_completed_cycles(
    intervals: pd.DataFrame,
    asset: BatteryConfig,
    *,
    deg_cost: float,
    initial_soc_mwh: float,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Check discharges against the weighted cost of charged inventory.

    Opening inventory is tracked separately and is not called a completed cycle.
    Charged inventory is valued after charge efficiency; discharge revenue is
    allocated on the corresponding stored-energy basis.
    """
    charged_inventory = 0.0
    charged_inventory_cost = 0.0
    opening_inventory = max(0.0, initial_soc_mwh - asset.soc_min_mwh)
    margins: list[float] = []
    matched_connection_mwh = 0.0
    for row in intervals.itertuples(index=False):
        charge_mwh = float(row.charge_mwh)
        discharge_mwh = float(row.discharge_mwh)
        if charge_mwh > tolerance:
            stored = charge_mwh * asset.charge_efficiency
            charged_inventory += stored
            charged_inventory_cost += (
                float(row.load_loss_factor) * float(row.rrp) * charge_mwh
            )
        if discharge_mwh <= tolerance:
            continue
        stored_used = discharge_mwh / asset.discharge_efficiency
        opening_used = min(opening_inventory, stored_used)
        opening_inventory -= opening_used
        matched_stored = min(charged_inventory, stored_used - opening_used)
        if matched_stored <= tolerance:
            continue
        average_cost = charged_inventory_cost / charged_inventory
        cost = average_cost * matched_stored
        matched_fraction = matched_stored / stored_used
        revenue = (
            float(row.generation_loss_factor) * float(row.rrp) - deg_cost
        ) * discharge_mwh * matched_fraction
        margins.append(revenue - cost)
        matched_connection_mwh += discharge_mwh * matched_fraction
        charged_inventory -= matched_stored
        charged_inventory_cost -= cost
        if charged_inventory < tolerance:
            charged_inventory = 0.0
            charged_inventory_cost = 0.0
    violations = [margin for margin in margins if margin < -tolerance]
    return {
        "completed_cycle_discharge_mwh": matched_connection_mwh,
        "completed_cycle_segment_count": len(margins),
        "completed_cycle_break_even_violation_count": len(violations),
        "minimum_completed_cycle_margin": min(margins) if margins else np.nan,
    }
