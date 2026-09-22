"""Independent run-audit calculations used by CLI and regression tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .battery import (
    SOLVER_INTEGRALITY_TOLERANCE,
    energy_balance_residual,
    soc_path,
)
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
        # min(charge, discharge) in MW, not the product: see validate_dispatch_trace.
        "max_simultaneous_charge_discharge_mw": float(
            np.max(np.minimum(charge, discharge))
        ),
        # The same quantity as a fraction of the power rating, which is what the
        # gate tests so that one tolerance is meaningful at any asset size.
        "max_simultaneous_dispatch_fraction": float(
            np.max(np.minimum(charge, discharge)) / asset.power_mw
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
        # Windows whose exclusivity binaries were all proven redundant solve as
        # pure LPs and report a zero gap exactly, not approximately.
        "pure_lp_window_count": (
            int(
                trace.groupby("window_id")["exclusivity_binary_count"].first().eq(0).sum()
            )
            if "exclusivity_binary_count" in trace
            else 0
        ),
        "total_exclusivity_binary_count": (
            int(trace.groupby("window_id")["exclusivity_binary_count"].first().sum())
            if "exclusivity_binary_count" in trace
            else np.nan
        ),
        "solver_non_optimal_window_count": (
            int(
                (~window_statuses["solver_termination"].str.lower().eq("optimal")).sum()
            )
            if not window_statuses.empty
            else 0
        ),
    }


PRICE_AUDIT_KEYS = (
    "price_duplicate_count",
    "price_missing_interval_count",
    "price_unexpected_interval_count",
)

DISPATCH_AUDIT_KEYS = (
    "energy_balance_residual_mwh",
    "max_interval_energy_balance_error_mwh",
    "max_simultaneous_dispatch_fraction",
    "solver_all_optimal",
    "max_solver_mip_gap",
    "completed_cycle_break_even_violation_count",
)


def assert_audit_passes(
    audit: Mapping[str, object],
    *,
    energy_tolerance: float = 1e-6,
    exclusivity_tolerance: float = SOLVER_INTEGRALITY_TOLERANCE,
    mip_gap_tolerance: float = 1.01e-6,
    require_price_audit: bool = False,
) -> None:
    """Fail a run that misses any declared gate.

    A gate whose key is absent is a missing check, not a passing one, so the
    required keys are asserted before any of them is evaluated. Callers that
    merged :func:`audit_prices` set ``require_price_audit`` to gate the inputs
    as well; callers that settle against already-audited prices do not.
    """
    required = list(DISPATCH_AUDIT_KEYS)
    if require_price_audit:
        required += list(PRICE_AUDIT_KEYS)
    if absent := [key for key in required if key not in audit]:
        raise ValueError(f"Run audit is incomplete, missing: {sorted(absent)}")

    failures: list[str] = []
    if require_price_audit:
        for field in PRICE_AUDIT_KEYS:
            if int(audit[field]) != 0:
                failures.append(f"{field}={audit[field]}")
    if abs(float(audit["energy_balance_residual_mwh"])) > energy_tolerance:
        failures.append("aggregate energy balance")
    if float(audit["max_interval_energy_balance_error_mwh"]) > energy_tolerance:
        failures.append("interval energy balance")
    if float(audit["max_simultaneous_dispatch_fraction"]) > exclusivity_tolerance:
        failures.append("charge/discharge exclusivity")
    if not bool(audit["solver_all_optimal"]):
        failures.append("non-optimal solver window")
    if float(audit["max_solver_mip_gap"]) > mip_gap_tolerance:
        failures.append("solver MIP gap")
    if int(audit["completed_cycle_break_even_violation_count"]) != 0:
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
