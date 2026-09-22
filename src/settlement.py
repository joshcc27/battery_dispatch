"""Authoritative conversion from a dispatch trace to settled energy revenue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .battery import validate_dispatch_trace
from .config import BatteryConfig, financial_year


@dataclass(frozen=True)
class RevenueResult:
    """Interval settlement detail and aggregate values."""

    intervals: pd.DataFrame
    energy_revenue: float
    degradation_cost: float
    total_revenue: float


@dataclass(frozen=True)
class SettlementCoefficients:
    """Per-MW interval cashflows used by both optimisation and reporting."""

    generation_value_per_mw: np.ndarray
    charge_value_per_mw: np.ndarray
    degradation_cost_per_mw: np.ndarray
    discharge_value_per_mw: np.ndarray
    generation_loss_factor: np.ndarray
    load_loss_factor: np.ndarray


def settlement_coefficients(
    prices: np.ndarray,
    asset: BatteryConfig,
    *,
    settlementdate: object | None = None,
    deg_cost: float = 0.0,
) -> SettlementCoefficients:
    """Return authoritative marginal settled cashflows for a dispatch window."""
    price = np.asarray(prices, dtype=float)
    if price.ndim != 1 or len(price) == 0 or not np.isfinite(price).all():
        raise ValueError("prices must be a finite, non-empty one-dimensional array")
    if deg_cost < 0:
        raise ValueError("deg_cost must be non-negative")

    if settlementdate is None:
        if asset.loss_factor_table is not None:
            raise ValueError("settlementdate is required with a dated loss-factor table")
        generation = np.full(len(price), asset.generation_loss_factor, dtype=float)
        load = np.full(len(price), asset.load_loss_factor, dtype=float)
    else:
        timestamps = pd.DatetimeIndex(settlementdate)
        if len(timestamps) != len(price):
            raise ValueError("Timestamp and price lengths differ")
        financial_years = {financial_year(timestamp) for timestamp in timestamps}
        if asset.loss_factor_table is None and len(financial_years) > 1:
            raise ValueError(
                "A window spanning financial years requires a dated loss-factor table"
            )
        factors = [asset.loss_factors_for(timestamp) for timestamp in timestamps]
        generation = np.asarray([factor.generation for factor in factors], dtype=float)
        load = np.asarray([factor.load for factor in factors], dtype=float)

    dt = asset.interval_hours
    generation_value = dt * generation * price
    charge_value = -dt * load * price
    degradation = np.full(len(price), dt * deg_cost, dtype=float)
    return SettlementCoefficients(
        generation_value_per_mw=generation_value,
        charge_value_per_mw=charge_value,
        degradation_cost_per_mw=degradation,
        discharge_value_per_mw=generation_value - degradation,
        generation_loss_factor=generation,
        load_loss_factor=load,
    )


def _price_frame(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("A price Series must have a DatetimeIndex")
        result = prices.rename("rrp").rename_axis("settlementdate").reset_index()
    else:
        result = prices.copy()
        result.columns = [str(column).lower() for column in result.columns]
        if "settlementdate" not in result and isinstance(result.index, pd.DatetimeIndex):
            result = result.rename_axis("settlementdate").reset_index()
    required = {"settlementdate", "rrp"}
    if missing := required.difference(result.columns):
        raise ValueError(f"Prices are missing columns: {sorted(missing)}")
    if result["settlementdate"].duplicated().any():
        raise ValueError("Prices contain duplicate settlement timestamps")
    return result[["settlementdate", "rrp"]]


def settle(
    dispatch_trace: pd.DataFrame,
    prices: pd.DataFrame | pd.Series,
    asset: BatteryConfig,
    *,
    deg_cost: float = 0.0,
) -> RevenueResult:
    """Settle one meter-side dispatch trace against RRP.

    This is the only reporting path that combines prices and dispatched energy.
    Efficiency is already represented by the SOC trace; applying it again here
    would double-count losses.
    """
    if deg_cost < 0:
        raise ValueError("deg_cost must be non-negative")
    trace = dispatch_trace.copy()
    trace.columns = [str(column).lower() for column in trace.columns]
    validate_dispatch_trace(trace, asset)
    price = _price_frame(prices)
    if trace["settlementdate"].duplicated().any():
        raise ValueError("Dispatch trace contains duplicate settlement timestamps")
    merged = trace.merge(price, on="settlementdate", how="left", validate="one_to_one")
    if merged["rrp"].isna().any():
        missing = merged.loc[merged["rrp"].isna(), "settlementdate"].head(3).tolist()
        raise ValueError(f"Missing prices for dispatch timestamps, e.g. {missing}")
    if len(merged) != len(price):
        extra = price.loc[~price["settlementdate"].isin(trace["settlementdate"])]
        if not extra.empty:
            raise ValueError("Prices and dispatch trace do not cover exactly the same timestamps")

    dt = asset.interval_hours
    merged["charge_mwh"] = merged["charge_mw"] * dt
    merged["discharge_mwh"] = merged["discharge_mw"] * dt
    coefficients = settlement_coefficients(
        merged["rrp"].to_numpy(dtype=float),
        asset,
        settlementdate=merged["settlementdate"],
        deg_cost=deg_cost,
    )
    merged["generation_loss_factor"] = coefficients.generation_loss_factor
    merged["load_loss_factor"] = coefficients.load_loss_factor
    merged["generation_revenue"] = (
        coefficients.generation_value_per_mw * merged["discharge_mw"]
    )
    merged["load_cost"] = -coefficients.charge_value_per_mw * merged["charge_mw"]
    merged["energy_revenue"] = merged["generation_revenue"] - merged["load_cost"]
    merged["degradation_cost"] = (
        coefficients.degradation_cost_per_mw * merged["discharge_mw"]
    )
    merged["total_revenue"] = merged["energy_revenue"] - merged["degradation_cost"]
    energy_revenue = float(merged["energy_revenue"].sum())
    degradation_cost = float(merged["degradation_cost"].sum())
    return RevenueResult(
        intervals=merged,
        energy_revenue=energy_revenue,
        degradation_cost=degradation_cost,
        total_revenue=energy_revenue - degradation_cost,
    )


def break_even_discharge_price(
    charge_price: float,
    asset: BatteryConfig,
    *,
    deg_cost: float = 0.0,
    settlementdate: object | None = None,
) -> float:
    """Minimum discharge RRP for a marginal complete cycle."""
    eta_rt = asset.charge_efficiency * asset.discharge_efficiency
    factors = (
        asset.loss_factors
        if settlementdate is None
        else asset.loss_factors_for(settlementdate)
    )
    energy_term = factors.load * charge_price / (factors.generation * eta_rt)
    return energy_term + deg_cost / factors.generation
