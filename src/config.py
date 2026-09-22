"""Model configuration and dated NEM market settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Mapping

import pandas as pd


AEST = timezone(timedelta(hours=10), name="AEST")
MARKET_FLOOR_PRICE = -1_000.0
FIVE_MINUTES_HOURS = 1.0 / 12.0

# AEMC schedules of reliability settings. Unknown years deliberately fail rather
# than reusing a stale cap.
MARKET_PRICE_CAP_BY_FY: dict[str, float] = {
    "2021-22": 15_100.0,
    "2022-23": 15_500.0,
    "2023-24": 16_600.0,
    "2024-25": 17_500.0,
    "2025-26": 20_300.0,
    "2026-27": 23_200.0,
}

JUNE_2022_EVENT_START = pd.Timestamp("2022-06-12 18:50:00", tz=AEST)
JUNE_2022_EVENT_END = pd.Timestamp("2022-06-24 14:00:00", tz=AEST)


def financial_year(value: object) -> str:
    """Return the FY containing an interval-ending timestamp.

    The interval stamped 1 July 00:00 covers the final five minutes of 30 June,
    so it belongs to the financial year which has just ended.
    """
    ts = pd.Timestamp(value) - pd.Timedelta(nanoseconds=1)
    year = ts.year if ts.month >= 7 else ts.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def market_price_cap(value: object) -> float:
    """Look up the price cap for a timestamp, failing on unconfigured years."""
    fy = financial_year(value)
    try:
        return MARKET_PRICE_CAP_BY_FY[fy]
    except KeyError as exc:
        raise KeyError(f"No market price cap configured for financial year {fy}") from exc


@dataclass(frozen=True)
class LossFactors:
    """Annual marginal loss factors applied at settlement."""

    generation: float
    load: float
    source: str | None = None

    def __post_init__(self) -> None:
        if self.generation <= 0 or self.load <= 0:
            raise ValueError("Loss factors must be positive")
        if self.source is not None and not self.source.strip():
            raise ValueError("Loss-factor source must be non-empty when supplied")


@dataclass(frozen=True)
class LossFactorTable:
    """Asset-specific MLFs keyed by financial year."""

    by_financial_year: Mapping[str, LossFactors]

    def for_timestamp(self, value: object) -> LossFactors:
        fy = financial_year(value)
        try:
            return self.by_financial_year[fy]
        except KeyError as exc:
            raise KeyError(f"No loss factors configured for financial year {fy}") from exc

    def for_financial_year(self, fy: str) -> LossFactors:
        try:
            return self.by_financial_year[fy]
        except KeyError as exc:
            raise KeyError(f"No loss factors configured for financial year {fy}") from exc


@dataclass(frozen=True)
class BatteryConfig:
    """Physical and settlement parameters for a battery run.

    The loss factors are explicit and scalar because a run/window belongs to one
    financial year. ``LossFactorTable`` is used by callers to construct the
    corresponding config for each year.
    """

    generation_loss_factor: float
    load_loss_factor: float
    power_mw: float = 100.0
    energy_mwh: float = 200.0
    round_trip_efficiency: float = 0.87
    soc_min_fraction: float = 0.05
    soc_max_fraction: float = 0.95
    interval_minutes: int = 5
    region: str = "SA1"
    connection_point: str | None = None
    loss_factor_table: LossFactorTable | None = None
    loss_factor_source: str | None = None

    def __post_init__(self) -> None:
        if self.power_mw <= 0 or self.energy_mwh <= 0:
            raise ValueError("Power and energy ratings must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("Round-trip efficiency must be in (0, 1]")
        if not 0 <= self.soc_min_fraction < self.soc_max_fraction <= 1:
            raise ValueError("SOC fractions must satisfy 0 <= min < max <= 1")
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        if self.generation_loss_factor <= 0 or self.load_loss_factor <= 0:
            raise ValueError("Loss factors must be positive")
        if self.loss_factor_source is not None and not self.loss_factor_source.strip():
            raise ValueError("loss_factor_source must be non-empty when supplied")

    @property
    def charge_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def discharge_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def interval_hours(self) -> float:
        return self.interval_minutes / 60.0

    @property
    def soc_min_mwh(self) -> float:
        return self.energy_mwh * self.soc_min_fraction

    @property
    def soc_max_mwh(self) -> float:
        return self.energy_mwh * self.soc_max_fraction

    @property
    def usable_energy_mwh(self) -> float:
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def initial_soc_mwh(self) -> float:
        # Starting at the lower operating bound avoids treating unpriced opening
        # inventory as arbitrage revenue in a finite backtest.
        return self.soc_min_mwh

    @property
    def loss_factors(self) -> LossFactors:
        return LossFactors(
            self.generation_loss_factor,
            self.load_loss_factor,
            self.loss_factor_source,
        )

    def loss_factors_for(self, value: object) -> LossFactors:
        """Resolve dated MLFs, falling back to the explicit run-level pair."""
        if self.loss_factor_table is not None:
            return self.loss_factor_table.for_timestamp(value)
        return self.loss_factors

    def with_loss_factors(self, factors: LossFactors) -> "BatteryConfig":
        return replace(
            self,
            generation_loss_factor=factors.generation,
            load_loss_factor=factors.load,
            loss_factor_table=None,
            loss_factor_source=factors.source,
        )

    def config_hash(self, *, extra: Mapping[str, object] | None = None) -> str:
        payload: dict[str, object] = asdict(self)
        if extra:
            payload.update(extra)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def reference_asset(
    *,
    generation_loss_factor: float,
    load_loss_factor: float,
    loss_factor_source: str,
) -> BatteryConfig:
    """Build the 100 MW / 200 MWh, 87%-efficient SA1 reference asset."""
    return BatteryConfig(
        generation_loss_factor=generation_loss_factor,
        load_loss_factor=load_loss_factor,
        loss_factor_source=loss_factor_source,
    )
