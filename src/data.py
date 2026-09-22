"""NEMOSIS-backed, partitioned market-data cache with strict ingest checks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import (
    AEST,
    JUNE_2022_EVENT_END,
    JUNE_2022_EVENT_START,
    MARKET_FLOOR_PRICE,
    market_price_cap,
)


PRICE_COLUMNS = ["SETTLEMENTDATE", "REGIONID", "RRP", "INTERVENTION"]
CONTEXT_COLUMNS = [
    "SETTLEMENTDATE",
    "REGIONID",
    "INTERVENTION",
    "TOTALDEMAND",
    "AVAILABLEGENERATION",
]


def _aest_series(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(AEST)
    return timestamps.dt.tz_convert(AEST)


def _assert_complete_range(
    frame: pd.DataFrame,
    *,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> None:
    frequency = pd.Timedelta(minutes=5)
    if (expected_start is None) != (expected_end is None):
        raise ValueError("expected_start and expected_end must be supplied together")
    if expected_start is not None:
        if expected_end <= expected_start:
            raise ValueError("expected_end must be after expected_start")
        # Inputs are physical period boundaries while AEMO timestamps are
        # interval-ending: (start, end].
        expected = pd.date_range(expected_start + frequency, expected_end, freq=frequency)
    for region, group in frame.groupby("regionid", sort=False):
        region_expected = expected if expected_start is not None else pd.date_range(
            group["settlementdate"].min(),
            group["settlementdate"].max(),
            freq=frequency,
        )
        actual = pd.DatetimeIndex(group["settlementdate"])
        missing = region_expected.difference(actual)
        unexpected = actual.difference(region_expected)
        if len(missing):
            raise ValueError(f"Data for {region} has {len(missing)} gaps; first is {missing[0]}")
        if len(unexpected):
            raise ValueError(f"Data for {region} has timestamps outside the expected range")


def validate_prices(
    data: pd.DataFrame,
    *,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalise DISPATCHPRICE rows and assert the complete ingest contract."""
    frame = data.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"settlementdate", "regionid", "rrp", "intervention"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"DISPATCHPRICE data is missing columns: {sorted(missing)}")
    frame["settlementdate"] = _aest_series(frame["settlementdate"])
    frame["intervention"] = pd.to_numeric(frame["intervention"], errors="raise")
    frame["rrp"] = pd.to_numeric(frame["rrp"], errors="raise")
    frame = frame.loc[frame["intervention"] == 0].copy()
    frame = frame.sort_values(["regionid", "settlementdate"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No non-intervention price rows remain")
    key = ["settlementdate", "regionid"]
    if frame.duplicated(key).any():
        examples = frame.loc[frame.duplicated(key, keep=False), key].head(3).to_dict("records")
        raise ValueError(f"Duplicate price keys after intervention filtering: {examples}")
    if frame[["settlementdate", "regionid", "rrp"]].isna().any().any():
        raise ValueError("Price data contains NaNs")

    _assert_complete_range(
        frame, expected_start=expected_start, expected_end=expected_end
    )

    caps = frame["settlementdate"].map(market_price_cap).to_numpy(dtype=float)
    invalid = (frame["rrp"].to_numpy() < MARKET_FLOOR_PRICE) | (
        frame["rrp"].to_numpy() > caps
    )
    if invalid.any():
        examples = frame.loc[invalid, ["settlementdate", "regionid", "rrp"]].head(3)
        raise ValueError(f"RRP lies outside the dated market bounds:\n{examples}")
    frame["june_2022_event"] = frame["settlementdate"].between(
        JUNE_2022_EVENT_START, JUNE_2022_EVENT_END, inclusive="both"
    )
    return frame


def validate_context(
    data: pd.DataFrame,
    *,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalise and validate context-only DISPATCHREGIONSUM rows."""
    frame = data.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {
        "settlementdate",
        "regionid",
        "intervention",
        "totaldemand",
        "availablegeneration",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"DISPATCHREGIONSUM data is missing columns: {sorted(missing)}")
    frame["settlementdate"] = _aest_series(frame["settlementdate"])
    for column in ["intervention", "totaldemand", "availablegeneration"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.loc[frame["intervention"] == 0].copy()
    frame = frame.sort_values(["regionid", "settlementdate"]).reset_index(drop=True)
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Context data is empty or contains NaNs")
    if frame.duplicated(["settlementdate", "regionid"]).any():
        raise ValueError("Duplicate context keys after intervention filtering")
    _assert_complete_range(
        frame, expected_start=expected_start, expected_end=expected_end
    )
    return frame


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first_interval_end = start + pd.Timedelta(minutes=5)
    first = first_interval_end.tz_localize(None).to_period("M").start_time
    last = end.tz_localize(None).to_period("M").start_time
    return list(pd.date_range(first, last, freq="MS"))


class MarketDataCache:
    """One parquet per table/region/month plus an auditable JSONL manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.partition_root = self.root / "partitions"
        self.raw_root = self.root / "nemosis_raw"
        self.manifest_path = self.root / "manifest.jsonl"

    def partition_path(self, table: str, region: str, month: pd.Timestamp) -> Path:
        return (
            self.partition_root
            / table.lower()
            / f"region={region}"
            / f"year={month.year:04d}"
            / f"month={month.month:02d}.parquet"
        )

    def _record_manifest(
        self,
        *,
        table: str,
        region: str,
        month: pd.Timestamp,
        path: Path,
        row_count: int,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "table": table,
            "region": region,
            "month": month.strftime("%Y-%m"),
            "path": path.relative_to(self.root).as_posix(),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": row_count,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def fetch_prices(
        self,
        start: object,
        end: object,
        regions: Iterable[str],
        *,
        compiler: Callable[..., pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Return interval-ending prices for physical period ``[start, end)``.

        Consequently the returned timestamp range is ``(start, end]``.
        Existing partitions never hit NEMWEB.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        start_ts = start_ts.tz_localize(AEST) if start_ts.tz is None else start_ts.tz_convert(AEST)
        end_ts = end_ts.tz_localize(AEST) if end_ts.tz is None else end_ts.tz_convert(AEST)
        if end_ts <= start_ts:
            raise ValueError("end must be after start")
        if start_ts < pd.Timestamp("2021-10-01", tz=AEST):
            raise ValueError("The study period starts at five-minute settlement on 2021-10-01")
        region_list = list(dict.fromkeys(regions))
        if not region_list:
            raise ValueError("At least one region is required")

        partitions: list[pd.DataFrame] = []
        for month in _month_starts(start_ts, end_ts):
            month_end = month + pd.offsets.MonthBegin(1)
            for region in region_list:
                path = self.partition_path("DISPATCHPRICE", region, month)
                if path.exists():
                    partitions.append(pd.read_parquet(path))
                    continue
                if compiler is None:
                    try:
                        from nemosis import dynamic_data_compiler
                    except ImportError as exc:
                        raise RuntimeError("Install nemosis to fetch missing market data") from exc
                    compiler = dynamic_data_compiler
                self.raw_root.mkdir(parents=True, exist_ok=True)
                raw = compiler(
                    month.strftime("%Y/%m/%d 00:00:00"),
                    month_end.strftime("%Y/%m/%d 00:00:00"),
                    "DISPATCHPRICE",
                    str(self.raw_root),
                    select_columns=PRICE_COLUMNS,
                    filter_cols=["REGIONID"],
                    filter_values=([region],),
                    keep_csv=False,
                    keep_zip=True,
                )
                normal = validate_prices(raw)
                month_start_aest = month.tz_localize(AEST)
                month_end_aest = month_end.tz_localize(AEST)
                normal = normal.loc[
                    normal["settlementdate"].between(
                        month_start_aest, month_end_aest, inclusive="left"
                    )
                ].copy()
                if normal.empty:
                    raise ValueError(f"NEMOSIS returned no rows for {region} in {month:%Y-%m}")
                path.parent.mkdir(parents=True, exist_ok=True)
                normal.to_parquet(path, index=False)
                self._record_manifest(
                    table="DISPATCHPRICE",
                    region=region,
                    month=month,
                    path=path,
                    row_count=len(normal),
                )
                partitions.append(normal)

        combined = pd.concat(partitions, ignore_index=True)
        combined["settlementdate"] = _aest_series(combined["settlementdate"])
        combined = combined.loc[
            combined["settlementdate"].gt(start_ts)
            & combined["settlementdate"].le(end_ts)
            & combined["regionid"].isin(region_list)
        ]
        return validate_prices(
            combined, expected_start=start_ts, expected_end=end_ts
        )

    def fetch_context(
        self,
        start: object,
        end: object,
        regions: Iterable[str],
        *,
        compiler: Callable[..., pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Fetch demand/available-generation context with the same cache contract."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        start_ts = start_ts.tz_localize(AEST) if start_ts.tz is None else start_ts.tz_convert(AEST)
        end_ts = end_ts.tz_localize(AEST) if end_ts.tz is None else end_ts.tz_convert(AEST)
        if end_ts <= start_ts:
            raise ValueError("end must be after start")
        region_list = list(dict.fromkeys(regions))
        if not region_list:
            raise ValueError("At least one region is required")
        partitions: list[pd.DataFrame] = []
        for month in _month_starts(start_ts, end_ts):
            month_end = month + pd.offsets.MonthBegin(1)
            for region in region_list:
                path = self.partition_path("DISPATCHREGIONSUM", region, month)
                if path.exists():
                    partitions.append(pd.read_parquet(path))
                    continue
                if compiler is None:
                    try:
                        from nemosis import dynamic_data_compiler
                    except ImportError as exc:
                        raise RuntimeError("Install nemosis to fetch missing market data") from exc
                    compiler = dynamic_data_compiler
                self.raw_root.mkdir(parents=True, exist_ok=True)
                raw = compiler(
                    month.strftime("%Y/%m/%d 00:00:00"),
                    month_end.strftime("%Y/%m/%d 00:00:00"),
                    "DISPATCHREGIONSUM",
                    str(self.raw_root),
                    select_columns=CONTEXT_COLUMNS,
                    filter_cols=["REGIONID"],
                    filter_values=([region],),
                    keep_csv=False,
                    keep_zip=True,
                )
                normal = validate_context(raw)
                month_start_aest = month.tz_localize(AEST)
                month_end_aest = month_end.tz_localize(AEST)
                normal = normal.loc[
                    normal["settlementdate"].between(
                        month_start_aest, month_end_aest, inclusive="left"
                    )
                ].copy()
                if normal.empty:
                    raise ValueError(f"NEMOSIS returned no context for {region} in {month:%Y-%m}")
                path.parent.mkdir(parents=True, exist_ok=True)
                normal.to_parquet(path, index=False)
                self._record_manifest(
                    table="DISPATCHREGIONSUM",
                    region=region,
                    month=month,
                    path=path,
                    row_count=len(normal),
                )
                partitions.append(normal)
        combined = pd.concat(partitions, ignore_index=True)
        combined["settlementdate"] = _aest_series(combined["settlementdate"])
        combined = combined.loc[
            combined["settlementdate"].gt(start_ts)
            & combined["settlementdate"].le(end_ts)
            & combined["regionid"].isin(region_list)
        ]
        return validate_context(
            combined, expected_start=start_ts, expected_end=end_ts
        )
