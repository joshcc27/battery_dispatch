from __future__ import annotations

import pandas as pd
import pytest

from battery_dispatch.config import AEST
from battery_dispatch.data import MarketDataCache, validate_context, validate_prices


def rows(prices=(10.0, 20.0, 30.0)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SETTLEMENTDATE": pd.date_range("2022-01-01 00:05", periods=len(prices), freq="5min"),
            "REGIONID": "SA1",
            "RRP": prices,
            "INTERVENTION": 0,
        }
    )


def test_filters_intervention_and_flags_aest() -> None:
    frame = rows()
    duplicate = frame.iloc[[1]].assign(INTERVENTION=1, RRP=999)
    result = validate_prices(pd.concat([frame, duplicate], ignore_index=True))
    assert len(result) == 3
    assert str(result["settlementdate"].dt.tz) == "AEST"


def test_june_2022_event_includes_suspension_end() -> None:
    frame = pd.DataFrame(
        {
            "SETTLEMENTDATE": ["2022-06-24 13:55", "2022-06-24 14:00", "2022-06-24 14:05"],
            "REGIONID": "SA1",
            "RRP": 100.0,
            "INTERVENTION": 0,
        }
    )
    result = validate_prices(frame)
    assert result["june_2022_event"].tolist() == [True, True, False]


def test_rejects_duplicate_gap_nan_and_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prices(pd.concat([rows(), rows().iloc[[1]]], ignore_index=True))
    with pytest.raises(ValueError, match="gaps"):
        validate_prices(rows().iloc[[0, 2]])
    bad_nan = rows()
    bad_nan.loc[1, "RRP"] = None
    with pytest.raises(ValueError, match="NaNs"):
        validate_prices(bad_nan)
    with pytest.raises(ValueError, match="outside"):
        validate_prices(rows((10, 16_000, 20)))


def test_explicit_range_catches_missing_boundary_and_context_is_filtered() -> None:
    start = pd.Timestamp("2022-01-01 00:00", tz=AEST)
    end = pd.Timestamp("2022-01-01 00:15", tz=AEST)
    with pytest.raises(ValueError, match="gaps"):
        validate_prices(rows().iloc[1:], expected_start=start, expected_end=end)
    context = rows().rename(columns={"RRP": "TOTALDEMAND"})
    context["AVAILABLEGENERATION"] = [100, 110, 120]
    intervention = context.iloc[[1]].assign(INTERVENTION=1)
    result = validate_context(
        pd.concat([context, intervention]), expected_start=start, expected_end=end
    )
    assert len(result) == 3


def test_cache_hit_never_calls_compiler(tmp_path) -> None:
    calls = 0

    def compiler(*args, **kwargs):
        nonlocal calls
        calls += 1
        return rows((10, 20))

    cache = MarketDataCache(tmp_path)
    start = pd.Timestamp("2022-01-01 00:00", tz=AEST)
    end = pd.Timestamp("2022-01-01 00:10", tz=AEST)
    first = cache.fetch_prices(start, end, ["SA1"], compiler=compiler)
    assert calls == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("cache hit touched the network compiler")

    second = cache.fetch_prices(start, end, ["SA1"], compiler=forbidden)
    assert calls == 1
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert cache.manifest_path.exists()
