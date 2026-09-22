"""Dropping provably-redundant exclusivity binaries must not change dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from battery_dispatch.config import BatteryConfig
from battery_dispatch.optimiser import (
    exclusivity_binding_mask,
    solve_window,
)
from battery_dispatch.settlement import settlement_coefficients


def make_asset(**changes: float) -> BatteryConfig:
    values = {
        "generation_loss_factor": 0.9652,
        "load_loss_factor": 0.9681,
        "power_mw": 100.0,
        "energy_mwh": 200.0,
        "round_trip_efficiency": 0.87,
        "interval_minutes": 5,
    }
    values.update(changes)
    return BatteryConfig(**values)


def binding_mask(prices: np.ndarray, asset: BatteryConfig, deg_cost: float) -> np.ndarray:
    return exclusivity_binding_mask(
        settlement_coefficients(prices, asset, deg_cost=deg_cost), asset
    )


def test_mask_matches_the_documented_price_threshold() -> None:
    """The mask must reproduce p < -6.776 * deg for the reference factors."""
    asset = make_asset()
    deg_cost = 25.0
    eta_rt = asset.charge_efficiency * asset.discharge_efficiency
    threshold = -deg_cost * eta_rt / (
        asset.load_loss_factor - asset.generation_loss_factor * eta_rt
    )
    assert threshold == pytest.approx(-169.42, abs=0.01)

    prices = np.array([threshold - 1.0, threshold + 1.0], dtype=float)
    mask = binding_mask(prices, asset, deg_cost)
    assert mask[0]
    assert not mask[1]


def test_zero_degradation_keeps_a_binary_at_every_negative_price() -> None:
    """At A$0/MWh the threshold collapses to p < 0, including exactly zero."""
    asset = make_asset()
    prices = np.array([-100.0, -0.01, 0.0, 0.01, 100.0], dtype=float)
    mask = binding_mask(prices, asset, 0.0)
    assert list(mask) == [True, True, True, False, False]


def _solve_with_all_binaries(prices, asset, soc_initial, deg_cost, monkeypatch):
    """Solve with the original formulation: one binary on every interval."""
    monkeypatch.setattr(
        "battery_dispatch.optimiser.exclusivity_binding_mask",
        lambda cashflows, asset, **kwargs: np.ones(
            len(cashflows.charge_value_per_mw), dtype=bool
        ),
    )
    return solve_window(prices, asset, soc_initial, deg_cost=deg_cost)


@pytest.mark.parametrize("deg_cost", [0.0, 25.0, 50.0])
def test_reduced_model_reproduces_the_full_binary_dispatch(
    deg_cost: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reduced MILP must return the same dispatch as the full one."""
    asset = make_asset()
    rng = np.random.default_rng(20220701)
    # A spread wide enough to straddle every threshold under test, including a
    # deeply negative tail where the binary genuinely binds.
    prices = np.concatenate(
        [
            rng.normal(80.0, 120.0, 180),
            rng.uniform(-400.0, 0.0, 60),
            rng.uniform(-1000.0, -150.0, 24),
        ]
    )
    soc_initial = asset.soc_min_mwh

    reduced = solve_window(prices, asset, soc_initial, deg_cost=deg_cost)
    full = _solve_with_all_binaries(prices, asset, soc_initial, deg_cost, monkeypatch)

    assert reduced.exclusivity_binary_count < len(prices)
    assert full.exclusivity_binary_count == len(prices)
    assert reduced.objective_value == pytest.approx(full.objective_value, rel=1e-9)
    np.testing.assert_allclose(reduced.charge_mw, full.charge_mw, atol=1e-6)
    np.testing.assert_allclose(reduced.discharge_mw, full.discharge_mw, atol=1e-6)
    np.testing.assert_allclose(reduced.soc_mwh, full.soc_mwh, atol=1e-6)


def test_reduced_model_still_forbids_simultaneous_operation() -> None:
    """Exclusivity must hold on the dropped intervals too, not just the kept ones."""
    asset = make_asset()
    prices = np.array([-50.0, -5.0, 0.0, 30.0, 300.0, -900.0] * 8, dtype=float)
    result = solve_window(prices, asset, asset.soc_min_mwh, deg_cost=25.0)
    assert result.exclusivity_binary_count < len(prices)
    assert np.max(result.charge_mw * result.discharge_mw) <= 1e-8


def test_headline_cost_eliminates_almost_every_binary() -> None:
    """A realistic price shape should leave only a handful of binaries."""
    asset = make_asset()
    rng = np.random.default_rng(7)
    prices = rng.normal(90.0, 110.0, 576)
    mask = binding_mask(prices, asset, 25.0)
    assert mask.mean() < 0.02
