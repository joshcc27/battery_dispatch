"""Verify that critical economic and data-contract mutations turn tests red."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    (
        "loss factors",
        "settlement.py",
        "generation_value = dt * generation * price",
        "generation_value = dt * price",
        [
            "tests/test_optimiser.py::test_core_policy_optimises_settled_loss_factor_cashflows",
            "tests/test_optimiser.py::test_completed_cycle_respects_loss_factor_break_even",
        ],
    ),
    (
        "charge efficiency",
        "optimiser.py",
        "eta_c = asset.charge_efficiency",
        "eta_c = 1.0",
        ["tests/test_optimiser.py::test_energy_balance_soc_bounds_and_negative_price_binary"],
    ),
    (
        "binary exclusivity",
        "optimiser.py",
        "mode = model.add_variables(binary=True, coords=[intervals], name=\"charge_mode\")",
        "mode = model.add_variables(lower=0.0, upper=1.0, coords=[intervals], name=\"charge_mode\")",
        ["tests/test_optimiser.py::test_energy_balance_soc_bounds_and_negative_price_binary"],
    ),
    (
        "settlement sign",
        "settlement.py",
        "merged[\"energy_revenue\"] = merged[\"generation_revenue\"] - merged[\"load_cost\"]",
        "merged[\"energy_revenue\"] = merged[\"generation_revenue\"] + merged[\"load_cost\"]",
        ["tests/test_settlement.py::test_hand_computed_six_interval_settlement"],
    ),
    (
        "degradation sign",
        "settlement.py",
        "degradation = np.full(len(price), dt * deg_cost, dtype=float)",
        "degradation = np.full(len(price), -dt * deg_cost, dtype=float)",
        ["tests/test_optimiser.py::test_realised_revenue_is_non_increasing_in_degradation_cost"],
    ),
    (
        "intervention filter",
        "data.py",
        "frame[\"rrp\"] = pd.to_numeric(frame[\"rrp\"], errors=\"raise\")\n    frame = frame.loc[frame[\"intervention\"] == 0].copy()",
        "frame[\"rrp\"] = pd.to_numeric(frame[\"rrp\"], errors=\"raise\")\n    frame = frame.loc[frame[\"intervention\"] == 1].copy()",
        ["tests/test_data.py::test_filters_intervention_and_flags_aest"],
    ),
    (
        "interval ending",
        "data.py",
        "expected = pd.date_range(expected_start + frequency, expected_end, freq=frequency)",
        "expected = pd.date_range(expected_start, expected_end - frequency, freq=frequency)",
        ["tests/test_data.py::test_cache_interval_ending_range_across_financial_year"],
    ),
    (
        "SOC carry",
        "horizon.py",
        "current_soc = float(part[\"soc_mwh\"].iloc[-1])",
        "current_soc = asset.initial_soc_mwh",
        ["tests/test_horizon_metrics.py::test_rolling_horizon_settlement_metrics_and_outputs"],
    ),
    (
        "committed prefix",
        "horizon.py",
        ".iloc[:take].copy()",
        ".iloc[-take:].copy()",
        ["tests/test_horizon_metrics.py::test_rolling_horizon_settlement_metrics_and_outputs"],
    ),
]


def main() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="battery-dispatch-mutants-") as temporary:
        temp = Path(temporary)
        shutil.copytree(ROOT / "src", temp / "battery_dispatch")
        shutil.copytree(ROOT / "tests", temp / "tests")
        for name, relative, old, new, tests in MUTATIONS:
            target = temp / "battery_dispatch" / relative
            original = target.read_text(encoding="utf-8")
            if original.count(old) != 1:
                failures.append(f"{name}: mutation target count was {original.count(old)}")
                continue
            target.write_text(original.replace(old, new), encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(temp)
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", *tests],
                cwd=temp,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            target.write_text(original, encoding="utf-8")
            if result.returncode == 0:
                failures.append(f"{name}: tests stayed green")
            else:
                print(f"PASS {name}: mutation detected")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
