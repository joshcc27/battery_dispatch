"""End-to-end CLI tests with a stubbed market-data compiler.

The CLI is the documented entry point and the only caller that gates inputs
with ``require_price_audit=True``, so these exercise argument handling, both
loss-factor modes and the refusal to write a run that fails its audit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from battery_dispatch import cli
from battery_dispatch.config import AEST

START = "2022-11-01 00:00"
END = "2022-11-01 02:00"
MLF_FILE = "config/loss_factors.csv"


def stub_compiler(*args, **kwargs):
    """One physical November 2022 day of alternating cheap/expensive prices.

    The cache asks for a whole calendar month, so this returns the whole month
    and the CLI's own window narrows it to the requested period.
    """
    timestamps = pd.date_range("2022-11-01 00:05", "2022-12-01 00:00", freq="5min")
    prices = [10.0 if index % 2 else 300.0 for index in range(len(timestamps))]
    return pd.DataFrame(
        {
            "SETTLEMENTDATE": timestamps,
            "REGIONID": "SA1",
            "RRP": prices,
            "INTERVENTION": 0,
        }
    )


@pytest.fixture
def cached_prices(tmp_path, monkeypatch):
    """A cache primed from the stub, so no test reaches the network."""
    from battery_dispatch.data import MarketDataCache

    cache_root = tmp_path / "cache"
    MarketDataCache(cache_root).fetch_prices(
        pd.Timestamp("2022-11-01 00:00", tz=AEST),
        pd.Timestamp("2022-12-01 00:00", tz=AEST),
        ["SA1"],
        compiler=stub_compiler,
    )
    return cache_root


def run_cli(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["battery-dispatch", *argv])
    cli.main()


def test_cli_writes_audited_outputs_with_a_loss_factor_file(
    tmp_path, monkeypatch, cached_prices, capsys
) -> None:
    output = tmp_path / "outputs"
    run_cli(
        monkeypatch,
        [
            "--start", START,
            "--end", END,
            "--region", "SA1",
            "--connection-point", "Hornsdale 275 kV",
            "--loss-factor-file", MLF_FILE,
            "--degradation-cost", "25",
            "--cache", str(cached_prices),
            "--output", str(output),
        ],
    )
    assert "Wrote" in capsys.readouterr().out
    summary = pd.read_csv(output / "results.csv")
    assert len(summary) == 1
    row = summary.iloc[0]
    # The run is only written once every gate has passed, so the persisted row
    # is itself the evidence that the audit ran.
    assert row["solver_all_optimal"]
    assert row["completed_cycle_break_even_violation_count"] == 0
    assert row["price_missing_interval_count"] == 0
    assert row["max_simultaneous_dispatch_fraction"] == 0.0
    assert row["total_revenue"] > 0
    assert row["loss_factor_source"].startswith(MLF_FILE)


def test_cli_accepts_scalar_loss_factors(tmp_path, monkeypatch, cached_prices) -> None:
    output = tmp_path / "outputs"
    run_cli(
        monkeypatch,
        [
            "--start", START,
            "--end", END,
            "--region", "SA1",
            "--connection-point", "Hornsdale 275 kV",
            "--generation-mlf", "0.9652",
            "--load-mlf", "0.9681",
            "--mlf-source", "test fixture",
            "--cache", str(cached_prices),
            "--output", str(output),
        ],
    )
    assert (output / "results.csv").exists()


def test_scalar_and_file_loss_factor_modes_are_mutually_exclusive() -> None:
    args = cli.build_parser().parse_args(
        [
            "--start", START,
            "--end", END,
            "--loss-factor-file", MLF_FILE,
            "--generation-mlf", "0.9652",
        ]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        cli.asset_from_args(args)


def test_scalar_mode_demands_complete_provenance() -> None:
    args = cli.build_parser().parse_args(
        ["--start", START, "--end", END, "--generation-mlf", "0.9652"]
    )
    with pytest.raises(ValueError, match="Scalar MLF mode requires"):
        cli.asset_from_args(args)


def test_cli_refuses_to_write_a_run_that_fails_its_audit(
    tmp_path, monkeypatch, cached_prices
) -> None:
    """A failed gate must stop the run before any output file appears."""
    output = tmp_path / "outputs"

    def failing_audit(*args, **kwargs):
        raise ValueError("Run audit failed: injected")

    monkeypatch.setattr(cli, "assert_audit_passes", failing_audit)
    with pytest.raises(ValueError, match="injected"):
        run_cli(
            monkeypatch,
            [
                "--start", START,
                "--end", END,
                "--region", "SA1",
                "--connection-point", "Hornsdale 275 kV",
                "--loss-factor-file", MLF_FILE,
                "--cache", str(cached_prices),
                "--output", str(output),
            ],
        )
    assert not output.exists()
