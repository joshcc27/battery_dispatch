"""Plotting helpers that consume durable CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dispatch_csv(csv_path: str | Path, output_path: str | Path) -> Path:
    """Plot RRP and SOC from an interval-level run CSV."""
    data = pd.read_csv(csv_path, parse_dates=["settlementdate"])
    required = {"settlementdate", "rrp", "soc_mwh"}
    if missing := required.difference(data.columns):
        raise ValueError(f"Dispatch CSV is missing columns: {sorted(missing)}")
    figure, price_axis = plt.subplots(figsize=(12, 6))
    soc_axis = price_axis.twinx()
    price_axis.plot(data["settlementdate"], data["rrp"], color="tab:blue", linewidth=0.8)
    soc_axis.plot(data["settlementdate"], data["soc_mwh"], color="tab:orange", linewidth=1.0)
    price_axis.set_ylabel("RRP ($/MWh)", color="tab:blue")
    soc_axis.set_ylabel("SOC (MWh)", color="tab:orange")
    price_axis.set_xlabel("Settlement interval end (AEST)")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
