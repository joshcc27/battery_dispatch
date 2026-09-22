"""Validated, provenance-preserving marginal-loss-factor schedules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd

from .config import LossFactors, LossFactorTable


REQUIRED_COLUMNS = {
    "region",
    "financial_year",
    "connection_point",
    "generation_duid",
    "load_duid",
    "generation_loss_factor",
    "load_loss_factor",
    "source_title",
    "source_url",
    "source_page",
}


@dataclass(frozen=True)
class LossFactorSchedule:
    region: str
    connection_point: str
    generation_duid: str
    load_duid: str
    table: LossFactorTable
    source_file: Path
    source_sha256: str

    @property
    def provenance(self) -> str:
        return f"{self.source_file.as_posix()} sha256:{self.source_sha256}"


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_loss_factor_schedule(
    path: str | Path,
    *,
    region: str,
    connection_point: str | None = None,
) -> LossFactorSchedule:
    """Load one connection-point schedule from a provenance-complete CSV."""
    source = Path(path)
    frame = pd.read_csv(source, dtype=str)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"Loss-factor file is missing columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Loss-factor file contains blank required values")
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Loss-factor column {column} contains blank values")

    selected = frame.loc[frame["region"] == region].copy()
    if connection_point is not None:
        selected = selected.loc[selected["connection_point"] == connection_point]
    if selected.empty:
        raise ValueError(
            f"No loss factors found for region={region!r}, connection_point={connection_point!r}"
        )
    connection_points = selected["connection_point"].unique().tolist()
    if len(connection_points) != 1:
        raise ValueError(
            "Loss-factor file contains multiple connection points; select one explicitly"
        )
    if selected["financial_year"].duplicated().any():
        raise ValueError("Loss-factor file has duplicate financial-year rows")
    if selected["generation_duid"].nunique() != 1 or selected["load_duid"].nunique() != 1:
        raise ValueError("DUIDs must be consistent across a connection-point schedule")
    if not selected["source_url"].str.startswith("https://").all():
        raise ValueError("Every loss-factor source_url must be an HTTPS URL")

    generation = pd.to_numeric(selected["generation_loss_factor"], errors="raise")
    load = pd.to_numeric(selected["load_loss_factor"], errors="raise")
    if ((generation <= 0) | (load <= 0)).any():
        raise ValueError("Loss factors must be positive")

    by_fy: dict[str, LossFactors] = {}
    for index, row in selected.iterrows():
        provenance = (
            f"{row['source_title']}, p.{row['source_page']}, {row['source_url']}"
        )
        by_fy[row["financial_year"]] = LossFactors(
            generation=float(generation.loc[index]),
            load=float(load.loc[index]),
            source=provenance,
        )
    return LossFactorSchedule(
        region=region,
        connection_point=connection_points[0],
        generation_duid=selected["generation_duid"].iloc[0],
        load_duid=selected["load_duid"].iloc[0],
        table=LossFactorTable(by_fy),
        source_file=source,
        source_sha256=file_sha256(source),
    )
