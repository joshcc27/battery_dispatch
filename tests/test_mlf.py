from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from battery_dispatch.mlf import load_loss_factor_schedule


def write_schedule(path: Path, **changes: str) -> None:
    row = {
        "region": "SA1",
        "financial_year": "2022-23",
        "connection_point": "Hornsdale 275 kV",
        "generation_duid": "HPRG1",
        "load_duid": "HPRL1",
        "generation_loss_factor": "0.9652",
        "load_loss_factor": "0.9681",
        "source_title": "AEMO MLF report",
        "source_url": "https://example.test/aemo.pdf",
        "source_page": "34",
    }
    row.update(changes)
    pd.DataFrame([row]).to_csv(path, index=False)


def test_loads_provenance_complete_schedule(tmp_path: Path) -> None:
    path = tmp_path / "mlf.csv"
    write_schedule(path)
    schedule = load_loss_factor_schedule(
        path, region="SA1", connection_point="Hornsdale 275 kV"
    )
    factors = schedule.table.for_financial_year("2022-23")
    assert factors.generation == pytest.approx(0.9652)
    assert factors.load == pytest.approx(0.9681)
    assert "p.34" in factors.source
    assert len(schedule.source_sha256) == 64


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_url": "http://example.test/aemo.pdf"}, "HTTPS"),
        ({"generation_loss_factor": "0"}, "positive"),
        ({"source_title": ""}, "blank"),
    ],
)
def test_rejects_invalid_schedule(tmp_path: Path, changes: dict[str, str], message: str) -> None:
    path = tmp_path / "mlf.csv"
    write_schedule(path, **changes)
    with pytest.raises(ValueError, match=message):
        load_loss_factor_schedule(path, region="SA1")
