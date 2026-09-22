"""Fetch the reference month and check it against the committed manifest.

The manifest records the identity of the input data, not a description of it.
A change in AEMO's archive, in the ingest contract, or in the intervention or
interval-ending conventions moves the checksum and fails here, before any
result is computed from the changed data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from battery_dispatch.data import MarketDataCache
from battery_dispatch.validation import audit_prices

MANIFEST = Path("data/manifests/sa1_2022-11.json")
REGION = "SA1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.getLogger("nemosis").setLevel(logging.WARNING)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    period = manifest["period"]
    start = period["physical_start_inclusive"]
    end = period["physical_end_exclusive"]

    prices = MarketDataCache(args.cache).fetch_prices(start, end, [REGION])
    audit = audit_prices(prices, expected_start=start, expected_end=end)
    expected = manifest["validation"]

    fields = [
        ("row_count", "price_row_count"),
        ("duplicate_key_count", "price_duplicate_count"),
        ("missing_interval_count", "price_missing_interval_count"),
        ("unexpected_interval_count", "price_unexpected_interval_count"),
        ("rrp_min_aud_per_mwh", "rrp_min"),
        ("rrp_max_aud_per_mwh", "rrp_max"),
        ("canonical_price_sha256", "price_sha256"),
    ]
    checks = [(name, audit[key], expected[name]) for name, key in fields]
    failures = [(name, got, want) for name, got, want in checks if got != want]
    for name, got, want in checks:
        status = "OK " if got == want else "BAD"
        detail = "" if got == want else f" != {want}"
        print(f"{status} {name}: {got}{detail}")
    if failures:
        raise SystemExit(
            "Reference inputs do not match the committed manifest: "
            + ", ".join(name for name, _, _ in failures)
        )
    print(f"\nReference inputs match {args.manifest}")


if __name__ == "__main__":
    main()
