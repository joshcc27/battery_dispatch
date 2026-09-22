# Input data contract

The model accepts AEMO five-minute `DISPATCHPRICE` data only. A study period is
expressed as physical boundaries `[start, end)`, while AEMO timestamps are
interval-ending, so the required observations are `(start, end]` in fixed
UTC+10 (`AEST`). The timestamp at 1 July 00:00 belongs to the financial year
that has just ended.

## Required price fields

| Field | Contract |
|---|---|
| `SETTLEMENTDATE` | Parseable timestamp; converted to fixed UTC+10 |
| `REGIONID` | Requested NEM region; part of the unique key |
| `RRP` | Numeric AUD/MWh, inside the dated market floor/cap |
| `INTERVENTION` | Numeric; only `0` is retained |

After intervention filtering, `(settlementdate, regionid)` must be unique.
There must be exactly one observation every five minutes for every requested
region, with no nulls, gaps, or timestamps outside the requested range. Unknown
financial years fail because no unverified market cap is carried forward.

## Cache and provenance

The cache stores one parquet file per table, region and physical calendar
month under `data/cache/partitions`. A monthly partition contains interval
ends in `(month_start, next_month_start]`; this convention prevents midnight
month-boundary gaps. Missing partitions are downloaded through NEMOSIS and
recorded in `data/cache/manifest.jsonl`. Existing partitions are immutable
cache hits and require no network access.

The committed reference manifest is
[`data/manifests/sa1_2022-11.json`](../data/manifests/sa1_2022-11.json). It
records source, exact boundaries, partition row counts, quality checks, the
rationale for the period, and a canonical checksum, without committing the
market-data files themselves. `scripts/verify_reference.py` re-derives every one
of those values from a fresh fetch and fails if any differs, so the manifest is
an enforced contract rather than a description.

## Loss-factor contract

`config/loss_factors.csv` is a version-controlled schedule keyed by region,
connection point and financial year. Every row must include separate generation
and load DUIDs, separate positive MLFs, an HTTPS source URL, title and page.
Duplicate financial years, inconsistent DUIDs or missing provenance fail at
load time. A run hash includes the selected asset configuration and the summary
records the loss-factor file SHA-256.

## Output contract

A run writes two files. `dispatch_<config_hash>.csv` is the interval-level
trace: one row per five-minute interval with dispatch, SOC, prices, loss
factors, settlement components and the solver evidence for the window it came
from. `results.csv` holds one row per run, keyed by `config_hash` and upserted,
so re-running an identical configuration replaces its row rather than appending
a duplicate.

That summary row is deliberately wide — currently 67 columns. It is an **audit
record, not a report.** Alongside the headline revenue and cycling figures it
carries every value the acceptance gates were evaluated against: energy-balance
residuals, the exclusivity fraction, solver status, gaps and bounds per run,
completed-cycle evidence, the SOC decile distribution, and the price-audit
counts and checksum. Keeping them on one row means a stored result can be
re-checked without re-running it, and without joining two files. Read
`docs/RESULTS.md` for the figures that are meant to be quoted.
