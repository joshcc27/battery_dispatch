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

The optional context table `DISPATCHREGIONSUM` is validated separately and is
never used to settle model revenue. It requires `TOTALDEMAND` and
`AVAILABLEGENERATION` in addition to the timestamp, region and intervention
fields.

## Cache and provenance

The cache stores one parquet file per table, region and physical calendar
month under `data/cache/partitions_v2`. A monthly partition contains interval
ends in `(month_start, next_month_start]`; this convention prevents midnight
month-boundary gaps. Missing partitions are downloaded through NEMOSIS and
recorded in `data/cache/manifest.jsonl`. Existing partitions are immutable
cache hits and require no network access.

The committed FY2022-23 reference manifest is
[`data/manifests/sa1_fy2022-23.json`](../data/manifests/sa1_fy2022-23.json). It
records source, exact boundaries, partition row counts, quality checks and a
canonical checksum without committing the large market-data files.

## Loss-factor contract

`config/loss_factors.csv` is a version-controlled schedule keyed by region,
connection point and financial year. Every row must include separate generation
and load DUIDs, separate positive MLFs, an HTTPS source URL, title and page.
Duplicate financial years, inconsistent DUIDs or missing provenance fail at
load time. A run hash includes the selected asset configuration and the summary
records the loss-factor file SHA-256.
