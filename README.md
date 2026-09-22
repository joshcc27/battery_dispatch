# Battery dispatch

An auditable, energy-only, five-minute NEM battery backtest. The reference case
is a **hypothetical** 100 MW / 200 MWh battery at the Hornsdale 275 kV
connection point in SA1; it is not a model of the actual Hornsdale Power
Reserve.

For a meter-side dispatch trace, cash revenue is

```text
sum(dt * RRP * (generation_MLF * discharge - load_MLF * charge))
- degradation_cost_per_discharged_MWh * discharged_MWh
```

The asset has 87% round-trip efficiency, split symmetrically between charge and
discharge, and a 5%–95% SOC envelope. SOC is stored energy; power is measured at
the connection point. The headline case starts at minimum SOC so unpriced
opening inventory is not counted as arbitrage revenue.

## Verified FY2022-23 reference result

The full SA1 financial year contains 105,120 unique five-minute observations
from `(2022-07-01 00:00, 2023-07-01 00:00]` AEST. The headline degradation cost
is an illustrative A$25/discharged MWh.

| Metric | Result |
|---|---:|
| Gross wholesale-energy revenue | A$42.320m |
| Illustrative degradation cost | A$3.932m |
| Net model revenue | A$38.388m |
| Gross energy revenue / MW-year | A$423,490 |
| Net revenue / MW-year | A$384,143 |
| Equivalent full cycles | 786.4 |
| Terminal SOC | 10.0 MWh |
| Optimal rolling windows | 365 / 365 |

All price, solver, energy-balance, exclusivity and completed-cycle audit gates
passed. The canonical price checksum and monthly counts are committed in
[`data/manifests/sa1_fy2022-23.json`](data/manifests/sa1_fy2022-23.json).
Large cache and dispatch files remain intentionally untracked.

## Scope

The model includes wholesale-energy arbitrage only. FCAS, caps, contracts,
network support, compensation, availability outages, bidding behavior and
market impact are excluded. FCAS is therefore exactly 0% of **modelled**
revenue, not an estimate of its share for a real battery. Published fleet
results are retained only as scope-aware contextual checks; see
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

No ramp constraint is imposed because a grid battery can reverse within a
five-minute interval. A binary mode prevents simultaneous charging and
discharging, including at negative prices. A non-settled A$0.0001/MWh
throughput tie-breaker selects the least-cycling schedule among economically
equivalent optima. The rolling policy optimizes 48 hours, commits 24 hours,
carries SOC, and discards the look-ahead tail.

## Reproduce

Install and run the automated gates:

```powershell
uv sync --extra dev
uv run pytest --basetemp .test-tmp
uv run python scripts/run_fixture.py
uv run python scripts/run_mutation_checks.py
```

Run the reference financial year:

```powershell
uv run battery-dispatch `
  --start "2022-07-01 00:00" `
  --end "2023-07-01 00:00" `
  --region SA1 `
  --connection-point "Hornsdale 275 kV" `
  --loss-factor-file config/loss_factors.csv `
  --degradation-cost 25
```

Run the declared endpoint/degradation sensitivity grid:

```powershell
uv run python scripts/run_reference_sensitivities.py
```

The price downloader calls NEMOSIS only for missing physical-calendar-month
partitions. Outputs are interval CSVs plus a config-keyed summary CSV. Inputs
fail closed on gaps, duplicates, intervention rows, unknown dated price caps or
incomplete MLF provenance.

## Repository map

```text
config/              reference asset, MLF schedule, external benchmarks
data/manifests/       committed data identities and quality evidence
docs/                 data contract, validation, results and benchmark policy
src/data.py           NEMOSIS download, partition cache, ingest checks
src/mlf.py            provenance-preserving annual MLF ingestion
src/battery.py        SOC physics and feasibility checks
src/settlement.py     authoritative loss-adjusted settlement calculation
src/optimiser.py      linopy/HiGHS per-window MILP
src/horizon.py        rolling-horizon policy and solver trace
src/metrics.py        run summary and durable output
src/validation.py     independent run and cycle audits
src/sensitivity.py    endpoint/degradation scenarios
src/benchmark.py      period/scope-aware external comparisons
scripts/              deterministic fixture, mutations and sensitivity runner
tests/                hand-worked, invariant and contract tests
```

Detailed contracts and evidence are in [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)
and [`docs/VALIDATION.md`](docs/VALIDATION.md). The repository does not include
the phase-5 publication/audit bundle, as requested.
