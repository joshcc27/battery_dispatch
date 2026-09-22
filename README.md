# Battery dispatch

An auditable, energy-only, five-minute NEM battery backtest. The reference case
is a **hypothetical** 100 MW / 200 MWh battery at the Hornsdale 275 kV
connection point in SA1; it is not a model of the actual Hornsdale Power
Reserve.

The whole reference result — five scenarios, every audit gate — runs in about
100 seconds on a laptop and is re-run by CI on every push. Nothing in this
repository is a number you have to take on trust.

For a meter-side dispatch trace, cash revenue is

```text
sum(dt * RRP * (generation_MLF * discharge - load_MLF * charge))
- degradation_cost_per_discharged_MWh * discharged_MWh
```

The asset has 87% round-trip efficiency, split symmetrically between charge and
discharge, and a 5%–95% SOC envelope. SOC is stored energy; power is measured at
the connection point. The headline case starts at minimum SOC so unpriced
opening inventory is not counted as arbitrage revenue.

## Reference result: SA1, November 2022

The reference period is one month: `(2022-11-01 00:00, 2022-12-01 00:00]` AEST,
8,640 five-minute observations. November 2022 is the hardest month of FY2022-23
— 38.0% of intervals are negative-priced and 18 and 20 November are negative in
all 288 intervals — so the exclusivity logic and negative-price charging are
exercised rather than assumed.

| Scenario | Degradation | Net revenue | EFC | Terminal SOC | Max MIP gap | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Headline | A$25/MWh | A$3,608,856 | 75.50 | 10 MWh | 0.0 | 7.4s |
| Low degradation | A$10/MWh | A$3,866,584 | 100.90 | 10 MWh | 9.46e-07 | 73.8s |
| High degradation | A$50/MWh | A$3,290,082 | 54.57 | 10 MWh | 0.0 | 6.7s |
| Midpoint initial SOC | A$25/MWh | A$3,609,405 | 75.62 | 10 MWh | 0.0 | 7.2s |
| A$400/MWh salvage | A$25/MWh | A$3,589,229 | 75.24 | 190 MWh | 9.36e-07 | 7.4s |

Every window solved to optimality. All price, solver, energy-balance,
exclusivity and completed-cycle audit gates passed. The canonical price checksum
is committed in [`data/manifests/sa1_2022-11.json`](data/manifests/sa1_2022-11.json)
and re-verified before any result is computed.

Full detail and caveats are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Scope

The model includes wholesale-energy arbitrage only. FCAS, caps, contracts,
network support, compensation, availability outages, bidding behavior and
market impact are excluded. FCAS is therefore exactly 0% of **modelled**
revenue, not an estimate of its share for a real battery. These figures are a
lower bound on what a real battery at this connection point would earn and are
not comparable to published total battery revenue.

One month is not a distribution, and November 2022 was deliberately chosen as
an extreme. No claim is made that it represents a typical month.

The complete list of modelling limitations is
[`docs/RESULTS.md#limitations`](docs/RESULTS.md#limitations). It is maintained
in one place; this section is a summary, not the full set.

A **zero-degradation case is out of scope.** At A$0/MWh the model is paid to
burn energy through round-trip losses, the exclusivity binary binds at 38% of
November intervals instead of 2.7%, and the resulting MILP is not tractable at a
strict optimality gap. The reasoning is derived in
[`docs/VALIDATION.md`](docs/VALIDATION.md); A$10/MWh is included as the
low-degradation bound instead.

No ramp constraint is imposed because a grid battery can reverse within a
five-minute interval. A binary mode prevents simultaneous charging and
discharging, including at negative prices. A non-settled A$0.0001/MWh
throughput tie-breaker selects the least-cycling schedule among economically
equivalent optima. The rolling policy optimizes 48 hours, commits 24 hours,
carries SOC, and discards the look-ahead tail.

## Reproduce

Install and run the offline gates:

```powershell
uv sync --extra dev
uv run pytest --basetemp .test-tmp
uv run python scripts/run_fixture.py
uv run python scripts/run_mutation_checks.py
```

Verify the reference inputs, then run the reference grid:

```powershell
uv run python scripts/verify_reference.py
uv run python scripts/run_reference_sensitivities.py
```

Run any other period or asset through the CLI:

```powershell
uv run battery-dispatch `
  --start "2022-11-01 00:00" `
  --end "2022-12-01 00:00" `
  --region SA1 `
  --connection-point "Hornsdale 275 kV" `
  --loss-factor-file config/loss_factors.csv `
  --degradation-cost 25
```

The price downloader calls NEMOSIS only for missing physical-calendar-month
partitions. Outputs are interval CSVs plus a config-keyed summary CSV. Inputs
fail closed on gaps, duplicates, intervention rows, unknown dated price caps or
incomplete MLF provenance.

## Repository map

```text
config/               reference asset MLF schedule
data/manifests/       committed input identity and quality evidence
docs/                 data contract, validation and results
src/data.py           NEMOSIS download, partition cache, ingest checks
src/mlf.py            provenance-preserving annual MLF ingestion
src/battery.py        SOC physics and feasibility checks
src/settlement.py     authoritative loss-adjusted settlement calculation
src/optimiser.py      linopy/HiGHS per-window MILP and exclusivity reduction
src/horizon.py        rolling-horizon policy and solver trace
src/metrics.py        run summary and durable output
src/validation.py     independent run and cycle audits
src/sensitivity.py    endpoint/degradation scenarios
scripts/              input verification, fixture, mutations, reference grid
tests/                hand-worked, invariant and contract tests
```

Detailed contracts and evidence are in
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) and
[`docs/VALIDATION.md`](docs/VALIDATION.md).
