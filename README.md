# Battery dispatch

An energy-only, five-minute NEM battery backtest. For a meter-side dispatch
trace it computes:

```text
sum(dt * RRP * (generation_MLF * discharge - load_MLF * charge))
- degradation_cost_per_MWh * discharged_MWh
```

The reference physical asset is 100 MW / 200 MWh in SA1, with 87% round-trip
efficiency and a configurable 5%–95% SOC envelope. Round-trip efficiency is
split symmetrically between charging and discharging. SOC is stored energy;
charge and discharge are measured at the connection point.
Annual runs start at minimum SOC so opening inventory is not counted as free
arbitrage revenue.

## Scope decision: energy only

FCAS, caps, contracts, network support revenue, and compensation are excluded.
Accordingly, FCAS is exactly **0% of modelled revenue**. That is a model scope
statement, not an estimate of FCAS's share of a real battery's realised
revenue. A realised FCAS share cannot be stated honestly without naming the
asset and period; it must be attached to each external AEMO Quarterly Energy
Dynamics comparison. The annual energy result should be meaningfully below a
comparable real battery's total energy-plus-FCAS revenue.

June 2022 administered pricing and market suspension rows are retained and
flagged as `june_2022_event`. Report runs both including and excluding that flag.

## Mechanics

- NEM timestamps are parsed as fixed UTC+10 (`AEST`), never a daylight-saving
  timezone. `SETTLEMENTDATE` is the interval end.
- `INTERVENTION = 0` is selected before key uniqueness is checked.
- RRP is checked against the dated AEMC market price cap and the −$1,000/MWh
  market floor. An unknown financial year fails instead of reusing a stale cap.
- Generation and load MLFs are explicit, separately sourced inputs. They are not
  assumed reciprocal.
  A `LossFactorTable` can key them by financial year, and settlement resolves
  the applicable pair for every interval-ending timestamp. The optimiser and
  settlement share the same MLF-adjusted marginal cashflow coefficients.
- A binary charge mode prevents simultaneous charging and discharging,
  including under negative prices.
- No ramp constraint is imposed: a grid battery can reverse within a five-minute
  dispatch interval, making such a constraint non-binding at this resolution.
- The rolling policy solves 48 hours, implements 24, carries SOC, and discards
  the look-ahead tail. It does not impose an artificial terminal SOC.

## Layout

```text
src/config.py       asset parameters, annual MLF types, dated price caps
src/data.py         NEMOSIS download, monthly parquet cache, ingest assertions
src/battery.py      SOC physics and feasibility checks
src/settlement.py   the sole authoritative revenue calculation
src/optimiser.py    linopy/HiGHS per-window MILP
src/horizon.py      rolling-horizon driver
src/metrics.py      run summary and durable CSV output
src/plots.py        plots generated from interval CSVs
src/study.py        region/FY/degradation sweep orchestration
tests/              hand-worked and invariant tests
```

A dispatch trace is always a dataframe with `settlementdate`, `charge_mw`,
`discharge_mw`, and end-of-interval `soc_mwh`. Optimised and future simulated
policies therefore share settlement and metrics code.

## Install and test

```powershell
uv sync --extra dev
uv run pytest
```

The core API is:

```python
from battery_dispatch import BatteryConfig, solve_window

asset = BatteryConfig(
    generation_loss_factor=0.98,  # use the published asset/FY value
    load_loss_factor=1.01,
)
window = solve_window(prices, asset, soc_initial=asset.initial_soc_mwh)
```

To download one period, run the rolling policy, settle it, and write outputs:

```powershell
uv run battery-dispatch `
  --start "2022-07-01 00:00" `
  --end "2023-07-01 00:00" `
  --region SA1 `
  --generation-mlf 0.98 `
  --load-mlf 1.01 `
  --mlf-source "AEMO MLF report, asset connection point, FY2022-23"
```

`--start` and `--end` are physical period boundaries. Because AEMO records are
interval-ending, the example loads timestamps `(2022-07-01 00:00,
2023-07-01 00:00]`: the first row is 00:05 and the last is 00:00 at the next
financial-year boundary.

NEMOSIS is called only for missing `(table, region, month)` partitions. Curated
files live under `data/cache/partitions`; `manifest.jsonl` records fetch time and
row count. A cache hit reads parquet without invoking NEMOSIS or touching the
network.

One interval-level `dispatch_<config hash>.csv` is written per run. Summary rows
are upserted in `results.csv`, keyed by that hash. Plot helpers read those files,
not in-memory model state.
The summary includes June-2022 event revenue and total revenue excluding that
event, plus both period and annualised $/MW and cycle metrics.

`run_parameter_sweep(...)` accepts any requested region list, annual MLF tables,
and degradation-cost grid, then executes and writes every region/FY/cost run.
The repository does not invent asset MLFs: published values for the connection
point under study must be supplied explicitly.

## Validation suite

The tests cover a hand-computed six-interval case, energy balance, flat prices,
monotonicity in power/energy/degradation, negative-price exclusivity, SOC
bounds, loss-factor break-even economics, timestamp/intervention handling,
strict duplicate/gap/NaN/price-bound rejection, and the no-network cache-hit
contract.

The next validation step requiring external data is to plot one cached
region-month against the corresponding AEMO chart, then compare annual $/MW
energy revenue with named SA battery results in AEMO Quarterly Energy Dynamics.
