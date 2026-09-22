# Battery dispatch implementation handover

Status captured 22 September 2026. Branch: `main`. HEAD: `94708f1` (`Implement
NEM battery energy-arbitrage backtest:`). All work described below is currently
**uncommitted**. No background model process is left running: the reference
scenario run was deliberately stopped part-way and must be restarted from
scratch. See [Outstanding work](#outstanding-work).

## Objective and scope

The requested work was to implement every phase of the agreed realism and
validation plan except phase 5. Phase 5 (the publication/audit bundle) has not
been implemented. The reference remains a hypothetical 100 MW / 200 MWh,
87%-round-trip-efficient battery at the Hornsdale 275 kV connection point in
SA1; it is not the actual Hornsdale Power Reserve.

The model is wholesale-energy-only. FCAS, contracts, caps, network support,
outages, bidding behavior and market impact are explicitly excluded.

## Phase status

| Phase | Status | Main evidence |
|---|---|---|
| 1. Reproducible baseline | Complete before this session | Baseline commit `94708f1` |
| 2. Formal input contract | Implemented | `docs/DATA_CONTRACT.md`, FY manifest and strict ingest checks |
| 3. Structured MLF ingestion | Implemented | `src/mlf.py`, `config/loss_factors.csv`, tests |
| 4. Full financial-year run | Completed | 105,120 FY2022-23 intervals and 365 optimal windows |
| 5. Publication/audit bundle | Excluded by request | Not implemented |
| 6. Complete validation | Complete | Runtime audits, cycle audit, fixture and mutations; all gates rerun 22 Sep 2026 |
| 7. Endpoint sensitivity | Implemented; results pending rerun | Minimum, midpoint, A$100/MWh and A$400/MWh look-ahead salvage cases |
| 8. Degradation sensitivity | Diagnosed and unblocked; results pending rerun | A$0 case runs under a declared solver tolerance; see `docs/VALIDATION.md` |
| 9. External benchmark | Implemented | Scope-aware benchmark loader/comparison and source table |
| 10. CI | Implemented | CI and manual annual-backtest GitHub workflows |

## Implemented changes

### Data and provenance

- `src/data.py` now treats user boundaries as physical `[start, end)` and AEMO
  timestamps as interval-ending `(start, end]` in fixed UTC+10.
- Monthly cache partitions use `(month_start, next_month_start]` and live under
  `partitions_v2`. This fixed 12 missing midnight observations found on the
  first annual run.
- Obsolete `keep_zip=True` was removed because the installed NEMOSIS version
  forwarded it to pandas/pyarrow and failed.
- `data/manifests/sa1_fy2022-23.json` records exact boundaries, all 12 monthly
  row counts, data-quality counts and canonical price checksum
  `3beacbd0bd39b09bd44d936f11fa3811ab44d2304fae1b4c4fd246e4fdd8b30a`.
- `src/mlf.py` validates a provenance-complete annual MLF schedule. The
  reference CSV uses HPRG1 generation MLF 0.9652 and HPRL1 load MLF 0.9681 for
  FY2022-23, sourced from AEMO's MLF report, p.34. The input file SHA-256 is
  `70e267d9f0d93a27898166d7c71a99797f3ea14972bc1cb9a627da1ae77b1f98`.

### Optimization, settlement and reporting

- Rolling-window output now includes window ID, solver status/termination,
  solve time, relative MIP gap, window boundaries and opening SOC.
- HiGHS relative MIP gap is explicitly fixed at `1e-6` and audited.
- Starting SOC tolerates and clips only `1e-8`-scale solver roundoff; material
  envelope violations still fail.
- A non-settled A$0.0001/MWh throughput tie-breaker removes economically
  indifferent cycling. `objective_value` and all reported revenue exclude this
  numerical preference.
- Endpoint sensitivity supports initial SOC and a look-ahead salvage value.
  Inventory change and cash-plus-inventory value are reported separately.
- Metrics now distinguish gross energy revenue from net-of-degradation revenue
  and record power/energy capacity.
- `src/validation.py` independently checks price identity, SOC balance,
  exclusivity, limits, solver outcomes and weighted-inventory completed-cycle
  break-even. The CLI refuses to write a failed run.

### Solver tolerance for the A$0/MWh scenario

- The A$0 blocker is diagnosed, not worked around. The exclusivity binary can
  only bind where simultaneous charge and discharge pays, which a perturbation
  argument shows requires `p < -6.776 * deg` at the FY2022-23 Hornsdale loss
  factors. That is 0.55% of intervals at A$25/MWh but 23% at A$0/MWh, across
  317 of 365 days, with every one of the 288 intervals negative-priced on 18
  and 20 November 2022. The derivation and measured table are in
  `docs/VALIDATION.md`; `scripts/analyse_exclusivity_binding.py` regenerates
  the numbers from cache.
- `SolverPolicy` in `src/optimiser.py` carries a per-scenario `mip_rel_gap` and
  optional `time_limit_seconds`, threaded through `run_rolling_horizon`,
  `run_sensitivities` and the scenario table. The strict `1e-6` policy remains
  the default for every other scenario and for the CLI.
- A window stopped at a time limit is accepted **only** when the policy asked
  for a time limit. Any other non-optimal termination still fails the run, and
  `assert_audit_passes` takes an explicit `mip_gap_tolerance`; the looser gap
  is never implied.
- Every window records `solver_objective_value`, `solver_dual_bound` and
  `solver_absolute_gap`. Runs report `max_solver_absolute_gap_aud` and
  `total_solver_absolute_gap_aud`, so a loosened scenario states the optimality
  gap it achieved instead of claiming exactness. The summed gap bounds forgone
  window objective, not annual settled revenue.
- `run_rolling_horizon` takes an optional `progress` callback, and the scenario
  runner reports every Nth window plus every time-limited window with solve
  time, achieved gap and elapsed total. A full-year loose-policy run is
  otherwise silent for minutes at a time and indistinguishable from a hang.

### Automation and documentation

- `scripts/run_fixture.py` provides a deterministic end-to-end CI run.
- `scripts/run_mutation_checks.py` verifies that nine critical errors make tests
  fail: MLF removal, efficiency removal, binary relaxation, settlement sign,
  degradation sign, intervention filtering, interval convention, SOC carry and
  look-ahead-tail commitment.
- `scripts/run_reference_sensitivities.py` supports named scenarios, incremental
  checkpointing and isolated output paths.
- `.github/workflows/ci.yml` runs tests/coverage, compile checks, the fixture and
  mutations. `.github/workflows/annual-backtest.yml` is a manual data-backed
  annual run using the committed MLF schedule.
- README and `docs/DATA_CONTRACT.md`, `docs/VALIDATION.md` and
  `docs/BENCHMARKS.md` describe the model, contracts and caveats.

## Verified annual results

The strict-gap headline result is stored in the ignored local file
`outputs/sensitivities/headline.csv`:

| Metric | Strict FY2022-23 result |
|---|---:|
| Gross energy revenue | A$42,319,996.94 |
| Degradation cost at A$25/MWh | A$3,931,979.48 |
| Net revenue | A$38,388,017.45 |
| Net revenue / MW-year | A$384,143.11 |
| Equivalent full cycles | 786.396 |
| Initial / terminal SOC | 10.0 / 10.0 MWh |
| Optimal windows | 365 / 365 |
| Maximum observed relative MIP gap | `3.31e-10` |
| Completed-cycle break-even violations | 0 |

Independent annual checks found 105,120 unique timestamps, no missing or extra
intervals, no simultaneous charge/discharge, SOC inside 10–190 MWh and maximum
interval energy-balance error below `7e-11` MWh.

Completed strict sensitivity results:

| Scenario | Net revenue | EFC | Terminal SOC | Max MIP gap |
|---|---:|---:|---:|---:|
| Headline: A$25/MWh, minimum initial SOC | A$38.388m | 786.396 | 10 MWh | `3.31e-10` |
| A$50/MWh, minimum initial SOC | A$35.010m | 591.100 | 10 MWh | `9.08e-7` |
| A$25/MWh, midpoint initial SOC | A$38.421m | 786.396 | 10 MWh | `3.31e-10` |
| A$25/MWh, A$100/MWh look-ahead salvage | A$38.388m | 786.396 | 10 MWh | `2.19e-16` |

The midpoint case adds about A$32.7k because 90 MWh of unpriced opening stored
energy is depleted.

The A$100/MWh salvage case is identical to the headline to the cent, which is
the expected result rather than a sign the parameter was ignored: it is a
genuinely different solve, with its own `config_hash` and a different maximum
MIP gap, but terminal SOC sits on the 10 MWh floor in every scenario, so the
salvage term never binds. A$100/MWh is also well below the A$282/MWh realised
spread, so it tests nothing. `lookahead_salvage_400` was added to the scenario
table to put a salvage value above the realised spread, where the term can
actually change the committed dispatch. It has not been run yet.

These four rows predate the `SolverPolicy` work. Their dispatch is unaffected,
because the strict policy passes HiGHS exactly the options the previous code
passed, but their `config_hash` values are now stale (the hash includes the
solver policy) and they lack the new solver-bound columns. They are kept only
as a cross-check for the rerun and must not be copied into
`analysis/reference_sensitivities.csv`.

## External benchmark status

`config/benchmarks.csv` contains two page-sourced AEMO observations:

- SA battery fleet, Q1 2024: A$8.7m energy plus A$11.7m FCAS (A$20.4m total),
  with no stored capacity denominator; QED Q1 2024, p.39.
- NEM battery fleet, Q4 2024: A$48.1m energy plus A$21.3m FCAS, with 1,087 MW
  average availability; QED Q4 2024, p.38.

Neither is period-, geography- or asset-scope-aligned with the FY2022-23
reference. `src/benchmark.py` therefore returns explicit alignment/scope flags,
does not invent a Q1 per-MW value, and labels the Q4 denominator as average
availability rather than installed capacity.

## Tests run

All four gates were rerun after the `SolverPolicy` changes and all pass:

- `uv run pytest --basetemp .test-tmp` - `40 passed in 11.51s`.
- `uv run python scripts/run_fixture.py` - deterministic run reproduced, price
  SHA-256 `a07b7826778ff3b0f6018df2efb59aabf68986e2e25a1c3d09bed1f74a47b9c2`,
  zero audit violations.
- `uv run python scripts/run_mutation_checks.py` - all nine mutations detected.
- `uv run python -m compileall -q src tests scripts` - clean.

The suite grew from 33 to 34 tests during implementation, and to 40 with
`tests/test_solver_policy.py`, which covers policy validation, the optimality
bound HiGHS reports, the binding-price threshold against the settled
coefficients, and the refusal to accept a time-limited window unless the policy
asked for one.

Some local pytest invocations showed only sandbox temp/cache permission setup
errors. Use a direct workspace base temp rather than a missing nested path.

## Outstanding work

Completed since this handover was first written: the README result table was
corrected to the strict headline run (gross A$42.320m, degradation A$3.932m,
gross A$423,490/MW-year, net A$384,143/MW-year, EFC 786.4 - five of the eight
figures were stale or truncated); all four gates were rerun green; and the
A$0/MWh scenario was diagnosed and unblocked under a declared solver tolerance.

1. **Rerun the reference scenarios.** This is the only thing blocking the rest.

   ```powershell
   uv run python -u scripts/run_reference_sensitivities.py `
     --output analysis/reference_sensitivities.csv --progress-every 20
   ```

   The run is checkpointed per scenario and skips completed rows, so an
   interrupted run resumes by reissuing the same command. It produces all six
   scenarios: `degradation_0`, `headline`, `degradation_50`,
   `midpoint_initial_soc`, `lookahead_salvage_100` and `lookahead_salvage_400`.

   Evidence from the stopped attempt, which is the best available runtime
   estimate: `degradation_0` reached window 300 of 365 in 308 seconds with
   **zero time-limited windows**. The hardest day of the year, 18 November
   2022, solved in 1.6 seconds. The `1e-3` relative gap alone is sufficient and
   the 120-second backstop was never reached, so expect roughly 6 to 8 minutes
   for A$0 and a similar order for each strict scenario. An earlier attempt was
   abandoned at 40 minutes only because it had no progress output and was
   assumed to be hung; it was not.

   No scenario completed before the run was stopped, so
   `analysis/reference_sensitivities.csv` does not exist yet and no A$0 result
   has ever been accepted.

2. **Fill the tables in `docs/RESULTS.md`.** The prose, scope statement and
   caveats are written; the `<!-- TABLES -->` marker is where the per-scenario
   table goes, built from the CSV that step 1 produces. Report the A$0 row with
   its achieved `max_solver_mip_gap` and `total_solver_absolute_gap_aud`
   alongside revenue, and state that it was solved under the documented
   tolerance rather than to `1e-6`.

3. **Refresh the figures that quote a run.** After step 1, update this
   document's "Verified annual results" tables and the README result table from
   `analysis/reference_sensitivities.csv`, and record each scenario's
   `wall_clock_seconds` so the next person has a real runtime baseline. Confirm
   the strict rows reproduce the existing headline figures; a difference there
   would mean the `SolverPolicy` refactor changed strict behaviour and must be
   investigated before anything is committed.

4. Review `git diff`, confirm generated `data/cache/` and `outputs/` remain
   ignored, and commit the implementation. No commit has been created for this
   session. Consider two commits: the implementation, then the A$0 resolution
   and results.

## Useful reproduction commands

Headline annual CLI run:

```powershell
uv run battery-dispatch `
  --start "2022-07-01 00:00" `
  --end "2023-07-01 00:00" `
  --region SA1 `
  --connection-point "Hornsdale 275 kV" `
  --loss-factor-file config/loss_factors.csv `
  --degradation-cost 25 `
  --cache data/cache `
  --output outputs
```

Run one sensitivity without output collisions:

```powershell
uv run python scripts/run_reference_sensitivities.py `
  --scenario degradation_50 `
  --output outputs/sensitivities/degradation_50.csv
```

Valid scenario names are `degradation_0`, `headline`, `degradation_50`,
`midpoint_initial_soc`, `lookahead_salvage_100` and `lookahead_salvage_400`.

Measure where the exclusivity binary binds, which is the evidence behind the
A$0 scenario's declared tolerance:

```powershell
uv run python scripts/analyse_exclusivity_binding.py
```

## Source links

- AEMO FY2022-23 MLF report:
  <https://www.aemo.com.au/-/media/files/electricity/nem/security_and_reliability/loss_factors_and_regional_boundaries/2022-23/marginal-loss-factors-for-the-2022-23-financial-year.pdf>
- AEMO QED Q1 2024:
  <https://www.aemo.com.au/-/media/files/major-publications/qed/2024/qed-q1-2024.pdf>
- AEMO QED Q4 2024:
  <https://www.aemo.com.au/-/media/files/major-publications/qed/2024/qed-q4-2024.pdf>
