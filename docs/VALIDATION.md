# Validation and acceptance gates

The reference run is accepted only if both input and dispatch audits pass.
Failures raise an exception before output is written.

## Runtime gates

- exact expected interval count, unique keys, no gaps or extra timestamps;
- dated RRP floor/cap checks and non-intervention filtering;
- every rolling window reports solver status `ok` and termination `optimal`,
  except under an explicitly declared time-limited policy (see
  [Solver tolerance](#solver-tolerance-and-where-exclusivity-binds));
- every window's relative MIP gap is at most the scenario's declared tolerance,
  `1e-6` for every scenario except A$0/MWh degradation;
- aggregate and interval-by-interval SOC identities within `1e-6` MWh;
- charge/discharge exclusivity, power limits and SOC bounds;
- independently recomputed settlement components;
- no completed charged-inventory segment discharged below its loss-adjusted
  acquisition and degradation cost.

The optimizer includes a non-settled A$0.0001/MWh throughput tie-breaker. This
is a numerical secondary preference that removes economically indifferent
cycling and zero-cost MIP branches; all reported revenue is recomputed without
it by the authoritative settlement function.

For the FY2022-23 SA1 reference case, 105,120 intervals and 365 daily committed
windows passed. The maximum independently recomputed interval energy-balance
error was below `1e-12` MWh, and no simultaneous charge/discharge row or
completed-cycle break-even violation was found.

## Solver tolerance and where exclusivity binds

Every reported scenario except A$0/MWh degradation is solved to a `1e-6`
relative MIP gap. The A$0 case is not tractable at that tolerance: a single
48-hour window over 20 November 2022 runs past 900 seconds without closing the
gap, and the original full-year attempt consumed more than 9,700 CPU seconds
before being abandoned. The cause is structural, not a solver defect.

The model carries one binary per interval forbidding simultaneous charge and
discharge. Take any solution with both charge `c` and discharge `d` positive in
one interval and shift it by `-a` on charge and `-a*eta_c*eta_d` on discharge:
the stored-energy change, and so the entire SOC path, is unchanged. The
objective moves by `-a * (A_c + A_d * eta_c * eta_d)` for settled per-MW
coefficients `A_c = -dt * L * p` and `A_d = dt * (G * p - deg)`. Simultaneous
operation therefore only pays when

    p < -deg * eta_c * eta_d / (L - G * eta_c * eta_d)

that is, only at prices below a multiple of the degradation cost. With the
FY2022-23 Hornsdale factors (G 0.9652, L 0.9681, round trip 0.87) the multiple
is 6.776, so the binary genuinely binds only below `-6.776 * deg`. Measured
against the SA1 FY2022-23 prices by
[`scripts/analyse_exclusivity_binding.py`](../scripts/analyse_exclusivity_binding.py):

| Degradation cost | Binary binds when | Intervals | Share | Days affected | Worst day |
|---|---:|---:|---:|---:|---:|
| A$0/MWh | p < A$0.00 | 24,173 | 23.00% | 317 / 365 | 288 of 288 |
| A$25/MWh | p < A$-169.42 | 576 | 0.55% | 86 / 365 | 49 of 288 |
| A$50/MWh | p < A$-338.85 | 236 | 0.22% | 60 / 365 | 29 of 288 |

At A$0/MWh the model is paid to burn energy through round-trip losses, so it
wants simultaneous operation at every negative price and the binary binds 42
times more often than at the headline cost. On 18 and 20 November 2022 every
one of the 288 intervals is negative-priced, and the many equivalent
charge/discharge interleavings across a whole day leave a large symmetric
search tree. This is inherent to the formulation.

The A$0 scenario is therefore declared as a `SolverPolicy` with a `1e-3`
relative gap and a 120-second per-window backstop, and it reports the gap it
actually achieved rather than claiming exactness. The tolerance was chosen from
the measured tractability knee on the hardest window: `5e-4` solves in 63
seconds and `7e-4` in 2.9 seconds, while `2e-4` and `3e-4` both exhaust a
180-second limit. Two properties keep the looser scenario honest:

- a time-limited termination is accepted only when the policy asked for a time
  limit; any other non-optimal termination still fails the run;
- every window records `solver_dual_bound` and `solver_absolute_gap`, and the
  run reports `max_solver_mip_gap`, `max_solver_absolute_gap_aud` and
  `total_solver_absolute_gap_aud`.

The summed absolute gap bounds how much window objective the dispatch decisions
left on the table. It is not a bound on reported annual revenue, which is the
exact settlement of the dispatch actually implemented, and it says nothing about
the gap between rolling-horizon dispatch and annual perfect foresight.

All other gates, including charge/discharge exclusivity, apply unchanged to the
A$0 run. The throughput tie-breaker is strictly positive on both legs, so a
solution carrying simultaneous operation is strictly worse wherever the binary
does not bind, and the exclusivity audit still passes at the looser tolerance.

## Automated tests

`pytest` covers hand-worked settlement, loss-factor economics, SOC physics,
rolling-window carry-forward, timestamp boundaries, cache behavior, MLF input
validation, benchmark scope handling and sensitivity behavior. The deterministic
CI fixture exercises fetch-independent optimization, settlement, metrics and
audits end to end.

`scripts/run_mutation_checks.py` deliberately introduces nine high-risk errors:
loss-factor removal, charge-efficiency removal, binary relaxation, settlement
sign reversal, degradation sign reversal, wrong intervention selection, wrong
interval-ending convention, SOC reset and committing the look-ahead tail. Each
mutation must make its targeted test fail.

CI runs tests with coverage, byte-compiles source/tests/scripts, executes the
deterministic fixture and runs all mutation checks. The separate manual annual
workflow downloads AEMO data, uses the committed MLF schedule and uploads the
large run outputs as an artifact.
