# Validation and acceptance gates

The reference run is accepted only if both input and dispatch audits pass.
Failures raise an exception before output is written.

## Runtime gates

- exact expected interval count, unique keys, no gaps or extra timestamps;
- dated RRP floor/cap checks and non-intervention filtering;
- every rolling window reports solver status `ok` and termination `optimal`;
- every window's relative MIP gap is at most `1.01e-6`;
- aggregate and interval-by-interval SOC identities within `1e-6` MWh;
- charge/discharge exclusivity, power limits and SOC bounds;
- independently recomputed settlement components;
- no completed charged-inventory segment discharged below its loss-adjusted
  acquisition and degradation cost;
- the result does not exceed the full-period perfect-foresight ceiling by more
  than A$1 plus 1e-6 of the ceiling.

## Perfect-foresight ceiling

`src/ceiling.py` solves the entire period as one LP: same asset, prices, dated
loss factors, starting SOC and terminal valuation, but no rolling horizon, no
exclusivity binary and no throughput tie-breaker. Every rolling dispatch is a
feasible point of that LP, so the LP optimum bounds it from above, and the
reported `horizon_gap_aud` is a proven upper limit on what better look-ahead
could add. The LP has no binaries, so a full financial year solves in about 15
seconds. A run above the ceiling beyond solver tolerance fails the audit,
because it can only mean the two formulations disagree.

Adding exclusivity binaries lazily — only where the ceiling LP actually charges
and discharges at once — was tried and rejected. It reproduced the same revenue
with far fewer binaries (26 against 468 at A$25/MWh) but was slower: at A$10/MWh
the November grid took 1,027s one interval at a time and 343s adding whole
negative-price blocks, against 145s for the dominance mask. Fewer binaries gave
the branch-and-bound a harder search, not an easier one.

A gate whose audit key is absent is treated as a **missing check, not a passing
one**. `assert_audit_passes` asserts every required key is present before
evaluating any of them, so a caller that forgets to merge one of the audit
functions fails loudly instead of silently skipping its gates.

The optimizer includes a non-settled A$0.0001/MWh throughput tie-breaker. This
is a numerical secondary preference that removes economically indifferent
cycling and zero-cost MIP branches; all reported revenue is recomputed without
it by the authoritative settlement function.

For the November 2022 SA1 reference month, 8,640 intervals and 30 daily
committed windows passed in every scenario.

## Exclusivity is tested in MW, not MW²

The gate on simultaneous charge and discharge tests
`min(charge_mw, discharge_mw)` against a tolerance scaled by the power rating,
`power_mw * 1e-6`, where `1e-6` is the HiGHS integrality tolerance.

An earlier version tested the product `charge_mw * discharge_mw` against a fixed
`1e-6`. That quantity has units of MW², so its sensitivity scaled with the
*square* of the power rating: the same solver slack that passed comfortably on
the 1 MW assets used throughout the test suite failed on the 100 MW reference
asset, a factor of 10,000. The A$10/MWh scenario duly failed on a window where
charge was 100 MW and discharge was 7.5e-8 MW — a hundredth of a microwatt of
integrality residue, not dispatch. Testing `min()` in MW is dimensionally
correct and means the same thing at any asset size.

The lesson generalises: a tolerance on a derived quantity must be checked
against the units and scale of the assets it will actually see, not only the
ones in the tests.

## Solver tolerance and where exclusivity binds

The model forbids simultaneous charge and discharge. Take any solution with
both charge `c` and discharge `d` positive in one interval and shift it by `-a`
on charge and `-a*eta_c*eta_d` on discharge: the stored-energy change, and so
the entire SOC path, is unchanged. The objective moves by
`-a * (A_c + A_d * eta_c * eta_d)` for settled per-MW coefficients
`A_c = -dt * L * p` and `A_d = dt * (G * p - deg)`. Simultaneous operation
therefore only pays when

    p < -deg * eta_c * eta_d / (L - G * eta_c * eta_d)

that is, only at prices below a multiple of the degradation cost. With the
FY2022-23 Hornsdale factors (G 0.9652, L 0.9681, round trip 0.87) the multiple
is 6.776, so the binary genuinely binds only below `-6.776 * deg`. Measured
against the SA1 November 2022 prices by
[`scripts/analyse_exclusivity_binding.py`](../scripts/analyse_exclusivity_binding.py):

| Degradation cost | Binary binds when | Intervals | Share | Days affected | Worst day |
|---|---:|---:|---:|---:|---:|
| A$0/MWh | p < A$0.00 | 3,287 | 38.04% | 28 / 30 | 288 of 288 |
| A$25/MWh | p < A$-169.42 | 234 | 2.71% | 13 / 30 | 48 of 288 |
| A$50/MWh | p < A$-338.85 | 85 | 0.98% | 12 / 30 | 23 of 288 |

### Eliminating the binaries that cannot bind

The threshold is used, not merely reported. Read the condition the other way
round: wherever `p > -6.776 * deg`, simultaneous operation is *strictly
dominated*, so every optimal solution already has `min(c, d) = 0` and the binary
is redundant rather than merely slack. The argument is purely local, because the
perturbation leaves the entire SOC path unchanged, so no global quantity such as
the shadow value of stored energy can rescue a locally dominated burn.

`exclusivity_binding_mask` in `src/optimiser.py` therefore evaluates the
bracket `A_c + eta_rt * A_d` per interval and creates a binary only where it is
non-negative. The bracket is evaluated directly rather than through the
rearranged price threshold, so dated loss factors and an adverse sign of
`L - G * eta_rt` are both handled without special cases. In the reference month
this leaves 468 binaries of a possible 17,280 at the headline A$25/MWh cost, and
13 of 30 windows solve as pure LPs whose gap is exactly zero rather than merely
small. Runs report `pure_lp_window_count` and `total_exclusivity_binary_count`
so a reported zero gap is attributable to an LP window instead of an unexplained
claim of exactness.

This is an exact reformulation, and `tests/test_exclusivity_reduction.py`
asserts that the reduced model reproduces the full-binary dispatch, objective
and SOC path. `scripts/run_mutation_checks.py` additionally verifies that a mask
which drops binaries it should have kept turns the exclusivity tests red.

### Why A$0/MWh is out of scope

At A$0/MWh the threshold collapses to `p < 0`. The model is then paid to burn
energy through round-trip losses, so it wants simultaneous operation at every
negative price, and in November 2022 that is 38% of intervals across 28 of 30
days. On 18 and 20 November every one of the 288 intervals is negative-priced,
and the many equivalent charge/discharge interleavings across a whole day leave
a large symmetric search tree that the reduction cannot touch. This is inherent
to the formulation, not a solver defect.

The measured cost of approaching that regime is steep. Solve time scales sharply
with the number of surviving binaries in a window:

| Binaries in window | Solve time |
|---:|---:|
| 2 | 0.17s |
| 74 | 1.64s |
| 154 | 23.8s |
| 284 | 88.1s |

Model construction is flat at about 0.17s across all of these, so the cost is
branch-and-bound, not model building. A$10/MWh already carries 1,988 binaries
across the month and takes ten times longer than the headline case; A$0/MWh
carries an order of magnitude more again and is not tractable at a strict gap.

Rather than declare a looser tolerance for one scenario and then have to defend
it, the zero-degradation case is excluded and the reasoning recorded here.
A$10/MWh serves as the low-degradation bound. Every reported scenario is solved
at a strict `1e-6` relative gap with no time limit and no fallback.

## Automated tests

`pytest` covers hand-worked settlement, loss-factor economics, SOC physics,
rolling-window carry-forward, timestamp boundaries, cache behavior, MLF input
validation, audit-gate completeness and sensitivity behavior. The deterministic
CI fixture exercises fetch-independent optimization, settlement, metrics and
audits end to end.

`scripts/run_mutation_checks.py` deliberately introduces twelve high-risk
errors: loss-factor removal, charge-efficiency removal, binary relaxation,
dropping binaries that can bind, skipping the audit-completeness assertion, disabling the ceiling gate,
settlement sign reversal, degradation sign reversal, wrong intervention
selection, wrong interval-ending convention, SOC reset and committing the
look-ahead tail. Each mutation must make its targeted test fail.

CI runs two jobs. The offline job runs tests with coverage, byte-compiles
source/tests/scripts, executes the deterministic fixture and runs all mutation
checks. The reference job downloads real AEMO data, verifies it against the
committed manifest checksum with `scripts/verify_reference.py`, and re-runs the
full published scenario grid. Because the whole grid takes about 100 seconds,
the published result is continuously verified rather than quoted from a run
nobody can afford to repeat.
