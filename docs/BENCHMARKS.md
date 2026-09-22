# External benchmark policy

Published battery results are contextual checks, not calibration targets. A
comparison is like-for-like only when period, region, asset scope, capacity
basis and revenue services match. The reference model is a hypothetical single
100 MW / 200 MWh battery and includes wholesale-energy arbitrage only.

`config/benchmarks.csv` preserves the published period, geography, fleet scope,
energy revenue, FCAS revenue, capacity basis and page-level source. The loader
refuses incomplete provenance. Per-MW figures remain missing when a source does
not publish a usable capacity basis; the code does not invent one.

The two included AEMO Quarterly Energy Dynamics observations are deliberately
out of period with FY2022-23 and cover fleets rather than the reference asset:

- SA batteries earned A$20.4 million in Q1 2024: A$8.7 million from energy and
  A$11.7 million from FCAS (QED Q1 2024, p.39). No comparable capacity basis is
  provided in the stored observation.
- NEM batteries earned A$48.1 million from energy and A$21.3 million from FCAS
  in Q4 2024, with 1,087 MW average availability (QED Q4 2024, p.38).

These figures demonstrate why the model must not be described as total battery
revenue and why a numerical difference cannot be attributed solely to model
error. `compare_to_benchmark` always returns explicit alignment and scope flags
alongside the values.
