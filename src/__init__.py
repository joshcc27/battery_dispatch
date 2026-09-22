"""Energy-only NEM battery dispatch backtesting."""

from .config import BatteryConfig, LossFactors, LossFactorTable
from .horizon import run_rolling_horizon
from .optimiser import (
    STRICT_SOLVER_POLICY,
    DispatchWindow,
    SolverPolicy,
    solve_window,
)
from .settlement import RevenueResult, settle

__all__ = [
    "STRICT_SOLVER_POLICY",
    "BatteryConfig",
    "DispatchWindow",
    "LossFactors",
    "LossFactorTable",
    "RevenueResult",
    "SolverPolicy",
    "run_rolling_horizon",
    "settle",
    "solve_window",
]
