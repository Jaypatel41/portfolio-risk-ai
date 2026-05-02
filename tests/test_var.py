"""Tests for Task 5's VaR/CVaR + Monte Carlo math. No network."""

from __future__ import annotations

import numpy as np
import pytest

from timecell.models import Portfolio
from timecell.var import (
    compute_var_report,
    historical_var_cvar,
    monte_carlo_metrics,
    monte_carlo_paths,
)


def _portfolio(value: float = 10_000_000, expenses: float = 80_000) -> Portfolio:
    return Portfolio.model_validate({
        "total_value_inr": value,
        "monthly_expenses_inr": expenses,
        "assets": [
            {"name": "BTC", "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD", "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH", "allocation_pct": 10, "expected_crash_pct": 0},
        ],
    })


def test_historical_var_uses_empirical_percentile() -> None:
    """For a uniform sequence, the 95% VaR should be the 5th percentile."""
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.0, scale=0.02, size=2000)
    var = historical_var_cvar(returns, confidence=0.95, portfolio_value=1_000_000)
    expected = float(np.expm1(np.percentile(returns, 5)) * 100)
    assert var.var_pct == pytest.approx(expected, abs=0.05)


def test_cvar_is_at_least_as_severe_as_var() -> None:
    """By construction CVaR averages the tail beyond the VaR cutoff,
    so CVaR ≤ VaR (more negative = worse)."""
    rng = np.random.default_rng(1)
    returns = rng.normal(loc=0.0, scale=0.03, size=5000)
    v = historical_var_cvar(returns, 0.95, 1_000_000)
    assert v.cvar_pct <= v.var_pct


def test_99_pct_var_is_worse_than_95_pct_var() -> None:
    """Higher confidence reaches further into the tail, so the loss is worse."""
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=0.0, scale=0.02, size=5000)
    v95 = historical_var_cvar(returns, 0.95, 1_000_000)
    v99 = historical_var_cvar(returns, 0.99, 1_000_000)
    assert v99.var_pct < v95.var_pct


def test_var_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        historical_var_cvar(np.array([0.01]), 0.95, 1_000_000)


def test_monte_carlo_paths_are_reproducible_with_seed() -> None:
    """Same seed → same paths. Critical for explainability."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 0.02, size=(500, 4))
    weights = np.array([0.3, 0.4, 0.2, 0.1])
    a = monte_carlo_paths(matrix, weights, n_paths=1000, horizon_days=30, seed=42)
    b = monte_carlo_paths(matrix, weights, n_paths=1000, horizon_days=30, seed=42)
    np.testing.assert_array_equal(a, b)


def test_monte_carlo_paths_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 0.02, size=(500, 4))
    weights = np.array([0.3, 0.4, 0.2, 0.1])
    paths = monte_carlo_paths(matrix, weights, n_paths=500, horizon_days=10, seed=7)
    assert paths.shape == (500,)
    # Bootstrap of small daily returns over 10 days won't exceed sane bounds.
    assert paths.min() > -0.99
    assert paths.max() < 5.0


def test_monte_carlo_zero_expenses_gives_zero_ruin_probability() -> None:
    """No burn → ruin is impossible by definition (PASS in the discrete test too)."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 0.02, size=(500, 2))
    weights = np.array([0.5, 0.5])
    mc = monte_carlo_metrics(
        matrix, weights, portfolio_value=1_000_000, monthly_expenses=0,
        n_paths=1000, horizon_days=30, seed=0,
    )
    assert mc.ruin_probability_pct == 0.0


def test_zero_return_series_gives_zero_var() -> None:
    """All-zero history (cash-only world) → no risk, VaR/CVaR ≈ 0."""
    p = _portfolio(value=1_000_000, expenses=0)
    returns = {a.name: np.zeros(500) for a in p.assets}
    report = compute_var_report(p, returns, n_paths=500, horizon_days=30, seed=0)
    assert report.var_95_1d.var_pct == pytest.approx(0.0, abs=0.01)
    assert report.var_99_1d.var_pct == pytest.approx(0.0, abs=0.01)
    assert report.monte_carlo.ruin_probability_pct == 0.0


def test_compute_var_report_flags_synthetic_assets() -> None:
    """Assets without history are listed in `assets_synthetic` (CASH always is)."""
    p = _portfolio()
    rng = np.random.default_rng(0)
    # Only BTC and NIFTY50 have history; GOLD and CASH fall back.
    returns = {
        "BTC": rng.normal(0, 0.04, size=500),
        "NIFTY50": rng.normal(0, 0.015, size=500),
        "GOLD": np.array([]),
        "CASH": np.array([]),
    }
    report = compute_var_report(p, returns, n_paths=500, horizon_days=30, seed=0)
    assert set(report.assets_with_history) == {"BTC", "NIFTY50"}
    assert set(report.assets_synthetic) == {"GOLD", "CASH"}


def test_compute_var_report_raises_when_no_history_at_all() -> None:
    p = _portfolio()
    returns = {a.name: np.array([]) for a in p.assets}
    with pytest.raises(ValueError, match="no asset has historical data"):
        compute_var_report(p, returns)


def test_compute_var_report_aligns_to_shortest_history() -> None:
    """Different-length series should be truncated to the common minimum."""
    p = _portfolio()
    rng = np.random.default_rng(0)
    returns = {
        "BTC": rng.normal(0, 0.04, size=1000),
        "NIFTY50": rng.normal(0, 0.015, size=300),  # the bottleneck
        "GOLD": rng.normal(0, 0.01, size=800),
        "CASH": np.array([]),
    }
    report = compute_var_report(p, returns, n_paths=500, horizon_days=30, seed=0)
    assert report.historical_window_days == 300
