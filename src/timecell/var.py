"""Task 5 — Historical VaR/CVaR and Monte Carlo simulation.

Pure numpy. No I/O — caller passes a per-asset return dict; we return a typed
report. Risk is measured from the empirical distribution: no normality
assumption, and Monte Carlo bootstraps full rows from the historical matrix
so cross-asset correlation is preserved automatically.

Conventions:
  - Returns are *log* returns (additive across days). VaR/CVaR cutoffs are
    converted back to simple returns via expm1 before reporting.
  - VaR_95 means "the 5% worst-case loss"; numbers are reported as signed
    negatives (e.g. -3.2%) so they read the same as drawdowns elsewhere.
  - Monte Carlo end-value distribution drives the ruin probability — this is
    the natural extension of Task 1's binary PASS/FAIL ruin test.
"""

from __future__ import annotations

import numpy as np

from .models import MonteCarloMetrics, Portfolio, VaRMetrics, VaRReport

RUIN_THRESHOLD_MONTHS = 12.0


def _align_matrix(
    portfolio: Portfolio,
    asset_returns: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build a [days × assets] log-return matrix aligned to the shortest series.

    Assets without history fall back to a zero-return column (cash-like). This
    is conservative: it understates risk for the missing asset rather than
    inventing volatility, and the report flags which assets used the fallback.
    """
    real_lengths = [
        len(asset_returns[a.name])
        for a in portfolio.assets
        if a.name in asset_returns and len(asset_returns[a.name]) > 1
    ]
    if not real_lengths:
        raise ValueError(
            "no asset has historical data — cannot compute VaR. "
            "Check network access for yfinance/coingecko."
        )
    common_len = min(real_lengths)

    columns: list[np.ndarray] = []
    weights: list[float] = []
    real: list[str] = []
    synthetic: list[str] = []

    for a in portfolio.assets:
        weights.append(a.allocation_pct / 100)
        series = asset_returns.get(a.name)
        if series is not None and len(series) > 1:
            columns.append(np.asarray(series[-common_len:], dtype=float))
            real.append(a.name)
        else:
            columns.append(np.zeros(common_len))
            synthetic.append(a.name)

    return np.column_stack(columns), np.asarray(weights), real, synthetic


def historical_var_cvar(
    portfolio_log_returns: np.ndarray,
    confidence: float,
    portfolio_value: float,
    horizon_days: int = 1,
) -> VaRMetrics:
    """Empirical VaR/CVaR from a 1-D array of portfolio log returns.

    For multi-day horizons we use the square-root-of-time scaling on the
    1-day VaR — standard practice and exact under IID-normal returns; for
    the empirical distribution it is a defensible approximation. If the
    caller wants distributional honesty for >1-day horizons, use Monte
    Carlo (which bootstraps multi-day cumulative returns directly).
    """
    if portfolio_log_returns.size < 2:
        raise ValueError("need at least 2 daily returns to compute VaR")

    cutoff_pct = (1 - confidence) * 100
    var_log = float(np.percentile(portfolio_log_returns, cutoff_pct))
    tail = portfolio_log_returns[portfolio_log_returns <= var_log]
    cvar_log = float(tail.mean()) if tail.size else var_log

    scale = float(np.sqrt(horizon_days))
    var_simple = float(np.expm1(var_log * scale))
    cvar_simple = float(np.expm1(cvar_log * scale))

    return VaRMetrics(
        confidence_pct=round(confidence * 100, 2),
        horizon_days=horizon_days,
        var_pct=round(var_simple * 100, 2),
        var_inr=round(var_simple * portfolio_value, 2),
        cvar_pct=round(cvar_simple * 100, 2),
        cvar_inr=round(cvar_simple * portfolio_value, 2),
    )


def monte_carlo_paths(
    return_matrix: np.ndarray,
    weights: np.ndarray,
    n_paths: int,
    horizon_days: int,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap n_paths × horizon_days days; return horizon-total simple returns.

    Sampling whole rows (not per-asset) preserves the empirical correlation
    between assets — no covariance estimate, no Cholesky factorization. This
    is "historical bootstrap" and is the most assumption-light approach.
    """
    if return_matrix.shape[0] < 2:
        raise ValueError("need at least 2 historical observations to bootstrap")

    rng = np.random.default_rng(seed)
    daily_pf = return_matrix @ weights  # 1-D portfolio log-return series
    sample_idx = rng.integers(0, daily_pf.size, size=(n_paths, horizon_days))
    summed_log = daily_pf[sample_idx].sum(axis=1)
    return np.expm1(summed_log)


def monte_carlo_metrics(
    return_matrix: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float,
    monthly_expenses: float,
    n_paths: int = 10_000,
    horizon_days: int = 252,
    seed: int = 42,
) -> MonteCarloMetrics:
    pct_returns = monte_carlo_paths(return_matrix, weights, n_paths, horizon_days, seed)
    end_values = portfolio_value * (1 + pct_returns)
    p5, p50, p95 = (float(x) for x in np.percentile(end_values, [5, 50, 95]))

    if monthly_expenses > 0:
        runway = end_values / monthly_expenses
        ruin_prob = float((runway < RUIN_THRESHOLD_MONTHS).mean()) * 100
    else:
        ruin_prob = 0.0

    return MonteCarloMetrics(
        n_paths=n_paths,
        horizon_days=horizon_days,
        p5_post_value=round(p5, 2),
        p50_post_value=round(p50, 2),
        p95_post_value=round(p95, 2),
        ruin_probability_pct=round(ruin_prob, 2),
    )


def compute_var_report(
    portfolio: Portfolio,
    asset_returns: dict[str, np.ndarray],
    n_paths: int = 10_000,
    horizon_days: int = 252,
    seed: int = 42,
) -> VaRReport:
    """End-to-end: returns dict → typed VaRReport."""
    return_matrix, weights, real, synthetic = _align_matrix(portfolio, asset_returns)
    portfolio_log_returns = return_matrix @ weights
    pv = portfolio.total_value_inr

    return VaRReport(
        portfolio_value_inr=pv,
        historical_window_days=return_matrix.shape[0],
        var_95_1d=historical_var_cvar(portfolio_log_returns, 0.95, pv, horizon_days=1),
        var_99_1d=historical_var_cvar(portfolio_log_returns, 0.99, pv, horizon_days=1),
        monte_carlo=monte_carlo_metrics(
            return_matrix,
            weights,
            pv,
            portfolio.monthly_expenses_inr,
            n_paths=n_paths,
            horizon_days=horizon_days,
            seed=seed,
        ),
        assets_with_history=real,
        assets_synthetic=synthetic,
    )
