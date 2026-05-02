"""Tests for Task 1's risk math. No network, no LLM calls."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from timecell.models import Portfolio
from timecell.risk import (
    CONCENTRATION_THRESHOLD_PCT,
    compute_risk_metrics,
    compute_risk_report,
    render_allocation_chart,
)

# Brief's example portfolio — used as the canonical fixture.
BRIEF_PORTFOLIO = {
    "total_value_inr": 10_000_000,
    "monthly_expenses_inr": 80_000,
    "assets": [
        {"name": "BTC", "allocation_pct": 30, "expected_crash_pct": -80},
        {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
        {"name": "GOLD", "allocation_pct": 20, "expected_crash_pct": -15},
        {"name": "CASH", "allocation_pct": 10, "expected_crash_pct": 0},
    ],
}


def test_brief_example_returns_correct_dict_shape() -> None:
    """compute_risk_metrics must return EXACTLY the keys the brief asks for."""
    out = compute_risk_metrics(BRIEF_PORTFOLIO)
    assert set(out.keys()) == {
        "post_crash_value",
        "runway_months",
        "ruin_test",
        "largest_risk_asset",
        "concentration_warning",
    }


def test_brief_example_math_is_correct() -> None:
    """Hand-computed: BTC -80%×30% = -24%, NIFTY -40%×40% = -16%,
    GOLD -15%×20% = -3%, CASH 0. Total drawdown = -43%, post-crash = 5,700,000."""
    out = compute_risk_metrics(BRIEF_PORTFOLIO)
    assert out["post_crash_value"] == pytest.approx(5_700_000, abs=1)
    assert out["runway_months"] == pytest.approx(5_700_000 / 80_000, abs=0.01)
    assert out["ruin_test"] == "PASS"  # 71+ months > 12
    assert out["largest_risk_asset"] == "BTC"  # 30% × 80 = 2400 > 40 × 40 = 1600
    assert out["concentration_warning"] is False  # 40% is NOT > 40%


def test_concentration_threshold_is_strict_greater_than() -> None:
    """Exactly 40% must NOT trip the warning — only > 40%."""
    p_at_40 = {**BRIEF_PORTFOLIO}  # NIFTY is at 40%
    assert compute_risk_metrics(p_at_40)["concentration_warning"] is False

    p_above = {
        "total_value_inr": 10_000_000,
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC", "allocation_pct": 41, "expected_crash_pct": -80},
            {"name": "CASH", "allocation_pct": 59, "expected_crash_pct": 0},
        ],
    }
    assert compute_risk_metrics(p_above)["concentration_warning"] is True
    assert CONCENTRATION_THRESHOLD_PCT == 40.0


def test_all_cash_portfolio_is_safe() -> None:
    p = {
        "total_value_inr": 1_000_000,
        "monthly_expenses_inr": 10_000,
        "assets": [{"name": "CASH", "allocation_pct": 100, "expected_crash_pct": 0}],
    }
    out = compute_risk_metrics(p)
    assert out["post_crash_value"] == 1_000_000
    assert out["ruin_test"] == "PASS"
    assert out["concentration_warning"] is True  # 100% > 40%


def test_zero_monthly_expenses_gives_infinite_runway() -> None:
    p = {**BRIEF_PORTFOLIO, "monthly_expenses_inr": 0}
    out = compute_risk_metrics(p)
    assert math.isinf(out["runway_months"])
    assert out["ruin_test"] == "PASS"


def test_allocations_must_sum_to_100() -> None:
    bad = {
        "total_value_inr": 1_000_000,
        "monthly_expenses_inr": 10_000,
        "assets": [
            {"name": "A", "allocation_pct": 30, "expected_crash_pct": -10},
            {"name": "B", "allocation_pct": 30, "expected_crash_pct": -10},
        ],
    }
    with pytest.raises(ValidationError, match="must sum to 100"):
        Portfolio.model_validate(bad)


def test_duplicate_asset_names_are_rejected() -> None:
    bad = {
        "total_value_inr": 1_000_000,
        "monthly_expenses_inr": 10_000,
        "assets": [
            {"name": "BTC", "allocation_pct": 50, "expected_crash_pct": -80},
            {"name": "BTC", "allocation_pct": 50, "expected_crash_pct": -80},
        ],
    }
    with pytest.raises(ValidationError, match="duplicate"):
        Portfolio.model_validate(bad)


def test_positive_crash_pct_is_rejected() -> None:
    """expected_crash_pct must be ≤ 0 — a 'crash' that's a gain makes no sense."""
    bad = {
        "total_value_inr": 1_000_000,
        "monthly_expenses_inr": 10_000,
        "assets": [{"name": "A", "allocation_pct": 100, "expected_crash_pct": 10}],
    }
    with pytest.raises(ValidationError):
        Portfolio.model_validate(bad)


def test_moderate_scenario_is_half_severity() -> None:
    """Bonus: moderate scenario applies 50% of the expected crash magnitude."""
    report = compute_risk_report(Portfolio.model_validate(BRIEF_PORTFOLIO))
    # Severe drawdown = -43%, so moderate should be -21.5%.
    assert report.severe.drawdown_pct == pytest.approx(-43, abs=0.5)
    assert report.moderate.drawdown_pct == pytest.approx(-21.5, abs=0.5)


def test_allocation_chart_renders_without_external_libs() -> None:
    """Bonus: ASCII bar chart should produce one line per asset."""
    chart = render_allocation_chart(Portfolio.model_validate(BRIEF_PORTFOLIO))
    assert "BTC" in chart and "30.0%" in chart
    assert "█" in chart  # filled bar character
    assert chart.count("\n") == 4  # header + 4 assets
