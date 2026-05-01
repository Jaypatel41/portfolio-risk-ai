"""Task 1 — Portfolio risk math. Pure functions, no I/O.

`compute_risk_metrics(portfolio)` returns the exact dict shape the brief asks for.
`compute_risk_report(portfolio)` returns the typed `RiskReport` with the bonus
moderate scenario and concentration data — used by the CLI, the briefing, and
the Streamlit app.
"""

from __future__ import annotations

import math
from typing import Any

from .models import Portfolio, RiskReport, ScenarioResult

CONCENTRATION_THRESHOLD_PCT = 40.0
RUIN_THRESHOLD_MONTHS = 12.0
MODERATE_SEVERITY = 0.5  # 50% of expected crash magnitude


def _apply_scenario(
    portfolio: Portfolio,
    severity: float,
    label: str,
) -> ScenarioResult:
    """Run one crash scenario at `severity` × each asset's expected_crash_pct."""
    if not 0 <= severity <= 1:
        raise ValueError(f"severity must be in [0, 1], got {severity}")

    pre = portfolio.total_value_inr
    post = 0.0
    for a in portfolio.assets:
        asset_value = pre * (a.allocation_pct / 100)
        applied_drop_frac = (a.expected_crash_pct / 100) * severity  # ≤ 0
        post += asset_value * (1 + applied_drop_frac)

    drawdown_pct = (post - pre) / pre * 100 if pre else 0.0
    runway = math.inf if portfolio.monthly_expenses_inr == 0 else post / portfolio.monthly_expenses_inr
    return ScenarioResult(
        label=label,
        post_crash_value=round(post, 2),
        drawdown_pct=round(drawdown_pct, 2),
        runway_months=runway if math.isinf(runway) else round(runway, 2),
        ruin_test="PASS" if runway > RUIN_THRESHOLD_MONTHS else "FAIL",
    )


def _largest_risk_asset(portfolio: Portfolio) -> str:
    """Asset with highest (allocation × |crash magnitude|)."""
    return max(
        portfolio.assets,
        key=lambda a: a.allocation_pct * abs(a.expected_crash_pct),
    ).name


def compute_risk_report(portfolio: Portfolio) -> RiskReport:
    severe = _apply_scenario(portfolio, severity=1.0, label="Severe")
    moderate = _apply_scenario(portfolio, severity=MODERATE_SEVERITY, label="Moderate")
    top_alloc = max(a.allocation_pct for a in portfolio.assets)

    return RiskReport(
        portfolio_value_inr=portfolio.total_value_inr,
        monthly_expenses_inr=portfolio.monthly_expenses_inr,
        severe=severe,
        moderate=moderate,
        largest_risk_asset=_largest_risk_asset(portfolio),
        concentration_warning=top_alloc > CONCENTRATION_THRESHOLD_PCT,
        concentration_top_pct=round(top_alloc, 2),
    )


def compute_risk_metrics(portfolio: Portfolio | dict) -> dict[str, Any]:
    """Spec-shape output for Task 1.

    Returns exactly the keys the brief asks for. Accepts either a `Portfolio`
    or the raw dict, so the function signature matches the brief verbatim.
    """
    p = portfolio if isinstance(portfolio, Portfolio) else Portfolio.model_validate(portfolio)
    report = compute_risk_report(p)
    return {
        "post_crash_value": report.severe.post_crash_value,
        "runway_months": report.severe.runway_months,
        "ruin_test": report.severe.ruin_test,
        "largest_risk_asset": report.largest_risk_asset,
        "concentration_warning": report.concentration_warning,
    }


def render_allocation_chart(portfolio: Portfolio, width: int = 40) -> str:
    """ASCII bar chart — no external plotting libraries, per the bonus."""
    lines = ["Allocation breakdown:"]
    name_width = max(len(a.name) for a in portfolio.assets)
    for a in portfolio.assets:
        filled = int(round(a.allocation_pct / 100 * width))
        bar = "█" * filled + "░" * (width - filled)
        lines.append(f"  {a.name:<{name_width}}  {bar}  {a.allocation_pct:>5.1f}%")
    return "\n".join(lines)
