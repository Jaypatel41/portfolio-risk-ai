"""Task 4 — Natural-language stress-test.

The pitch: a wealth-management UI is the wrong shape for "what if" questions,
but a terminal that takes plain English is exactly right.

Pipeline:
  1. User types a scenario, e.g. "what if BTC crashes 70% and gold rallies 20%?"
  2. Claude parses it into per-asset shocks via the same Decision-Spine-style
     structured-output pattern used in Task 3.
  3. Shocks are applied to a *copy* of the portfolio. Rallies clamp to 0% loss
     for the crash-survival math — we're stress-testing downside, not upside.
  4. The risk engine from Task 1 re-runs on the shocked portfolio.

This is the open-problem submission: it composes Tasks 1 + 3, demonstrates
the LLM as a structured *parser* (not just a generator), and produces something
a Timecell user would actually use.
"""

from __future__ import annotations

from typing import Any

from .ai_client import call_structured
from .models import Asset, Portfolio, StressScenario
from .prompts import STRESS_SYSTEM_PROMPT, build_stress_user_prompt
from .risk import compute_risk_report


def parse_scenario(portfolio: Portfolio, scenario: str) -> StressScenario:
    """Send the scenario to Claude and get a typed StressScenario back."""
    parsed, _ = call_structured(
        system_prompt=STRESS_SYSTEM_PROMPT,
        user_prompt=build_stress_user_prompt(portfolio, scenario),
        output_schema=StressScenario,
        temperature=0.0,  # parser — strict, deterministic
    )
    return parsed


def apply_shocks(portfolio: Portfolio, stress: StressScenario) -> Portfolio:
    """Return a new Portfolio with `expected_crash_pct` overwritten for shocked assets.

    Rallies clamp to 0 — we're testing crash survival, not upside.
    """
    shock_map = {s.asset_name.lower(): s.shock_pct for s in stress.shocks}
    new_assets: list[Asset] = []
    for a in portfolio.assets:
        shock = shock_map.get(a.name.lower())
        if shock is None:
            new_assets.append(a.model_copy())  # not shocked
            continue
        # Clamp positive (rally) shocks to 0 loss for the crash-survival math.
        applied = min(shock, 0)
        new_assets.append(
            Asset(
                name=a.name,
                allocation_pct=a.allocation_pct,
                expected_crash_pct=applied,
            )
        )
    return Portfolio(
        total_value_inr=portfolio.total_value_inr,
        monthly_expenses_inr=portfolio.monthly_expenses_inr,
        assets=new_assets,
    )


def run_stress_test(portfolio: Portfolio, scenario: str) -> dict[str, Any]:
    """Full pipeline: parse → apply → re-run risk math."""
    parsed = parse_scenario(portfolio, scenario)
    shocked = apply_shocks(portfolio, parsed)
    report = compute_risk_report(shocked)
    return {
        "scenario": scenario,
        "parsed": parsed,
        "shocked_portfolio": shocked,
        "report": report,
    }
