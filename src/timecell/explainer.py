"""Task 3 — AI-powered portfolio explainer with the Decision Spine pattern.

Pipeline:
  1. Pre-compute risk metrics in Python (deterministic).
  2. Hand the metrics + tone + portfolio to Claude as a flat JSON block.
  3. Claude must cite a metric for every claim it makes (Decision Spine).
  4. Output is constrained to the `Explanation` Pydantic schema and validated.
  5. Optional second LLM call critiques the first against the same metrics.
"""

from __future__ import annotations

from .ai_client import call_structured
from .models import CriticReport, Explanation, Portfolio, RiskReport, Tone
from .prompts import (
    CRITIC_SYSTEM_PROMPT,
    EXPLAINER_SYSTEM_PROMPT,
    build_critic_user_prompt,
    build_explainer_user_prompt,
)
from .risk import compute_risk_report


def explain_portfolio(
    portfolio: Portfolio,
    tone: Tone = "experienced",
    report: RiskReport | None = None,
) -> tuple[Explanation, str]:
    """Generate a Decision-Spine-backed explanation of the portfolio.

    Returns (parsed Explanation, raw LLM text). The brief explicitly asks for
    both, so callers can render them side by side.
    """
    report = report or compute_risk_report(portfolio)
    return call_structured(
        system_prompt=EXPLAINER_SYSTEM_PROMPT,
        user_prompt=build_explainer_user_prompt(portfolio, report, tone),
        output_schema=Explanation,
        temperature=0.2,
    )


def critique_explanation(
    portfolio: Portfolio,
    explanation: Explanation,
    report: RiskReport | None = None,
) -> CriticReport:
    """Second LLM pass that fact-checks the first."""
    report = report or compute_risk_report(portfolio)
    parsed, _ = call_structured(
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_prompt=build_critic_user_prompt(portfolio, report, explanation.model_dump_json()),
        output_schema=CriticReport,
        temperature=0.0,  # critic is a verification task — deterministic > creative
    )
    return parsed
