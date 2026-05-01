"""All LLM prompts in one place.

The Decision Spine pattern is the heart of this submission:
  - Math runs in Python; numbers are pre-computed and handed to the LLM.
  - The LLM is *forbidden* from inventing numbers — every claim must cite a
    metric from the JSON block. If it cannot cite, it must drop the claim.
  - Output is constrained to a Pydantic schema, validated on parse, and any
    schema violation fails loudly rather than silently shipping bad output.
"""

from __future__ import annotations

import json
from typing import Any

from .models import Portfolio, RiskReport, Tone


def _metrics_block(portfolio: Portfolio, report: RiskReport) -> dict[str, Any]:
    """Flat JSON block of every fact the explainer is allowed to use."""
    return {
        "portfolio_value_inr": report.portfolio_value_inr,
        "monthly_expenses_inr": report.monthly_expenses_inr,
        "asset_allocations_pct": {a.name: a.allocation_pct for a in portfolio.assets},
        "asset_expected_crash_pct": {a.name: a.expected_crash_pct for a in portfolio.assets},
        "severe_post_crash_value": report.severe.post_crash_value,
        "severe_drawdown_pct": report.severe.drawdown_pct,
        "severe_runway_months": report.severe.runway_months,
        "severe_ruin_test": report.severe.ruin_test,
        "moderate_post_crash_value": report.moderate.post_crash_value,
        "moderate_drawdown_pct": report.moderate.drawdown_pct,
        "moderate_runway_months": report.moderate.runway_months,
        "moderate_ruin_test": report.moderate.ruin_test,
        "largest_risk_asset": report.largest_risk_asset,
        "concentration_warning": report.concentration_warning,
        "concentration_top_pct": report.concentration_top_pct,
    }


_TONE_GUIDE = {
    "beginner": (
        "Audience: a non-financial 60-year-old. Avoid jargon. If you must use a "
        "term, explain it in the same sentence. Use FD/SIP/EMI analogies sparingly."
    ),
    "experienced": (
        "Audience: an investor who reads the financial news. You may use terms "
        "like 'drawdown', 'concentration', 'runway' without defining them, but "
        "stay grounded in concrete numbers."
    ),
    "expert": (
        "Audience: another wealth manager. Be terse and quantitative. Skip "
        "preamble. Reference Sharpe-style framing if relevant, but never invent "
        "numbers — only what is in the metrics block."
    ),
}


EXPLAINER_SYSTEM_PROMPT = """You are an honest, friendly financial advisor explaining a portfolio's risk \
to a client. You do NOT compute numbers. Python has already done the math; \
your job is to narrate.

═══ HARD RULES — break any of these and the answer is invalid ═══

1. NEVER invent or estimate numbers. You will be given a JSON `metrics` block.
   Every number in your prose must come from that block. If a fact you want
   to state is not in `metrics`, drop the fact.

2. EVERY claim in your output (summary, doing_well, consider_changing) must
   map to one entry in the `spine` array:
     { "claim": "...", "cited_metric": "metric_key = value", "confidence_pct": 0..100 }
   If you cannot cite a metric for a claim, do not make that claim.

3. The verdict must be 'Aggressive', 'Balanced', or 'Conservative' — chosen
   from the metrics, not your priors. Heuristic:
     - severe_drawdown_pct worse than -50%, OR concentration_warning=true,
       OR severe_ruin_test=FAIL  → Aggressive
     - severe_drawdown_pct between -25% and -50%, runway > 12 months,
       no concentration_warning → Balanced
     - severe_drawdown_pct better than -25%, runway > 36 months → Conservative

4. Output ONLY the structured JSON the schema asks for. No markdown, no
   preamble, no trailing commentary.

═══ STYLE ═══

- summary: 3–4 sentences, plain English, written like a human advisor talking.
- doing_well: ONE specific thing the investor is doing well, with the metric
  that justifies the praise.
- consider_changing: ONE specific thing they should consider changing, with
  the metric that justifies the concern, and a brief 'why this matters'.

The tone-specific guidance for THIS request will appear in the user message.
"""


def build_explainer_user_prompt(
    portfolio: Portfolio,
    report: RiskReport,
    tone: Tone,
) -> str:
    metrics = _metrics_block(portfolio, report)
    tone_guidance = _TONE_GUIDE[tone]
    return (
        f"TONE GUIDANCE\n{tone_guidance}\n\n"
        f"METRICS (your only source of truth — never invent numbers outside this block)\n"
        f"```json\n{json.dumps(metrics, indent=2)}\n```\n\n"
        f"Produce the structured explanation now. Set the `tone` field to '{tone}'."
    )


# ── Critic ────────────────────────────────────────────────────────────────


CRITIC_SYSTEM_PROMPT = """You are a fact-checker reviewing a portfolio explanation \
written by another LLM. Your job is to verify every claim against the same \
metrics block.

For each claim in the explanation:
1. Find the metric the claim cites.
2. Check whether the claim is consistent with that metric.
3. If a claim contradicts a metric, or cites a metric that does not exist,
   flag it as 'major'. Stylistic nits are 'minor'.

Output ONLY the structured JSON:
- overall: 'PASS' if no issues, 'PASS_WITH_NITS' if only minor, 'FAIL' if any major.
- issues: [{ severity: 'minor' | 'major', claim: '...', problem: '...' }]
"""


def build_critic_user_prompt(
    portfolio: Portfolio,
    report: RiskReport,
    explanation_json: str,
) -> str:
    metrics = _metrics_block(portfolio, report)
    return (
        f"METRICS\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n"
        f"EXPLANATION TO REVIEW\n```json\n{explanation_json}\n```\n\n"
        f"Produce the critic report now."
    )


# ── Stress-test parser (Task 4) ───────────────────────────────────────────


STRESS_SYSTEM_PROMPT = """You translate plain-English market scenarios into a \
structured list of per-asset price shocks.

Input: a portfolio with named assets, plus a free-form scenario like
  "what if BTC crashes 70% and gold rallies 20%?"

Output a `StressScenario` with:
- rationale: ONE sentence summarizing how you read the prompt.
- shocks: a list of { asset_name, shock_pct } entries.

Rules:
- shock_pct is signed: -70 means a 70% drop, +20 means a 20% rally.
- Match asset names case-insensitively to the portfolio's assets. Use the
  portfolio's exact asset_name spelling in the output.
- If the user mentions a category ("all crypto"), expand to every matching
  asset in the portfolio.
- Assets the prompt does not mention should be omitted (no 0% shocks).
- If the prompt is ambiguous, prefer the more conservative interpretation
  (smaller magnitude) and note it in `rationale`.
"""


def build_stress_user_prompt(portfolio: Portfolio, scenario: str) -> str:
    asset_list = [{"name": a.name, "allocation_pct": a.allocation_pct} for a in portfolio.assets]
    return (
        f"PORTFOLIO ASSETS\n```json\n{json.dumps(asset_list, indent=2)}\n```\n\n"
        f"SCENARIO\n{scenario.strip()}\n\n"
        f"Produce the StressScenario now."
    )
