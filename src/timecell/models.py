"""Pydantic v2 source of truth for every typed object in the system."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Tone = Literal["beginner", "experienced", "expert"]
Verdict = Literal["Aggressive", "Balanced", "Conservative"]
RuinResult = Literal["PASS", "FAIL"]


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    allocation_pct: float = Field(ge=0, le=100)
    expected_crash_pct: float = Field(le=0, description="Negative or zero. -80 means -80%.")


class Portfolio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_value_inr: float = Field(gt=0)
    monthly_expenses_inr: float = Field(ge=0)
    assets: list[Asset] = Field(min_length=1)

    @field_validator("assets")
    @classmethod
    def _no_duplicate_names(cls, v: list[Asset]) -> list[Asset]:
        names = [a.name for a in v]
        if len(set(names)) != len(names):
            raise ValueError("duplicate asset names not allowed")
        return v

    @model_validator(mode="after")
    def _allocations_sum_to_100(self) -> Portfolio:
        total = sum(a.allocation_pct for a in self.assets)
        if abs(total - 100) > 0.01:
            raise ValueError(f"allocation_pct must sum to 100, got {total}")
        return self


# ── Task 1 outputs ────────────────────────────────────────────────────────


class ScenarioResult(BaseModel):
    """One crash scenario applied to the portfolio."""
    label: str
    post_crash_value: float
    drawdown_pct: float
    runway_months: float
    ruin_test: RuinResult


class RiskReport(BaseModel):
    """Full risk picture: severe + moderate scenarios + concentration data."""
    portfolio_value_inr: float
    monthly_expenses_inr: float
    severe: ScenarioResult
    moderate: ScenarioResult
    largest_risk_asset: str
    concentration_warning: bool
    concentration_top_pct: float


# ── Task 3 outputs (Decision Spine) ───────────────────────────────────────


class SpineEntry(BaseModel):
    """A single claim, the metric that justifies it, and the model's confidence."""
    claim: str
    cited_metric: str
    confidence_pct: int = Field(ge=0, le=100)


class Explanation(BaseModel):
    """Structured output from the explainer LLM."""
    summary: str = Field(description="3-4 sentence plain-English risk summary.")
    doing_well: str = Field(description="One specific thing the investor is doing well.")
    consider_changing: str = Field(description="One specific thing to consider changing, with reason.")
    verdict: Verdict
    spine: list[SpineEntry] = Field(min_length=1)
    tone: Tone


class CriticIssue(BaseModel):
    severity: Literal["minor", "major"]
    claim: str
    problem: str


class CriticReport(BaseModel):
    overall: Literal["PASS", "PASS_WITH_NITS", "FAIL"]
    issues: list[CriticIssue]


# ── Task 4 outputs (NL stress-test) ───────────────────────────────────────


class AssetShock(BaseModel):
    asset_name: str
    shock_pct: float = Field(description="-100 to +200. Negative = drop, positive = rally.")


class StressScenario(BaseModel):
    """Structured parse of a natural-language stress-test prompt."""
    rationale: str = Field(description="One-sentence summary of how the user's prompt was interpreted.")
    shocks: list[AssetShock]


# ── Task 5 outputs (historical VaR/CVaR + Monte Carlo) ────────────────────


class VaRMetrics(BaseModel):
    """One VaR/CVaR pair at a given confidence level, in % and INR."""
    confidence_pct: float
    horizon_days: int
    var_pct: float = Field(description="Loss threshold as a signed percent (e.g. -3.2).")
    var_inr: float
    cvar_pct: float = Field(description="Expected loss in the tail beyond VaR (signed percent).")
    cvar_inr: float


class MonteCarloMetrics(BaseModel):
    """Distribution of horizon-end portfolio values across simulated paths."""
    n_paths: int
    horizon_days: int
    p5_post_value: float
    p50_post_value: float
    p95_post_value: float
    ruin_probability_pct: float = Field(
        description="Share of paths whose end value gives runway < 12 months."
    )


class VaRReport(BaseModel):
    portfolio_value_inr: float
    historical_window_days: int
    var_95_1d: VaRMetrics
    var_99_1d: VaRMetrics
    monte_carlo: MonteCarloMetrics
    assets_with_history: list[str]
    assets_synthetic: list[str] = Field(
        description="Assets without fetched history; treated as zero-return cash-like in the matrix."
    )
