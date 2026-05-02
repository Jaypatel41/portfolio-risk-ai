"""Tests for Task 4's shock application. The LLM call is mocked — we only
exercise the math layer that turns a parsed StressScenario into a re-run risk
report. Network and LLM calls are out of scope here."""

from __future__ import annotations

from timecell.models import AssetShock, Portfolio, StressScenario
from timecell.stress import apply_shocks

BASE = Portfolio.model_validate({
    "total_value_inr": 10_000_000,
    "monthly_expenses_inr": 80_000,
    "assets": [
        {"name": "BTC", "allocation_pct": 30, "expected_crash_pct": -80},
        {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
        {"name": "GOLD", "allocation_pct": 20, "expected_crash_pct": -15},
        {"name": "CASH", "allocation_pct": 10, "expected_crash_pct": 0},
    ],
})


def test_shock_overrides_expected_crash_for_named_asset() -> None:
    stress = StressScenario(
        rationale="user said BTC drops 70%",
        shocks=[AssetShock(asset_name="BTC", shock_pct=-70)],
    )
    out = apply_shocks(BASE, stress)
    btc = next(a for a in out.assets if a.name == "BTC")
    assert btc.expected_crash_pct == -70


def test_unmentioned_assets_keep_their_original_crash_pct() -> None:
    stress = StressScenario(
        rationale="only BTC is shocked",
        shocks=[AssetShock(asset_name="BTC", shock_pct=-70)],
    )
    out = apply_shocks(BASE, stress)
    nifty = next(a for a in out.assets if a.name == "NIFTY50")
    gold = next(a for a in out.assets if a.name == "GOLD")
    assert nifty.expected_crash_pct == -40  # unchanged
    assert gold.expected_crash_pct == -15  # unchanged


def test_rallies_clamp_to_zero_for_crash_survival_framing() -> None:
    """A +20% rally is not 'free upside' for the crash-survival math; clamp to 0."""
    stress = StressScenario(
        rationale="gold rallies",
        shocks=[AssetShock(asset_name="GOLD", shock_pct=20)],
    )
    out = apply_shocks(BASE, stress)
    gold = next(a for a in out.assets if a.name == "GOLD")
    assert gold.expected_crash_pct == 0


def test_asset_name_match_is_case_insensitive() -> None:
    stress = StressScenario(
        rationale="lowercased input",
        shocks=[AssetShock(asset_name="btc", shock_pct=-50)],
    )
    out = apply_shocks(BASE, stress)
    btc = next(a for a in out.assets if a.name == "BTC")
    assert btc.expected_crash_pct == -50


def test_apply_shocks_does_not_mutate_input() -> None:
    """Pure function — original portfolio must be untouched."""
    stress = StressScenario(
        rationale="immutability check",
        shocks=[AssetShock(asset_name="BTC", shock_pct=-90)],
    )
    apply_shocks(BASE, stress)
    btc = next(a for a in BASE.assets if a.name == "BTC")
    assert btc.expected_crash_pct == -80  # original value preserved
