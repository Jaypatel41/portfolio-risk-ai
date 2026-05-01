"""Timecell — Streamlit web UI.

Wraps the same library functions the CLI calls — no duplicate logic.
Four tabs: 📊 Risk · 💱 Market · 🤖 Explainer · 🔥 Stress-Test.

Run locally:   streamlit run app.py     (or: timecell serve)
Streamlit Cloud: deploy this repo; set GEMINI_API_KEY in Cloud secrets.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from timecell import __version__
from timecell.ai_client import LLMError
from timecell.explainer import critique_explanation, explain_portfolio
from timecell.market import DEFAULT_SYMBOLS, fetch_quotes
from timecell.models import Portfolio
from timecell.risk import compute_risk_metrics, compute_risk_report
from timecell.stress import run_stress_test

load_dotenv()

# Surface Streamlit Cloud secret as env var so ai_client picks it up.
if "GEMINI_API_KEY" not in os.environ:
    with contextlib.suppress(KeyError, FileNotFoundError, AttributeError):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]


st.set_page_config(
    page_title="Timecell · Family-Office Risk",
    page_icon="🛟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* base */
      .stApp { background: #0b0f14; }
      h1, h2, h3 { color: #e6edf3 !important; font-family: 'JetBrains Mono', 'SF Mono', monospace; }
      .timecell-tag { color: #7ee787; font-family: monospace; font-size: 0.85rem; letter-spacing: 0.04em; }

      /* sidebar — give the input form its own visual identity */
      section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
      section[data-testid="stSidebar"] .stRadio label,
      section[data-testid="stSidebar"] label { color: #c9d1d9 !important; font-size: 0.85rem; }

      /* number inputs — dark surface, cyan accent on focus */
      div[data-testid="stNumberInput"] input,
      div[data-testid="stTextInput"] input,
      .stTextArea textarea {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #21262d !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
        font-size: 0.92rem !important;
        transition: border-color 0.15s, box-shadow 0.15s;
      }
      div[data-testid="stNumberInput"] input:focus,
      div[data-testid="stTextInput"] input:focus,
      .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.18) !important;
        outline: none !important;
      }
      div[data-testid="stNumberInput"] button { background: #21262d !important; color: #c9d1d9 !important; border: none !important; }
      div[data-testid="stNumberInput"] button:hover { background: #30363d !important; color: #58a6ff !important; }

      /* data_editor (assets table) — dark cells, cyan headers */
      div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        background: #0d1117 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
      }

      /* radio — selected pill */
      div[role="radiogroup"] label[data-checked="true"] {
        background: #21262d; border-radius: 6px; padding: 2px 8px;
      }

      /* file uploader — drag area */
      [data-testid="stFileUploader"] section {
        background: #161b22 !important;
        border: 1px dashed #30363d !important;
        border-radius: 8px !important;
      }
      [data-testid="stFileUploader"] section:hover { border-color: #58a6ff !important; }

      /* tabs */
      .stTabs [data-baseweb="tab"] { background: #161b22; border-radius: 6px 6px 0 0; padding: 8px 14px; }
      .stTabs [aria-selected="true"] { background: #21262d; color: #7ee787 !important; }

      /* verdict colors */
      .verdict-Aggressive { color: #f85149; font-weight: 600; }
      .verdict-Balanced { color: #d29922; font-weight: 600; }
      .verdict-Conservative { color: #7ee787; font-weight: 600; }

      /* validation chips */
      .ok-chip   { background: rgba(126,231,135,0.10); color: #7ee787; padding: 4px 10px; border-radius: 12px;
                   border: 1px solid rgba(126,231,135,0.35); font-family: monospace; font-size: 0.8rem; display: inline-block; }
      .warn-chip { background: rgba(248,81,73,0.10); color: #f85149; padding: 4px 10px; border-radius: 12px;
                   border: 1px solid rgba(248,81,73,0.35); font-family: monospace; font-size: 0.8rem; display: inline-block; }

      /* primary buttons (Generate explanation, Run stress-test, etc.) */
      button[kind="primary"] {
        background: linear-gradient(180deg, #1f6feb 0%, #1158c7 100%) !important;
        border: 1px solid #1f6feb !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
      }
      button[kind="primary"]:hover { background: linear-gradient(180deg, #388bfd 0%, #1f6feb 100%) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helpers ───────────────────────────────────────────────────────────────


@st.cache_data
def _load_examples() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted((Path(__file__).parent / "examples").glob("*.json")):
        out[p.stem.replace("_", " ").title()] = json.loads(p.read_text())
    return out


def _format_inr(v: float) -> str:
    return f"₹{v:,.0f}"


def _format_runway(months: float) -> str:
    if math.isinf(months):
        return "∞ (no burn)"
    if months >= 24:
        return f"{months:,.1f} mo · {months / 12:.1f} yrs"
    return f"{months:,.1f} months"


def _has_llm_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


# ── sidebar — portfolio source + global controls ─────────────────────────


with st.sidebar:
    st.markdown(
        f"<h2 style='margin-bottom:0'>🛟 Timecell</h2>"
        f"<div class='timecell-tag'>v{__version__} · math in Python · narration by Gemini</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    examples = _load_examples()
    src = st.radio(
        "Portfolio source",
        ["📋 Python dict / JSON", "🛠 Build (form)", "📁 Upload .json"],
        index=0,
        help="The assignment brief specifies a Python dictionary. JSON is the same shape — that's the default.",
    )

    raw_portfolio: dict[str, Any] | None = None

    if src == "🛠 Build (form)":
        # ── Quick-load: pre-fill the form from one of the examples ──────────
        st.markdown("**Quick-load**")
        choice = st.selectbox(
            "Quick-load example",
            list(examples.keys()),
            label_visibility="collapsed",
            key="example_choice",
        )
        # Each example gets its own widget keyspace so switching = fresh form.
        # (Avoids stale state when the user picks a different starting point.)
        ex = examples[choice]
        ns = choice.replace(" ", "_").lower()  # "Balanced" → "balanced"

        st.markdown("**Top-line numbers**")
        c1, c2 = st.columns(2)
        with c1:
            total_value = st.number_input(
                "Total value (INR)",
                min_value=1.0,
                value=float(ex["total_value_inr"]),
                step=100_000.0,
                format="%.0f",
                key=f"tv_{ns}",
            )
        with c2:
            monthly = st.number_input(
                "Monthly expenses (INR)",
                min_value=0.0,
                value=float(ex["monthly_expenses_inr"]),
                step=10_000.0,
                format="%.0f",
                key=f"me_{ns}",
            )

        st.markdown("**Assets** &nbsp;<span style='color:#8b949e;font-size:0.8rem'>(double-click cells; +/− rows on hover)</span>", unsafe_allow_html=True)
        edited_assets = st.data_editor(
            ex["assets"],
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key=f"assets_{ns}",
            column_config={
                "name": st.column_config.TextColumn(
                    "Asset",
                    help="e.g. BTC, NIFTY50, GOLD",
                    required=True,
                    max_chars=24,
                ),
                "allocation_pct": st.column_config.NumberColumn(
                    "Alloc %",
                    help="0–100. Must sum to 100 across all rows.",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    format="%.1f",
                ),
                "expected_crash_pct": st.column_config.NumberColumn(
                    "Crash %",
                    help="−100 to 0. The expected drawdown for this asset in a severe crash.",
                    min_value=-100.0,
                    max_value=0.0,
                    step=5.0,
                    format="%.0f",
                ),
            },
        )

        # ── live validation badge ─────────────────────────────────────────
        valid_rows = [
            a for a in edited_assets
            if a.get("name") and a.get("allocation_pct") is not None
        ]
        total_alloc = sum(a["allocation_pct"] for a in valid_rows)
        names = [a["name"] for a in valid_rows]
        is_balanced = abs(total_alloc - 100) < 0.01
        no_dupes = len(set(names)) == len(names)

        chips = []
        chips.append(
            f"<span class='{'ok-chip' if is_balanced else 'warn-chip'}'>Σ allocation = {total_alloc:.1f}%</span>"
        )
        chips.append(
            f"<span class='{'ok-chip' if no_dupes else 'warn-chip'}'>{len(valid_rows)} assets · "
            f"{'no dupes' if no_dupes else 'dupes!'}</span>"
        )
        st.markdown(" ".join(chips), unsafe_allow_html=True)

        if is_balanced and no_dupes and valid_rows:
            raw_portfolio = {
                "total_value_inr": total_value,
                "monthly_expenses_inr": monthly,
                "assets": valid_rows,
            }
        else:
            st.caption(
                "Fix the warnings above to compute risk metrics. Allocation must sum to 100% "
                "and asset names must be unique."
            )

    elif src == "📋 Python dict / JSON":
        st.caption(
            "This is the **exact input format the assignment brief specifies** — a Python "
            "dict with `total_value_inr`, `monthly_expenses_inr`, and `assets[]`. "
            "Edit the values below or paste your own."
        )
        choice = st.selectbox(
            "Pre-fill from example",
            list(examples.keys()),
            label_visibility="collapsed",
            key="json_example_choice",
        )
        prefilled = json.dumps(examples[choice], indent=2)
        pasted = st.text_area(
            "Portfolio (Python dict / JSON)",
            value=prefilled,
            height=380,
            key=f"json_text_{choice}",  # re-prefill when example changes
            label_visibility="collapsed",
        )
        if pasted.strip():
            try:
                raw_portfolio = json.loads(pasted)
                st.markdown(
                    "<span class='ok-chip'>✓ Valid JSON · matches brief format</span>",
                    unsafe_allow_html=True,
                )
            except json.JSONDecodeError as exc:
                st.markdown(
                    f"<span class='warn-chip'>⚠ JSON parse error: {exc}</span>",
                    unsafe_allow_html=True,
                )

    else:  # Upload
        uploaded = st.file_uploader("Upload portfolio.json", type=["json"])
        if uploaded:
            try:
                raw_portfolio = json.loads(uploaded.read())
            except json.JSONDecodeError as exc:
                st.error(f"JSON parse error: {exc}")

    st.divider()
    st.subheader("AI controls")
    tone = st.selectbox("Tone", ["beginner", "experienced", "expert"], index=1)
    use_critic = st.toggle("Run self-critic pass", value=False)
    st.caption(
        f"LLM: {'🟢 Google Gemini' if _has_llm_key() else '❌ GEMINI_API_KEY not set'}"
    )


# ── main ──────────────────────────────────────────────────────────────────


st.title("Family-Office Decision Support")
st.markdown(
    "<span class='timecell-tag'>Math runs in Python. Gemini only narrates. Decisions stay yours.</span>",
    unsafe_allow_html=True,
)

if not raw_portfolio:
    st.info("Pick or paste a portfolio in the sidebar to begin.")
    st.stop()

try:
    portfolio = Portfolio.model_validate(raw_portfolio)
except Exception as exc:  # noqa: BLE001
    st.error(f"Invalid portfolio: {exc}")
    st.stop()

report = compute_risk_report(portfolio)


# Top metric strip
m1, m2, m3, m4 = st.columns(4)
m1.metric("Portfolio value", _format_inr(report.portfolio_value_inr))
m2.metric(
    "Severe-crash drawdown",
    f"{report.severe.drawdown_pct:.1f}%",
    delta=f"moderate {report.moderate.drawdown_pct:.1f}%",
    delta_color="off",
)
m3.metric(
    "Severe-crash runway",
    _format_runway(report.severe.runway_months),
    delta=report.severe.ruin_test,
    delta_color="normal" if report.severe.ruin_test == "PASS" else "inverse",
)
m4.metric("Largest risk asset", report.largest_risk_asset)
st.divider()


tabs = st.tabs(["📊 Risk Report", "💱 Live Market", "🤖 AI Explainer", "🔥 Stress-Test"])


# ── Tab 1: Risk Report ────────────────────────────────────────────────────

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Allocation")
        alloc_fig = go.Figure(
            data=[go.Bar(
                x=[a.allocation_pct for a in portfolio.assets],
                y=[a.name for a in portfolio.assets],
                orientation="h",
                text=[f"{a.allocation_pct:.0f}% · crash {a.expected_crash_pct:.0f}%" for a in portfolio.assets],
                textposition="outside",
                marker={"color": "#58a6ff"},
            )]
        )
        alloc_fig.update_layout(
            paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
            font={"color": "#e6edf3"},
            xaxis={"range": [0, 110], "title": "%"},
            yaxis={"autorange": "reversed"},
            height=320, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(alloc_fig, use_container_width=True)

        if report.concentration_warning:
            st.warning(
                f"⚠ Concentration warning: top asset is **{report.concentration_top_pct:.1f}%** of book "
                f"(threshold > 40%)."
            )
        else:
            st.success(f"✓ Concentration OK · top asset {report.concentration_top_pct:.1f}%")

    with c2:
        st.subheader("Severe vs Moderate")
        scenario_data = {
            "Metric": ["Post-crash value", "Drawdown", "Runway", "Ruin test"],
            "Severe": [
                _format_inr(report.severe.post_crash_value),
                f"{report.severe.drawdown_pct:.1f}%",
                _format_runway(report.severe.runway_months),
                report.severe.ruin_test,
            ],
            "Moderate (50% severity)": [
                _format_inr(report.moderate.post_crash_value),
                f"{report.moderate.drawdown_pct:.1f}%",
                _format_runway(report.moderate.runway_months),
                report.moderate.ruin_test,
            ],
        }
        st.dataframe(scenario_data, hide_index=True, use_container_width=True)

        with st.expander("Spec-shape `compute_risk_metrics(portfolio)` →"):
            st.json(compute_risk_metrics(portfolio))


# ── Tab 2: Live Market ────────────────────────────────────────────────────

with tabs[1]:
    st.subheader("Live market snapshot")
    syms_in = st.text_input(
        "Symbols (comma-separated)",
        value=",".join(DEFAULT_SYMBOLS),
        help="Yahoo for stocks/indices (^NSEI, RELIANCE.NS, GLDM); CoinGecko for crypto (BTC, ETH, SOL).",
    )
    if st.button("Fetch live prices", type="primary"):
        with st.spinner("Fetching in parallel…"):
            symbols = [s.strip() for s in syms_in.split(",") if s.strip()]
            quotes = asyncio.run(fetch_quotes(symbols))
        rows = [
            {
                "Asset": q.asset,
                "Price": (f"{q.price:,.2f}" if q.ok and q.price else "—"),
                "Currency": q.currency,
                "Provider": q.provider,
                "Status": "OK" if q.ok else (q.error or "FAIL")[:40],
            }
            for q in quotes
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        if any(not q.ok for q in quotes):
            st.warning("Some providers failed; failure is isolated per quote — others succeeded.")


# ── Tab 3: AI Explainer ───────────────────────────────────────────────────

with tabs[2]:
    if not _has_llm_key():
        st.error("Set `GEMINI_API_KEY` in `.env` (local) or Streamlit secrets (Cloud) to enable AI features.")
    elif st.button("Generate explanation", type="primary", key="explain_btn"):
        with st.spinner("Gemini is narrating the math…"):
            try:
                explanation, raw_text = explain_portfolio(portfolio, tone=tone, report=report)
            except LLMError as exc:
                st.error(str(exc))
                st.stop()

        v = explanation.verdict
        st.markdown(
            f"### Verdict: <span class='verdict-{v}'>{v}</span> "
            f"<span style='color:#8b949e'>· tone: {explanation.tone}</span>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Summary")
            st.write(explanation.summary)
            st.markdown("#### ✓ Doing well")
            st.success(explanation.doing_well)
            st.markdown("#### ✎ Consider changing")
            st.warning(explanation.consider_changing)
        with c2:
            st.markdown("#### Decision Spine")
            st.caption("Every claim, with the metric that justifies it.")
            spine_rows = [
                {"Claim": s.claim, "Cited metric": s.cited_metric, "Confidence": f"{s.confidence_pct}%"}
                for s in explanation.spine
            ]
            st.dataframe(spine_rows, hide_index=True, use_container_width=True)

        if use_critic:
            with st.spinner("Running self-critic pass…"):
                try:
                    findings = critique_explanation(portfolio, explanation, report=report)
                except LLMError as exc:
                    st.error(f"Critic failed: {exc}")
                    findings = None
            if findings:
                color = {"PASS": "🟢", "PASS_WITH_NITS": "🟡", "FAIL": "🔴"}[findings.overall]
                st.markdown(f"#### Self-critique: {color} {findings.overall}")
                if findings.issues:
                    st.dataframe(
                        [{"Severity": i.severity, "Claim": i.claim, "Problem": i.problem} for i in findings.issues],
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.success("No issues found.")

        with st.expander("Raw LLM response"):
            st.code(raw_text, language="json")


# ── Tab 4: Stress-Test (Task 4 — the showcase) ────────────────────────────

with tabs[3]:
    st.subheader("Natural-language stress-test")
    st.caption(
        "Ask Gemini what would happen to this portfolio under any scenario you can describe in plain English. "
        "Gemini parses your scenario into per-asset shocks; Python re-runs the risk math. "
        "Rallies clamp to 0% loss — we're testing crash survival, not upside."
    )
    if not _has_llm_key():
        st.error("Set `GEMINI_API_KEY` to enable AI features.")
    else:
        scenario = st.text_input(
            "Scenario",
            value="what if BTC crashes 70% and gold rallies 20%?",
            help="Try: 'all crypto goes to zero', 'NIFTY drops 30% and RELIANCE halves', 'a 2008-style crash'.",
        )
        if st.button("Run stress-test", type="primary", key="stress_btn"):
            with st.spinner("Gemini is parsing your scenario…"):
                try:
                    result = run_stress_test(portfolio, scenario)
                except LLMError as exc:
                    st.error(str(exc))
                    st.stop()

            parsed = result["parsed"]
            stress_report = result["report"]

            st.info(f"**Gemini's parse:** {parsed.rationale}")

            shock_rows = []
            for s in parsed.shocks:
                sign = "+" if s.shock_pct > 0 else ""
                shock_rows.append({"Asset": s.asset_name, "Shock": f"{sign}{s.shock_pct:.1f}%"})
            st.dataframe(shock_rows, hide_index=True, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Post-crash value", _format_inr(stress_report.severe.post_crash_value))
            c2.metric(
                "Drawdown",
                f"{stress_report.severe.drawdown_pct:.1f}%",
                delta=f"baseline {report.severe.drawdown_pct:.1f}%",
                delta_color="inverse",
            )
            c3.metric(
                "Runway",
                _format_runway(stress_report.severe.runway_months),
                delta=stress_report.severe.ruin_test,
                delta_color="normal" if stress_report.severe.ruin_test == "PASS" else "inverse",
            )

            with st.expander("How the scenario was applied"):
                st.write(
                    "Gemini returned a structured `StressScenario` (asset_name + shock_pct). "
                    "Python overwrote each shocked asset's `expected_crash_pct`, clamped rallies to 0, "
                    "then re-ran `compute_risk_report()` from Task 1. The LLM never computed any number."
                )


st.divider()
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-family:monospace;font-size:0.8rem'>"
    "Computed in code. Narrated by Gemini. Decisions stay yours."
    "</div>",
    unsafe_allow_html=True,
)
