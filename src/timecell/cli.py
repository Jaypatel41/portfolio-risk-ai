"""Typer-powered CLI. Six commands; one per task plus `serve`.

  timecell risk     <portfolio.json>          # Task 1
  timecell market   [SYMBOL ...]              # Task 2
  timecell explain  <portfolio.json>          # Task 3
  timecell stress   <portfolio.json> "<...>"  # Task 4
  timecell var      <portfolio.json>          # Task 5 — historical VaR + Monte Carlo
  timecell serve                              # localhost UI (http://localhost:8501)
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .ai_client import LLMError
from .explainer import critique_explanation, explain_portfolio
from .history import fetch_returns
from .market import DEFAULT_SYMBOLS, fetch_quotes
from .models import Portfolio, Tone
from .risk import compute_risk_metrics, compute_risk_report, render_allocation_chart
from .stress import run_stress_test
from .var import compute_var_report

load_dotenv()

app = typer.Typer(
    name="timecell",
    help="Family-office portfolio risk: math in Python, narration by Gemini.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _load_portfolio(path: Path) -> Portfolio:
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(2)
    try:
        return Portfolio.model_validate(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValueError) as exc:
        console.print(f"[red]Invalid portfolio JSON:[/red] {exc}")
        raise typer.Exit(2) from exc


def _handle_llm_error(exc: Exception) -> None:
    if isinstance(exc, LLMError):
        console.print(Panel(Text(str(exc), style="red"), title="LLM error", border_style="red"))
        raise typer.Exit(3) from exc
    raise exc


# ── Task 1 ─────────────────────────────────────────────────────────────────


@app.command(help="Task 1 — Compute risk metrics for a portfolio JSON file.")
def risk(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    json_out: bool = typer.Option(False, "--json", help="Print spec-shape dict as JSON."),
    chart: bool = typer.Option(True, "--chart/--no-chart", help="Show allocation bar chart."),
) -> None:
    p = _load_portfolio(portfolio_path)

    if json_out:
        console.print_json(data=compute_risk_metrics(p))
        return

    report = compute_risk_report(p)
    if chart:
        console.print(Panel(render_allocation_chart(p), title="Portfolio", border_style="cyan"))

    table = Table(title="Risk scenarios", show_lines=True)
    table.add_column("Metric")
    table.add_column("Severe", style="red")
    table.add_column("Moderate (50% severity)", style="yellow")
    table.add_row(
        "Post-crash value (INR)",
        f"₹{report.severe.post_crash_value:,.0f}",
        f"₹{report.moderate.post_crash_value:,.0f}",
    )
    table.add_row(
        "Drawdown",
        f"{report.severe.drawdown_pct:.1f}%",
        f"{report.moderate.drawdown_pct:.1f}%",
    )
    table.add_row(
        "Runway (months)",
        f"{report.severe.runway_months}",
        f"{report.moderate.runway_months}",
    )
    table.add_row(
        "Ruin test",
        f"[{'green' if report.severe.ruin_test == 'PASS' else 'red'}]{report.severe.ruin_test}[/]",
        f"[{'green' if report.moderate.ruin_test == 'PASS' else 'red'}]{report.moderate.ruin_test}[/]",
    )
    console.print(table)
    console.print(f"Largest risk asset: [bold]{report.largest_risk_asset}[/bold]")
    if report.concentration_warning:
        console.print(
            f"[yellow]⚠ Concentration warning:[/yellow] top asset is "
            f"{report.concentration_top_pct:.1f}% (threshold > 40%)."
        )
    console.rule("[dim]compute_risk_metrics(portfolio) →[/dim]")
    console.print_json(data=compute_risk_metrics(p))


# ── Task 2 ─────────────────────────────────────────────────────────────────


@app.command(help="Task 2 — Fetch live prices for any symbols (defaults: NIFTY50, RELIANCE, BTC).")
def market(
    symbols: list[str] | None = typer.Argument(None, help="Symbols. Defaults to a stock+index+crypto trio."),
) -> None:
    syms = symbols or DEFAULT_SYMBOLS
    quotes = asyncio.run(fetch_quotes(syms))

    when = quotes[0].fetched_at.strftime("%Y-%m-%d %H:%M:%S IST") if quotes else "—"
    table = Table(title=f"Asset Prices — fetched at {when}", show_lines=False)
    table.add_column("Asset")
    table.add_column("Price", justify="right")
    table.add_column("Currency")
    table.add_column("Provider")
    table.add_column("Status")

    for q in quotes:
        if q.ok and q.price is not None:
            table.add_row(q.asset, f"{q.price:,.2f}", q.currency, q.provider, "[green]OK[/]")
        else:
            table.add_row(q.asset, "—", q.currency, q.provider, f"[red]{(q.error or 'FAIL')[:30]}[/]")
    console.print(table)


# ── Task 3 ─────────────────────────────────────────────────────────────────


@app.command(help="Task 3 — AI-powered plain-English explanation of a portfolio.")
def explain(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    tone: Tone = typer.Option("experienced", "--tone", help="beginner | experienced | expert"),
    raw: bool = typer.Option(False, "--raw", help="Also print the raw LLM response."),
    critic: bool = typer.Option(False, "--critic", help="Run a second LLM call to fact-check."),
) -> None:
    p = _load_portfolio(portfolio_path)
    try:
        explanation, raw_text = explain_portfolio(p, tone=tone)
    except Exception as exc:
        _handle_llm_error(exc)
        return

    if raw:
        console.print(Panel(Text(raw_text, style="dim"), title="Raw LLM response", border_style="dim"))

    verdict_color = {"Aggressive": "red", "Balanced": "yellow", "Conservative": "green"}[explanation.verdict]
    console.print(
        Panel(
            Text(explanation.summary),
            title=f"Summary · tone={explanation.tone} · verdict=[{verdict_color}]{explanation.verdict}[/]",
            border_style="cyan",
        )
    )
    console.print(Panel(Text(explanation.doing_well), title="✓ Doing well", border_style="green"))
    console.print(Panel(Text(explanation.consider_changing), title="✎ Consider changing", border_style="yellow"))

    spine_table = Table(title="Decision Spine — every claim, with the metric that justifies it", show_lines=False)
    spine_table.add_column("Claim")
    spine_table.add_column("Cited metric")
    spine_table.add_column("Confidence", justify="right")
    for s in explanation.spine:
        spine_table.add_row(s.claim, s.cited_metric, f"{s.confidence_pct}%")
    console.print(spine_table)

    if critic:
        try:
            findings = critique_explanation(p, explanation)
        except Exception as exc:
            _handle_llm_error(exc)
            return
        verdict_style = {"PASS": "green", "PASS_WITH_NITS": "yellow", "FAIL": "red"}[findings.overall]
        console.print(Panel(
            Text(f"Critic verdict: {findings.overall}", style=verdict_style),
            title="Self-critique",
            border_style=verdict_style,
        ))
        if findings.issues:
            issue_table = Table(show_lines=False)
            issue_table.add_column("Severity")
            issue_table.add_column("Claim")
            issue_table.add_column("Problem")
            for i in findings.issues:
                issue_table.add_row(i.severity, i.claim, i.problem)
            console.print(issue_table)


# ── Task 4 ─────────────────────────────────────────────────────────────────


@app.command(help="Task 4 — Natural-language stress-test (e.g. 'what if BTC crashes 70%?').")
def stress(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    scenario: str = typer.Argument(..., help="Plain-English what-if scenario, in quotes."),
) -> None:
    p = _load_portfolio(portfolio_path)
    try:
        result = run_stress_test(p, scenario)
    except Exception as exc:
        _handle_llm_error(exc)
        return

    parsed = result["parsed"]
    report = result["report"]

    console.print(Panel(Text(scenario), title="Scenario", border_style="cyan"))
    console.print(Panel(Text(parsed.rationale), title="Gemini's parse", border_style="cyan"))

    shock_table = Table(title="Per-asset shocks")
    shock_table.add_column("Asset")
    shock_table.add_column("Shock", justify="right")
    for s in parsed.shocks:
        sign = "+" if s.shock_pct > 0 else ""
        color = "red" if s.shock_pct < 0 else "green"
        shock_table.add_row(s.asset_name, f"[{color}]{sign}{s.shock_pct:.1f}%[/]")
    console.print(shock_table)

    metrics_table = Table(title="Resulting risk metrics (rallies clamp to 0% — crash-survival framing)")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Severe scenario applied", justify="right")
    metrics_table.add_row("Post-crash value", f"₹{report.severe.post_crash_value:,.0f}")
    metrics_table.add_row("Drawdown", f"{report.severe.drawdown_pct:.1f}%")
    metrics_table.add_row("Runway (months)", f"{report.severe.runway_months}")
    metrics_table.add_row(
        "Ruin test",
        f"[{'green' if report.severe.ruin_test == 'PASS' else 'red'}]{report.severe.ruin_test}[/]",
    )
    metrics_table.add_row("Largest risk asset", report.largest_risk_asset)
    console.print(metrics_table)


# ── Task 5 ─────────────────────────────────────────────────────────────────


@app.command(help="Task 5 — Historical VaR/CVaR + Monte Carlo simulation from real price history.")
def var(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    horizon: int = typer.Option(252, "--horizon", help="Monte Carlo horizon in trading days (252 ≈ 1y)."),
    paths: int = typer.Option(10_000, "--paths", help="Number of Monte Carlo paths."),
    years: int = typer.Option(5, "--years", help="Years of daily history to pull per asset."),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
) -> None:
    p = _load_portfolio(portfolio_path)
    asset_names = [a.name for a in p.assets]
    console.print(
        f"[cyan]Fetching {years}y of daily returns for {len(asset_names)} assets...[/cyan]"
    )
    returns = fetch_returns(asset_names, years=years)

    try:
        report = compute_var_report(p, returns, n_paths=paths, horizon_days=horizon, seed=seed)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    if report.assets_synthetic:
        console.print(
            f"[yellow]⚠ No history fetched for: {', '.join(report.assets_synthetic)} "
            f"(treated as zero-return cash-like — VaR contribution is zero).[/yellow]"
        )

    var_table = Table(
        title=f"Historical VaR/CVaR · {report.historical_window_days} trading days · "
              f"₹{report.portfolio_value_inr:,.0f} portfolio",
        show_lines=False,
    )
    var_table.add_column("Confidence")
    var_table.add_column("VaR (1-day)", justify="right", style="red")
    var_table.add_column("CVaR (1-day, expected tail loss)", justify="right", style="red")
    for v in (report.var_95_1d, report.var_99_1d):
        var_table.add_row(
            f"{v.confidence_pct:.0f}%",
            f"{v.var_pct:.2f}%   ₹{v.var_inr:,.0f}",
            f"{v.cvar_pct:.2f}%   ₹{v.cvar_inr:,.0f}",
        )
    console.print(var_table)

    mc = report.monte_carlo
    mc_table = Table(
        title=f"Monte Carlo · {mc.n_paths:,} paths · {mc.horizon_days}-day horizon · "
              f"historical bootstrap (preserves correlation)",
        show_lines=False,
    )
    mc_table.add_column("Outcome")
    mc_table.add_column("Portfolio value", justify="right")
    mc_table.add_row("5th percentile (worst case)", f"₹{mc.p5_post_value:,.0f}")
    mc_table.add_row("Median outcome", f"₹{mc.p50_post_value:,.0f}")
    mc_table.add_row("95th percentile (best case)", f"₹{mc.p95_post_value:,.0f}")
    ruin_color = "red" if mc.ruin_probability_pct > 5 else "green"
    mc_table.add_row(
        "Ruin probability (runway < 12 months)",
        f"[{ruin_color}]{mc.ruin_probability_pct:.2f}%[/]",
    )
    console.print(mc_table)


# ── Web UI ─────────────────────────────────────────────────────────────────


@app.command(help="Launch the Streamlit web UI on http://localhost:8501")
def serve(
    port: int = typer.Option(8501, "--port", "-p"),
    host: str = typer.Option("localhost", "--host"),
) -> None:
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--browser.gatherUsageStats", "false",
    ]
    console.print(f"[cyan]Starting Timecell web UI on http://{host}:{port}[/cyan]")
    raise typer.Exit(subprocess.call(cmd))


@app.callback(invoke_without_command=False)
def _root(version: bool = typer.Option(False, "--version", "-v")) -> None:
    if version:
        console.print(f"timecell {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
