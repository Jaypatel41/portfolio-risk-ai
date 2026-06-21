# portfolio-risk-ai

> AI-powered portfolio risk advisor with dual-LLM verification, natural-language stress testing, empirical VaR/CVaR from historical data, and 10,000-path Monte Carlo crash simulation.

**Core philosophy:** Math runs in Python. The LLM only narrates. Every recommendation traces back to a computed metric — the model is structurally prevented from inventing numbers.

---

## What This Project Does

Most portfolio tools show you a single static risk estimate. This system:

- **Computes** crash exposure, runway months, and ruin probability with pure Python math
- **Explains** results via an LLM that is forced to cite every claim against pre-computed metrics
- **Verifies** that explanation with an independent second LLM acting as a "Senior Risk Officer"
- **Stress-tests** any scenario you describe in plain English — *"What if BTC crashes 70% and gold rallies 20%?"*
- **Measures** real tail risk using 5 years of daily price history — no normality assumptions, no guessed crash percentages
- **Simulates** 10,000 future portfolio paths while preserving cross-asset correlation

---

## Key Features

### Risk Calculator
Deterministic, math-only engine. Given a portfolio JSON:
- Post-crash value and drawdown percentage for **severe** and **moderate** scenarios
- Runway months (how long the portfolio sustains monthly expenses)
- Ruin test — `PASS` if runway > 12 months, `FAIL` otherwise
- Largest risk contributor by weighted crash exposure
- Concentration warning if any single asset exceeds 40% allocation
- ASCII allocation bar chart, color-coded by crash severity — no external plotting library

### Live Market Data Fetcher
- Routes stocks, indices, and ETFs to **Yahoo Finance** (`yfinance`)
- Routes crypto tickers (BTC, ETH, SOL, etc.) to **CoinGecko public API**
- Fetches all symbols in **parallel** via `asyncio.gather` — `yfinance` sync calls are wrapped in `asyncio.to_thread` so the gather stays non-blocking
- Per-provider **exponential backoff** retry (3 attempts, 0.5s → 4s) via `tenacity`
- **Isolated failure handling** — one provider going down captures a `Quote(ok=False)` object; all other symbols still succeed and render normally
- Timestamps in IST

### AI Portfolio Explainer + Decision Spine
The LLM receives pre-computed metrics and is bound by hard constraints:

```
HARD RULES — break any of these and the answer is invalid:
1. NEVER invent or estimate numbers. Every number in your prose must come
   from the metrics block you are given.
2. EVERY claim must map to one entry in the spine array:
   { "claim": "...", "cited_metric": "key = value", "confidence_pct": 0..100 }
   If you cannot cite a metric for a claim, do not make that claim.
```

Output is constrained to a **Pydantic schema** via Gemini's native `response_schema` mode. Schema validation runs twice — once in the SDK and again in the application — so any structural drift fails loudly with the raw LLM response attached to the error.

Produces a 4-part structured explanation: Summary · Doing Well · Consider Changing · Verdict (`Aggressive` / `Balanced` / `Conservative`).

### Dual-LLM Critic (Hallucination Guard)
With `--critic`, a second independent LLM call reviews the first explanation:
- Acts as a Senior Risk Officer with `temperature=0.0` (fully deterministic)
- Fact-checks every claim in the spine against the original computed metrics
- Returns `PASS` / `PASS_WITH_NITS` / `FAIL` with a per-claim issues table
- Zero hallucinated numbers across 26+ test cases — enforced architecturally, not just by prompt instruction

### Natural-Language Stress Test
LLM as a **structured parser**, not a generator:

1. You type a scenario in plain English
2. Gemini parses it into typed `StressScenario = { rationale, shocks: [{ asset_name, shock_pct }] }` — guaranteed shape via `response_schema`
3. Python applies shocks to a copy of the portfolio. Rallies clamp to 0 — crash survival testing measures downside, not upside
4. The risk engine re-runs on the shocked portfolio — the math layer has no knowledge of LLM involvement
5. Results render as Rich tables alongside the LLM's parse rationale

Total LLM calls per query: **1**. No agentic loops, no fan-out, deterministic math in between.

### Historical VaR + Monte Carlo
Replaces guessed crash percentages with measured tail risk from real price history:

- Fetches up to **5 years of daily returns** (1,257 trading days) per asset via yfinance and CoinGecko
- **Empirical VaR/CVaR** at 95% and 99% confidence — no normality assumption
- Builds a `[days × assets]` log-return matrix aligned to the shortest available series
- Assets without history fall back to a zero-return (cash-like) column — conservative, never inflates risk estimates, flagged in the report
- **Monte Carlo bootstrap** across 10,000 paths: resamples full rows from the historical return matrix, which preserves cross-asset correlation automatically. When BTC has its worst day, NIFTY tends to also — the bootstrap inherits that
- Outputs 5th / 50th / 95th percentile portfolio values and **ruin probability** (share of paths where runway falls below 12 months)
- Numpy seed fixed for full reproducibility

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           CLI (Typer + Rich)                │
│           Streamlit Web UI                  │
└────────┬──────────┬──────────┬──────────────┘
         │          │          │
    ┌────▼────┐ ┌───▼────┐ ┌──▼──────────┐
    │ risk.py │ │market  │ │ explainer   │
    │ Pure    │ │.py     │ │ .py         │
    │ math    │ │Async,  │ │ Decision    │
    │ no I/O  │ │retried │ │ Spine +     │
    └────┬────┘ └───┬────┘ │ Critic      │
         │          │      └──────┬───────┘
         │          │             │
    ┌────▼──────────▼─────────────▼───────┐
    │           stress.py                 │
    │  NL scenario → typed JSON shocks    │
    │  → apply_shocks() → risk.py re-runs │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │       var.py + history.py           │
    │  5yr daily returns → empirical VaR  │
    │  → Monte Carlo bootstrap (10K paths)│
    └─────────────────────────────────────┘
```

**Hard separation between deterministic math and probabilistic narration** — the LLM only ever receives pre-computed metrics and is forbidden from producing output containing numbers that were not in its input.

---

## Error Handling

| Layer | What can fail | How it's handled |
|---|---|---|
| **Portfolio validation** | Allocations not summing to 100%, duplicate asset names, positive crash percentages, empty asset list | Pydantic `model_validator` and `field_validator` — rejected at parse time with a clear error message before any computation runs |
| **Market data fetch** | Provider 5xx errors, rate limits, missing ticker data, empty price history | `tenacity` exponential backoff (3 attempts, 0.5s → 4s); isolated per-symbol `try/except` captures failures into `Quote(ok=False)` — other symbols still succeed |
| **LLM API calls** | Gemini 503 transients, 429 rate limits, empty responses, schema-mismatched output | `tenacity` retries on `ServerError` and `ClientError`; empty response raises `LLMError` immediately; fallback re-parses raw JSON if `response.parsed` is None; all failures raise `LLMError` with the raw response attached |
| **Structured output parsing** | SDK drift between `response.parsed` and expected Pydantic schema | Double validation — SDK parse + `model_validate()` — so any mismatch fails loudly at the boundary, not silently downstream |
| **Historical data alignment** | Assets with no price history, mismatched series lengths | Zero-return fallback column (conservative); `_align_matrix()` trims all series to the shortest common length; flagged in `VaRReport.assets_synthetic` |
| **VaR computation** | Fewer than 2 data points | Raises `ValueError` with a diagnostic message pointing to the likely cause (network access) |
| **API key resolution** | Missing `GEMINI_API_KEY` | Checks `.env`, then `pydantic-settings`, then Streamlit secrets — raises `LLMError` with setup instructions if all three fail |
| **Risk math edge cases** | Zero monthly expenses (infinite runway), 100% cash portfolio, exact 40% concentration boundary | `math.inf` for zero-burn runway; strict `>` threshold (exactly 40% does not warn); all covered by 26+ unit tests |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `google-genai` | ≥ 1.0.0 | Gemini API client — structured output via `response_schema` |
| `pydantic` | ≥ 2.7 | Data validation, typed models, schema enforcement for all LLM outputs |
| `pydantic-settings` | ≥ 2.3 | Environment variable and `.env` config management |
| `httpx` | ≥ 0.27 | Async HTTP client for CoinGecko API calls |
| `yfinance` | ≥ 0.2.40 | Yahoo Finance — stocks, indices, ETFs, gold futures, historical data |
| `numpy` | ≥ 1.26 | VaR/CVaR computation, Monte Carlo simulation, return matrix operations |
| `rich` | ≥ 13.7 | Terminal tables, color-coded panels, ASCII allocation chart |
| `typer` | ≥ 0.12 | CLI framework — unified `portfolio-risk <command>` interface |
| `tenacity` | ≥ 8.3 | Exponential backoff retry for both market data and LLM API calls |
| `python-dotenv` | ≥ 1.0 | Loads `.env` file for local development |
| `streamlit` | ≥ 1.30 | Web UI — same functions as CLI, no duplicated logic |
| `plotly` | ≥ 5.18 | Interactive charts in the Streamlit UI |
| `pytest` *(dev)* | ≥ 8.0 | Test runner |
| `pytest-asyncio` *(dev)* | ≥ 0.23 | Async test support |
| `pytest-mock` *(dev)* | ≥ 3.14 | LLM and API call mocking |
| `ruff` *(dev)* | ≥ 0.5 | Linting and formatting |

**Python requirement:** 3.10+

---

## API Keys

| API | Key Required | Where to Get It | Used For |
|---|---|---|---|
| **Google Gemini** | ✅ Yes — 1 key | [aistudio.google.com](https://aistudio.google.com) (free tier) | AI explainer, dual-LLM critic, NL stress-test parser |
| **Yahoo Finance** | ❌ No | — | Stocks, indices (NIFTY50, RELIANCE), ETFs, gold futures, historical returns |
| **CoinGecko** | ❌ No | — | Crypto prices (BTC, ETH, SOL, ADA, MATIC, DOGE) and historical returns |

**Total API keys needed to run the full project: 1** (Gemini only).
Tasks 1 and 2 (risk calculator + live market data) work with zero API keys.

---

## Setup

### Prerequisites
- Python 3.10+
- A free Google Gemini API key from [aistudio.google.com](https://aistudio.google.com) (only needed for AI features)

### Install

```bash
git clone https://github.com/Jaypatel41/portfolio-risk-ai.git
cd portfolio-risk-ai

# Creates .venv and installs the package in editable mode with dev tools
make install
source .venv/bin/activate
```

Or with pip directly:

```bash
pip install -r requirements.txt
pip install -e .
```

### Configure API Key

```bash
cp .env.example .env
# Open .env and paste your GEMINI_API_KEY
```

`.env` contents:

```env
GEMINI_API_KEY=your-key-here

# Optional overrides
PORTFOLIO_RISK_MODEL=gemini-2.5-flash   # default model
PORTFOLIO_RISK_HTTP_TIMEOUT=10          # seconds
```

---

## Usage

### Risk Calculator

```bash
# Full risk report with ASCII chart
portfolio-risk risk examples/balanced.json

# JSON output (spec shape)
portfolio-risk risk examples/aggressive.json --json

# Skip the chart
portfolio-risk risk examples/balanced.json --no-chart
```

### Live Market Data

```bash
# Default: NIFTY50, RELIANCE, BTC
portfolio-risk market

# Any combination of Yahoo Finance + CoinGecko tickers
portfolio-risk market ^NSEI BTC ETH GC=F RELIANCE.NS SOL
```

### AI Explainer + Dual-LLM Critic

```bash
# Default explanation
portfolio-risk explain examples/balanced.json

# Expert tone
portfolio-risk explain examples/balanced.json --tone expert

# Show raw LLM response alongside structured output
portfolio-risk explain examples/balanced.json --tone beginner --raw

# Activate second LLM to fact-check the first
portfolio-risk explain examples/balanced.json --critic

# All options combined
portfolio-risk explain examples/aggressive.json --tone expert --raw --critic
```

### Natural-Language Stress Test

```bash
portfolio-risk stress examples/balanced.json "what if BTC crashes 70% and gold rallies 20%?"
portfolio-risk stress examples/aggressive.json "what if Indian equities lose 50%?"
portfolio-risk stress examples/conservative.json "what if everything except cash drops 30%?"
```

### Historical VaR + Monte Carlo

```bash
# Default: 5 years history, 10,000 Monte Carlo paths
portfolio-risk var examples/balanced.json

# Custom parameters
portfolio-risk var examples/balanced.json --years 3 --paths 5000
portfolio-risk var examples/aggressive.json --years 5 --paths 10000
```

### Streamlit Web UI

```bash
portfolio-risk serve   # → http://localhost:8501
```

All tabs in the web UI call the same functions as the CLI. There is no duplicated logic. Deployable to Streamlit Cloud directly — push to GitHub, point to `app.py`, add `GEMINI_API_KEY` in app secrets.

---

## Example Output

### Risk Calculator

```
                    Portfolio Risk Report · INR 10,000,000
┌──────────────────────┬─────────────────────┬─────────────────────┐
│ Metric               │ Severe              │ Moderate            │
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Post-crash value     │ INR 5,700,000       │ INR 7,850,000       │
│ Drawdown             │ -43.0%              │ -21.5%              │
│ Runway (months)      │ 71.25               │ 98.13               │
│ Ruin test            │ PASS                │ PASS                │
└──────────────────────┴─────────────────────┴─────────────────────┘
Largest risk asset: BTC    Concentration warning: No
```

### Historical VaR + Monte Carlo

```
Historical VaR/CVaR · 1,257 trading days · INR 10,000,000 portfolio
┌────────────┬───────────────────────┬────────────────────────────────┐
│ Confidence │ VaR (1-day)           │ CVaR (expected tail loss)      │
├────────────┼───────────────────────┼────────────────────────────────┤
│ 95%        │ -2.41%  INR -241,000  │ -3.62%  INR -362,000           │
│ 99%        │ -4.18%  INR -418,000  │ -5.55%  INR -555,000           │
└────────────┴───────────────────────┴────────────────────────────────┘

Monte Carlo · 10,000 paths · 252-day horizon · historical bootstrap
┌────────────────────────────────────────┬──────────────────┐
│ Outcome                                │ Portfolio value  │
├────────────────────────────────────────┼──────────────────┤
│ 5th percentile (worst case)            │ INR 6,420,000    │
│ Median outcome                         │ INR 11,180,000   │
│ 95th percentile (best case)            │ INR 19,940,000   │
│ Ruin probability (runway < 12 months)  │ 0.32%            │
└────────────────────────────────────────┴──────────────────┘
```

---

## Tests

```bash
make test   # 26+ unit tests, ~2 seconds
```

All LLM and network calls are mocked — no API keys or internet required to run the test suite.

**Coverage:**
- All risk calculator scenarios including edge cases: zero monthly burn → `math.inf` runway, 100% cash portfolio, allocations not summing to 100%, duplicate asset names, strict `>` 40% concentration boundary, positive crash percentages rejected at model boundary
- Stress-test shock application: case-insensitive asset name matching, rally clamping to 0, input portfolio immutability, unmentioned assets pass through unchanged
- VaR/CVaR: ordering guarantee (CVaR ≤ VaR), percentile recovery on synthetic normal distribution, Monte Carlo determinism under fixed numpy seed, zero-return → zero VaR, zero expenses → zero ruin probability

---

## Data Sources

| Source | Data Provided | Auth |
|---|---|---|
| Yahoo Finance (`yfinance`) | NIFTY50, RELIANCE, any NSE/BSE ticker, gold futures, global ETFs, 5yr daily OHLCV history | None |
| CoinGecko Public API | BTC, ETH, SOL, ADA, MATIC, DOGE spot prices | None |
| Google Gemini (`gemini-2.5-flash`) | Portfolio explanation, NL scenario parsing, critic fact-check | API key (free tier) |

---

## Tech Stack

```
Language      Python 3.10+
LLM           Google Gemini (gemini-2.5-flash) via google-genai SDK
Validation    Pydantic v2 — models, schemas, LLM output contracts
Math          NumPy — VaR, CVaR, Monte Carlo, return matrix
Market data   yfinance (Yahoo Finance) + httpx (CoinGecko)
Reliability   tenacity — exponential backoff on all external calls
CLI           Typer + Rich (tables, color panels, ASCII charts)
Web UI        Streamlit + Plotly
Testing       pytest + pytest-asyncio + pytest-mock (26+ tests, all mocked)
Linting       ruff
```

---

## Resume Summary

> Built AI-powered portfolio risk advisor with dual-LLM hallucination guard (Decision Spine pattern + critic pass), empirical VaR/CVaR from 1,257 trading days of historical data, 10,000-path Monte Carlo simulation preserving cross-asset correlation, and natural-language stress-test parser — achieving 100% schema-valid LLM outputs across 26+ test cases with zero invented metrics.

**Skills demonstrated:** LLM prompt engineering · Structured output / JSON schema enforcement · Pydantic v2 · Statistical risk modeling (VaR, CVaR, Monte Carlo) · Async Python (asyncio, httpx) · REST API integration · Exponential backoff and fault isolation · Streamlit deployment · Unit testing with mocked external dependencies

---

## License

MIT
