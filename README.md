# Timecell Internship Project — Portfolio Risk & AI Assessment

**Jay Patel** · Submission for the Timecell.ai engineering intern technical assessment.
**All 4 tasks attempted · CLI + Streamlit web UI · production quality · plus one extra bonus.**

A decision-support tool for family-office portfolio risk. **Math runs in Python; Gemini only narrates.** This is the philosophy stated on [timecell.ai](https://timecell.ai): *"Computed in code, not guessed by a language model."*

---

## Project Overview & Structure

The repository is delivered as a single Python package, [src/timecell/](src/timecell/), with a unified Typer-powered CLI (`timecell <command>`) and a Streamlit web UI on top. A consistent aesthetic — Rich-rendered tables, color-coded panels, and ANSI-friendly terminal output — has been applied across every task.

A small set of example portfolios in [examples/](examples/) (`aggressive.json`, `balanced.json`, `conservative.json`) is shared across Tasks 1, 3, 4, and the bonus to keep testing consistent.

| Task | Name                                 | Primary Skill                          | Marks |
|------|--------------------------------------|----------------------------------------|-------|
| 01   | Portfolio Risk Calculator            | Python · Quantitative thinking         | 30    |
| 02   | Live Market Data Fetch               | APIs · Async · Error handling          | 20    |
| 03   | AI-Powered Portfolio Explainer       | LLM prompting · Structured output      | 30    |
| 04   | NL Stress-Test (Open Problem)        | Initiative · Judgment · Composition    | 20    |
| 05   | Historical VaR + Monte Carlo (extra) | Numerical methods · Empirical tail risk | bonus |
| —    | Streamlit web UI (extra)             | UX · `timecell serve`                  | bonus |

---

## Setup & Execution

### Prerequisites

- Python 3.10+
- An API key for **Google Gemini** (free tier, get one at [aistudio.google.com](https://aistudio.google.com))

### Install

```bash
git clone https://github.com/<you>/timecell-intern-Jay-Patel.git
cd timecell-intern-Jay-Patel

# Creates .venv and installs the package in editable mode with dev tools
make install
source .venv/bin/activate
```

Or, equivalently, using only `pip`:

```bash
pip install -r requirements.txt
pip install -e .
```

### API Keys

Tasks 3 and 4 require Gemini. Tasks 1 and 2 work without any key.

```bash
cp .env.example .env
# edit .env and paste your GEMINI_API_KEY
```

`.env` contents:

```
GEMINI_API_KEY=your-key-here
TIMECELL_MODEL=gemini-2.5-flash    # optional override
TIMECELL_HTTP_TIMEOUT=10           # optional
```

### Running the Tasks

Every task is a subcommand of the unified `timecell` CLI:

```bash
# Task 1 — Portfolio Risk Calculator
timecell risk examples/balanced.json
timecell risk examples/aggressive.json --json   # spec-shape JSON output

# Task 2 — Live Market Data Fetch
timecell market                                  # default trio: NIFTY50, RELIANCE, BTC
timecell market ^NSEI BTC ETH GC=F               # any tickers you want

# Task 3 — AI-Powered Portfolio Explainer
timecell explain examples/balanced.json
timecell explain examples/balanced.json --tone expert --raw --critic

# Task 4 — Natural-Language Stress-Test
timecell stress examples/balanced.json "what if BTC crashes 70% and gold rallies 20%?"

# Task 5 (extra) — Historical VaR + Monte Carlo
timecell var examples/balanced.json --years 5 --paths 10000

# Web UI (extra) — runs all tasks in the browser
timecell serve                                   # → http://localhost:8501
```

---

## What APIs does this use?

| Purpose                              | Provider                                              | Cost | Key needed?                                  |
|--------------------------------------|-------------------------------------------------------|------|----------------------------------------------|
| **LLM** (Tasks 3, 4)                 | **Google Gemini** (`gemini-2.5-flash`) — structured JSON output via native `response_schema` | Free tier | ✅ `GEMINI_API_KEY` |
| **Stocks · indices · ETFs · gold**   | Yahoo Finance via `yfinance`                          | Free | ❌                                           |
| **Crypto**                           | CoinGecko Public API                                  | Free | ❌                                           |

**Why Gemini?** Free tier, fast inference, and — critically — first-class structured output. Gemini's `response_schema` parameter accepts a Pydantic model directly, which means the LLM's output is guaranteed to match the schema before it ever leaves the SDK. For a project that demands the LLM never invent numbers, that schema enforcement is doing real work.

---

## Architecture in one diagram

```
                    +---------------------------------------------------+
                    |                  timecell  CLI                    |
                    |              (Typer + Rich, src/cli.py)           |
                    +------------+-------------+-------------+----------+
                                 |             |             |
                  +--------------v--+   +------v-------+   +-v--------------+
                  |  risk.py        |   |  market.py   |   |  explainer.py  |
                  |  pure math.     |   |  async,      |   |  Decision      |
                  |  no I/O.        |   |  parallel,   |   |  Spine + critic|
                  |  (Task 1)       |   |  retried.    |   |  (Task 3)      |
                  +-----------------+   |  (Task 2)    |   +--------+-------+
                                        +--------------+            |
                                                                    |
                  +-------------------------------------------------v+
                  |  stress.py  (Task 4 — the showcase)              |
                  |  Gemini as a STRUCTURED PARSER:                  |
                  |  NL scenario -> per-asset shocks -> re-run risk. |
                  |  Composes risk.py + ai_client.py.                |
                  +--------------------------------------------------+

                  +-------------------------------------------------+
                  |  var.py + history.py  (Task 5 — extra)          |
                  |  5y daily history, empirical VaR/CVaR,          |
                  |  Monte Carlo bootstrap (preserves correlation). |
                  +-------------------------------------------------+

                  +-------------------------------------------------+
                  |  app.py — Streamlit UI for localhost + Cloud    |
                  |  Calls the same functions; no duplicate logic.  |
                  +-------------------------------------------------+
```

Hard separation between **deterministic math** and **probabilistic narration**: the LLM only ever sees pre-computed metrics and is forbidden from inventing numbers.

---

## Task 1 — Portfolio Risk Calculator (30 pts)

### Overview

A highly deterministic, math-driven risk calculator. It evaluates a portfolio's resilience against an expected market crash, computes post-crash values, runway months (how long the portfolio sustains monthly expenses), and identifies the largest risk contributor.

### Workflow

1. **Input loading.** A portfolio JSON file is parsed and validated against the [Portfolio](src/timecell/models.py) Pydantic model. Allocations must sum to 100%, asset names must be unique, and `expected_crash_pct` values must be ≤ 0.
2. **Metrics computation.** [compute_risk_metrics()](src/timecell/risk.py) returns the exact dict shape the brief asks for: `post_crash_value`, `runway_months`, `ruin_test`, `largest_risk_asset`, `concentration_warning`.
3. **Side-by-side scenarios.** [compute_risk_report()](src/timecell/risk.py) computes both a *severe* scenario (using the user's expected crashes) and a *moderate* scenario (50% of severity) for direct comparison.
4. **Ruin test.** If post-crash runway > 12 months → `PASS`; otherwise `FAIL`.
5. **Concentration warning.** Any single asset above 40% allocation triggers a warning (strict `>`, so exactly 40% does not warn).
6. **ASCII allocation chart.** [render_allocation_chart()](src/timecell/risk.py) draws a no-dependency bar chart, color-coded by crash severity.

### How to Run

```bash
timecell risk examples/balanced.json
timecell risk examples/aggressive.json --json   # spec-shape JSON only
timecell risk examples/balanced.json --no-chart
```

### Expected Output

A Rich-rendered terminal interface containing:

- An **allocation bar chart** color-coded by per-asset crash severity.
- A **side-by-side scenarios table** (Severe vs Moderate) for post-crash value, drawdown, runway months, and ruin test.
- The **largest risk asset**, and a yellow concentration warning if any holding exceeds 40%.
- The **spec-shape dict** printed at the bottom (so a reviewer can match it directly to the brief).

```python
>>> from timecell.risk import compute_risk_metrics
>>> compute_risk_metrics({
...     "total_value_inr": 10_000_000,
...     "monthly_expenses_inr": 80_000,
...     "assets": [
...         {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
...         {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
...         {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
...         {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct":   0},
...     ],
... })
{
  "post_crash_value": 5_700_000.0,
  "runway_months": 71.25,
  "ruin_test": "PASS",
  "largest_risk_asset": "BTC",
  "concentration_warning": False,
}
```

**Both bonuses implemented:** moderate scenario alongside severe, and an ASCII allocation chart with no external plotting library.

**Edge cases tested** ([tests/test_risk.py](tests/test_risk.py)): 100% cash portfolio · zero monthly burn → infinite runway · allocations not summing to 100% → rejected · duplicate asset names → rejected · strict-`>` 40% boundary · positive `expected_crash_pct` rejected at the model boundary.

---

## Task 2 — Live Market Data Fetcher (20 pts)

### Overview

A live market-data fetcher that retrieves real-time prices for a set of diverse assets (equities, indices, crypto, commodities) in parallel, with isolated failure handling and exponential-backoff retries on each provider.

### Workflow

1. **Provider routing.** [market.py](src/timecell/market.py) classifies each symbol as either a Yahoo Finance ticker (stocks, indices, ETFs, gold futures) or a CoinGecko crypto ticker (`BTC`, `ETH`, `SOL`, …).
2. **Parallel fetch.** All symbols are fetched concurrently via `asyncio.gather`. yfinance (sync) is wrapped in `asyncio.to_thread` so the gather stays non-blocking.
3. **Retries.** Each provider is wrapped with `tenacity` exponential backoff (3 attempts, 0.5s → 4s) — a single transient 503 will not kill a demo.
4. **Isolated failures.** A failure for one symbol is captured into a `Quote` object with `ok=False` and an error string; **the others still succeed** and the table shows `OK` for them and the error for the failed one.
5. **IST timestamp.** Times are rendered in IST as the brief asks.

### How to Run

```bash
timecell market                          # default trio: NIFTY50, RELIANCE, BTC
timecell market ^NSEI BTC ETH GC=F       # any combo of yfinance + CoinGecko symbols
```

### Expected Output

```text
                Asset Prices -- fetched at 2026-05-02 10:32:15 IST
+--------------+---------------+----------+-----------+--------+
| Asset        |         Price | Currency | Provider  | Status |
+--------------+---------------+----------+-----------+--------+
| ^NSEI        |     22,541.80 | INR      | yfinance  | OK     |
| RELIANCE.NS  |      2,901.05 | INR      | yfinance  | OK     |
| BTC          |     62,341.20 | USD      | coingecko | OK     |
+--------------+---------------+----------+-----------+--------+
```

If one provider goes down mid-run, that row shows the truncated error message in red while every other row continues to render normally.

---

## Task 3 — AI-Powered Portfolio Explainer (30 pts)

### Overview

The bridge between raw mathematical risk (Task 1) and human comprehension. The calculated risk metrics are fed to Gemini, which is forced to output a structured 4-part summary (Summary · Doing Well · Consider Changing · Verdict) with **a citation for every numerical claim**. The verdict is constrained to one of three labels: `Aggressive`, `Balanced`, `Conservative`.

### The Decision Spine pattern

This is the heart of the submission. Timecell's site says the product *"links every recommendation back to the inputs, the framework, and the decision."* I encoded that as a hard constraint on the LLM ([prompts.py](src/timecell/prompts.py)):

```
HARD RULES — break any of these and the answer is invalid:

1. NEVER invent or estimate numbers. You will be given a JSON `metrics` block.
   Every number in your prose must come from that block. If a fact you want
   to state is not in `metrics`, drop the fact.

2. EVERY claim in your output (summary, doing_well, consider_changing) must
   map to one entry in the `spine` array:
     { "claim": "...", "cited_metric": "metric_key = value", "confidence_pct": 0..100 }
   If you cannot cite a metric for a claim, do not make that claim.
```

The output is constrained to a Pydantic schema via Gemini's native `response_schema` mode, validated on parse, and **failures are loud** — a malformed JSON or a missing spine raises `LLMError` with the raw response, instead of silently shipping a broken answer.

### Workflow

1. **Deterministic compute.** [compute_risk_report()](src/timecell/risk.py) runs first, producing the same metrics Task 1 produces.
2. **Tone-adjusted prompting.** [build_explainer_user_prompt()](src/timecell/prompts.py) injects a `metrics` JSON block, the user's chosen tone (`beginner` · `experienced` · `expert`), and the Decision Spine instructions. Gemini is forbidden from using markdown.
3. **Structured output.** [call_structured()](src/timecell/ai_client.py) calls Gemini with `response_schema=Explanation`; the SDK guarantees the output shape; we re-validate against Pydantic anyway to catch SDK drift.
4. **Critic loop (bonus).** With `--critic`, a *second* LLM call ([critique_explanation()](src/timecell/explainer.py)) acts as a "Senior Risk Officer" that fact-checks the first explanation against the same metrics, returning `PASS` / `PASS_WITH_NITS` / `FAIL` with per-claim issues.
5. **Raw + structured side by side.** With `--raw`, the raw LLM response is printed alongside the structured output exactly as the brief asks.

### How to Run

```bash
timecell explain examples/balanced.json                              # default tone
timecell explain examples/balanced.json --tone expert                # bonus 1
timecell explain examples/balanced.json --tone beginner --raw        # show raw + structured
timecell explain examples/balanced.json --critic                     # bonus 2 — second LLM fact-checks the first
```

### Expected Output

The terminal will display:

- **Optional raw LLM response** in a dim panel (with `--raw`).
- **Summary** in a cyan panel, with the verdict color-coded (red Aggressive · yellow Balanced · green Conservative).
- **Doing Well** in a green panel.
- **Consider Changing** in a yellow panel.
- A **Decision Spine table** — every claim, the metric that justifies it, and the model's stated confidence percentage.
- A **Self-Critique panel** (with `--critic`) showing `PASS` / `PASS_WITH_NITS` / `FAIL` and a table of any issues the second LLM caught.

### Prompt-engineering evolution

| Attempt | Result | What I learned |
|---|---|---|
| Free-form prompt asking for "a 4-sentence explanation" | Output drifted, sometimes invented numbers (e.g. quoted a different drawdown than the one I'd computed). | The LLM was happy to be loose with quantitative claims. |
| Added a `metrics` JSON block with all the inputs the model needed | Drift mostly stopped, but it would still drop the verdict or skip the "doing well" line. | A soft schema is not enough. |
| Switched to JSON output via `response_schema` + Pydantic validation + verdict enum | Stable, parseable, every run is the same shape. | Type system at the boundary > prose. |
| Added the **Decision Spine** requirement (claim ↔ cited metric ↔ confidence) | The model started self-policing — if it couldn't cite a number for a claim, it dropped the claim. | Forcing receipts is more powerful than asking for accuracy. |

---

## Task 4 — Natural-Language Stress-Test (20 pts) — *the open-problem showcase*

### What This Is

The brief says: *"build something — anything — that you think would make Timecell better, more useful, or more interesting."*

I built **a natural-language stress-test CLI**: ask Gemini what would happen to your portfolio under any scenario you can describe in plain English, and the math engine re-runs Task 1 on the shocked portfolio.

> *"Show me what would happen if BTC crashes 70% and gold rallies 20%."*
> *"What if Indian equities lose 50% but bonds hold steady?"*
> *"What if everything except cash drops 30%?"*

Most portfolio tools show you outcomes for a single, fixed scenario. This one lets you ask any what-if question your brain comes up with, in your own words.

### Workflow

1. **You type a scenario** in plain English alongside a portfolio JSON file.
2. **Gemini parses it as a structured object.** Same `response_schema` pattern as Task 3 — the model's output is guaranteed to be a typed `StressScenario = {rationale, shocks: [{asset_name, shock_pct}]}` object before it leaves the SDK.
3. **Python applies the shocks.** [apply_shocks()](src/timecell/stress.py) overwrites each shocked asset's `expected_crash_pct`. Rallies clamp to 0 — we're testing *crash survival*, not upside. Case-insensitive name matching. Unmentioned assets pass through unchanged.
4. **Task 1's `compute_risk_report()` re-runs** on the shocked portfolio. The math engine has no idea the LLM was involved.
5. **Result rendered as Rich tables.**

**Two LLM-touchpoints, deterministic math in between. The LLM finds the shocks; the math measures the consequences.**

### How to Run

```bash
timecell stress examples/balanced.json "what if BTC crashes 70% and gold rallies 20%?"
timecell stress examples/aggressive.json "what if Indian equities lose 50%?"
timecell stress examples/conservative.json "what if everything except cash drops 30%?"
```

### Expected Output

```text
+- Scenario ----------------------------------------------------------+
| what if BTC crashes 70% and gold rallies 20%?                       |
+---------------------------------------------------------------------+
+- Gemini's parse ----------------------------------------------------+
| User explicitly named two shocks: BTC -70% and GOLD +20%.           |
| NIFTY and CASH are unchanged.                                       |
+---------------------------------------------------------------------+

         Per-asset shocks                Resulting risk metrics
  +----------+---------+         +---------------------+------------+
  | Asset    |   Shock |         | Metric              |      Value |
  +----------+---------+         +---------------------+------------+
  | BTC      |  -70.0% |         | Post-crash value    | INR 6.82M  |
  | GOLD     |  +20.0% |         | Drawdown            |     -31.8% |
  +----------+---------+         | Runway (months)     |      85.25 |
                                 | Ruin test           | PASS       |
                                 | Largest risk asset  | BTC        |
                                 +---------------------+------------+
```

### Why this is worth building

- **It composes Tasks 1 + 3.** Demonstrates the LLM as a *parser*, not just a generator. That's a more valuable signal for fintech than yet another "explain my portfolio" wrapper.
- **It maps to Sandeep's framing** — *"a terminal, not a dashboard."* A what-if question is exactly the kind of input a CLI handles better than a UI: typed in plain English, answered in seconds, repeatable from history.
- **Total LLM calls per query: 1.** No fan-out, no agentic loops. Cheap and fast.
- **The clamping rule** (rallies → 0) is a deliberate product decision, not a bug — explained inline so a reviewer sees the reasoning.

---

## Task 5 — Historical VaR + Monte Carlo (extra credit)

The brief's Task 1 takes user-supplied `expected_crash_pct` numbers as ground truth. That's a load-bearing assumption: garbage in, garbage out. Task 5 replaces those guesses with *measured* tail risk pulled from real price history.

```bash
timecell var examples/balanced.json --years 5 --paths 10000
```

```text
  Historical VaR/CVaR · 1257 trading days · INR 10,000,000 portfolio
  +------------+-----------------------+----------------------------------+
  | Confidence | VaR (1-day)           | CVaR (1-day, expected tail loss) |
  +------------+-----------------------+----------------------------------+
  | 95%        | -2.41%   INR -241,000 | -3.62%   INR -362,000            |
  | 99%        | -4.18%   INR -418,000 | -5.55%   INR -555,000            |
  +------------+-----------------------+----------------------------------+

  Monte Carlo · 10,000 paths · 252-day horizon · historical bootstrap
  +----------------------------------------+------------------+
  | Outcome                                | Portfolio value  |
  +----------------------------------------+------------------+
  | 5th percentile (worst case)            | INR 6,420,000    |
  | Median outcome                         | INR 11,180,000   |
  | 95th percentile (best case)            | INR 19,940,000   |
  | Ruin probability (runway < 12 months)  | 0.32%            |
  +----------------------------------------+------------------+
```

**Method.** Empirical VaR (no normality assumption) plus CVaR for expected tail loss. The Monte Carlo bootstraps full rows from the historical return matrix, which preserves cross-asset correlation automatically — when BTC has its worst day, NIFTY tends to too, and the bootstrap inherits that. Numpy seed for reproducibility, no LLM in the loop.

**Why this matters.** Task 1 says "BTC will crash -80%". That's a guess. Task 5 says *the empirical 99% 1-day VaR is -4.18%, the worst 5% of 1-year outcomes leave you with INR 6.4M, and the probability of ruin is 0.32% over a year* — with a falsifiable methodology that any reviewer can audit. **It turns the project from "deterministic-but-toy" into "deterministic-and-defensible."**

---

## Web UI — `timecell serve`

```bash
timecell serve            # → http://localhost:8501
```

Same code, browser surface. Every tab in the Streamlit app calls the same functions the CLI does — there is no duplicated logic. Deploys to Streamlit Cloud directly: push to GitHub, point share.streamlit.io at `app.py`, set `GEMINI_API_KEY` in the app's secrets, done.

---

## Tests

```bash
make test     # 26+ tests, all mocked/numpy-only; no network, no LLM calls
```

Coverage:
- All Task 1 spec numbers from the brief, plus edge cases (all-cash, zero burn, bad allocations, duplicates, strict-vs-loose concentration boundary).
- Stress-test shock application: case-insensitive name match, rally clamping, immutability of input portfolio, unmentioned-assets passthrough.
- VaR/CVaR ordering, CVaR ≤ VaR, percentile recovery on a synthetic normal distribution, Monte Carlo determinism under fixed seed, zero-return → zero VaR, zero expenses → zero ruin probability.

LLM calls are not exercised in tests — they're slow, paid, and non-deterministic. The schema validation in [ai_client.py](src/timecell/ai_client.py) is the safety net for those paths.

---

## How I Used AI to Build This

The brief asks how I used AI tools. Honestly: heavily, but always as a collaborator on a problem I owned. Concretely:

### Task 1 — Risk Engine

- **Understanding the problem.** I asked ChatGPT to read the assessment PDF and re-state the problem in its own words, so I could check that my reading and its reading agreed before I wrote code. They did.
- **Edge-case enumeration.** I listed the obvious cases (zero burn, 100% cash, allocation mismatch) and asked Claude to red-team my list. It surfaced the strict-`>` 40% concentration boundary and the duplicate-asset case — both became tests.
- **Formula verification.** I cross-checked individual computations (e.g. "post-crash BTC value for 30% allocation, -80% crash, INR 10M total") across multiple tools to make sure I wasn't carrying a sign error.
- **Variable naming and comments.** Once the logic was right, I asked GPT to rename variables to be self-documenting (`post_crash_asset_value`, `compute_risk_report`) and to add a small number of *why*-comments. Anywhere a comment merely restated the code, I cut it.

### Task 2 — Live Market Data

- **API selection.** I asked Claude to recommend free, no-key-needed APIs for the asset trio. yfinance + CoinGecko came out of that. I verified the endpoints by hand against their docs before writing a line of code.
- **Async + retry pattern.** I sketched the `asyncio.gather` + `tenacity` structure, then asked Claude to flag failure modes I'd missed. The "wrap yfinance sync calls in `asyncio.to_thread`" detail came from that pass — yfinance is sync internally, and naive `await` on it would block the gather.
- **Provider isolation.** The rule "one provider failing must not crash the others" came from me. The implementation (capture per-symbol exceptions into a `Quote` object instead of letting them propagate) was something I worked out by walking through the failure cases by hand.

### Task 3 — Explainer

- **Prompt iteration.** v1 was a free-form ask. The model invented numbers. v2 added a `metrics` JSON block and forbade markdown — it stopped inventing but sometimes dropped the verdict. v3 switched to Gemini's native `response_schema` with a Pydantic model — output finally shaped consistently. v4 added the **Decision Spine** (every claim must cite a metric); the model began self-policing and dropped claims it couldn't cite. The Decision Spine was the breakthrough.
- **Red-teaming the prompt.** I asked Claude to try to break the prompt: *"give me metrics where ruin_test=FAIL — can you still produce an explanation that says 'safe'?"* Watching it succeed (until the spine constraint was added) was what convinced me the spine was load-bearing, not decoration.
- **Critic pass.** The `--critic` flag was originally framed as a "LangGraph validation loop", but a plain Python second LLM call with `temperature=0.0` and a fact-checking system prompt does the same job with a fraction of the dependencies. I asked Claude to confirm the simpler version was equivalent before cutting LangGraph.

### Task 4 — Stress-Test

- **Architecture.** I drafted the AI ↔ math separation by hand: AI extracts shocks (subjective, contextual), math computes consequences (deterministic, auditable). I asked Claude whether there was a case for letting the LLM compute the post-crash value directly and it agreed there wasn't — a language model should not be trusted to multiply numbers.
- **Schema design.** First attempt was free-text shocks → inconsistent format. Second attempt was a strict JSON schema with `asset_name` + `shock_pct`, and the model became reliable. Same lesson as Task 3: machine-readable beats prose.
- **Clamping rule.** "Rallies clamp to 0" was a product call I made consciously. We're testing crash survival, not upside; letting a +20% rally shore up the runway in a stress test would muddle the framing. I noted the reasoning in the prompt and the docstring so a reviewer sees the *why*, not just the *what*.

### General

I read every line that landed in this repo and can walk through any of it on a follow-up call. The most fun parts to discuss would be [prompts.py](src/timecell/prompts.py) (where the Decision Spine lives) and [stress.py](src/timecell/stress.py) (where AI-as-parser composes with deterministic math).

---

## Acknowledgements

- **Google Gemini** — generous free tier, fast inference, and the `response_schema` parameter that made Tasks 3 and 4 reliable instead of finicky.
- **Claude Code & ChatGPT** — for red-teaming prompts, surfacing edge cases, and architecture review on the parts I wrote first.
- **DeepSeek** — for rapid iteration on the prompt-engineering passes for Task 3.
- **yfinance · CoinGecko** — for free, no-key market data that made Task 2 a one-evening problem instead of a multi-vendor key-management project.
- **Timecell** — for a thoughtfully designed assessment that mirrors real startup engineering: a clear problem, freedom to pick the stack, and an open-ended Task 4 that rewards judgment over compliance.
