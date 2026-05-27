# Helper Agent

A personal multi-crew AI assistant built with [crewAI](https://crewai.com). Takes a natural language query, routes it to the right specialist crew, and runs it — no manual crew selection needed.

---

## Crews

| Crew | Triggers On | Inputs |
|---|---|---|
| `EmailHelper` | emails, inbox, messages | `num_sentence_summary` |
| `StockExpert` | stocks, positions, trading, mutual funds | `user_question`, `stock` |

Routing is handled by a lightweight LLM call (`groq/llama-3.1-8b-instant`) in `main.py` — no full agent overhead for dispatch.

---

## Setup

Requires Python `>=3.10 <3.14` and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
crewai install
```

Create a `.env` file in the project root:

```env
MODEL=groq/llama-3.3-70b-versatile   # for default/quick testing
GROQ_API_KEY=...
GEMINI_API_KEY=...
```

> **Note:** Create a `tests/` folder in the project root. All intermediate outputs — helpful for response quality analysis and prompt refinement — are stored here.

---

## Running

Open a terminal at the root directory (the one containing `pyproject.toml`) and run:

```bash
crewai run
```

You'll be prompted:

```
What assistance can I provide you today with :: [Your Query Here]
```

Crews are lazy-imported at dispatch time so unused crews don't initialize on startup.

---

## Current Limitations & Roadmap

### Known Constraints
- **API rate limits** cap daily usage — market research in particular requires large context windows to track metrics and news for report generation.
- Even with additional agents, token exhaustion limits practical testing.

### Planned Improvements
- [ ] Better prompt quality and RAG pipelines
- [ ] Separate databases for crews and individual agents
- [ ] **Conversation memory** — so users can continue discussing a report or topic without re-running the full pipeline
- [ ] Math tools for agents (currently absent; agents attempt calculations without proper tooling, though correct, unreliable for complex applications)
- [ ] Reduce redundant calculations — growth and value agents currently compute overlapping metrics independently

### Potential Future Crews
Additional crews specifically for personal use cases will be added over time.

---

## Example Output

**Query asked on May 27, 2026 ~7 PM:**
> *ITC went ex-dividend today and price dropped to 291. I have ITC with average of 320, should I book losses and exit?*

---

# INSTITUTIONAL EQUITY RESEARCH & SYNTHESIS BRIEF

**To:** Private Client
**From:** Senior Investment Synthesis Analyst
**Date:** May 27, 2026
**Subject:** ITC Limited (NSE: ITC) — Comprehensive Investment Action Plan Post Ex-Dividend Price Adjustment

---

## 1. Executive Recommendation & Quantitative Summary

**Direct Actionable Decision: HOLD (Do Not Book Losses and Do Not Exit).**

Crystallizing a nominal loss of **9.06%** at the current market price (CMP) of **₹291.00** against your purchase average of **₹320.00** represents an inefficient capital allocation decision. This price drop is a mechanical, non-fundamental adjustment resulting from the stock going ex-dividend today (May 27, 2026) for a final dividend of **₹8.00 per share**.

When factoring in the guaranteed dividend cash inflow, your true economic position is a minor unrealized variance of **-6.25%**, while the business continues to trade at a steep discount to its intrinsic value with superior capital return metrics.

### Key Decision Metrics

| Metric | Value / Formula | Status vs. Threshold |
|:---|:---|:---|
| **Average Purchase Price** | ₹320.00 | Cost Basis |
| **Current Market Price** | ₹291.00 | Market Value (May 27, 2026) |
| **Nominal Capital Loss** | (291 − 320) / 320 = **-9.06%** | Temporary Market Variance |
| **Dividend Declared** | ₹8.00 per share | Record Date: May 27, 2026 |
| **Total Shareholder Return (TSR)** | (291 + 8 − 320) / 320 = **-6.25%** | Actual Economic Exposure |
| **Estimated Intrinsic Value** | ₹350.00 (via DDM/DCF) | Target Value |
| **Valuation Margin of Safety** | (350 − 291) / 350 = **16.86%** | Highly Favorable (> 15% Hurdle) |
| **Trailing P/E Ratio** | 291 / 16.36 = **17.79x** | Historically Undervalued |
| **Earnings Yield** | 16.36 / 291 = **5.62%** | Exceeds Risk-Free Rate |
| **ROCE (FY24 Standalone)** | **36.00%** | Exceeds Coffee Can Hurdle (15%) |
| **ROE (FY24 Standalone)** | **29.60%** | High Capital Efficiency |

---

## 2. The Ex-Dividend Mechanics & Economic Reality

The drop in ITC's stock price to **₹291.00** is a standard corporate action adjustment. On the ex-dividend date, the exchange calibrates the opening price downward by the dividend amount (**₹8.00**) because the cash outflow reduces the company's book value per share by that exact quantum.

### Mathematical Reconciliation of Position (1,000 shares assumed)

| Step | Calculation | Value |
|:---|:---|:---|
| Post-Ex-Div Stock Value | 1,000 × ₹291.00 | ₹2,91,000 |
| Dividend Cash Receivable (T+1/T+2) | 1,000 × ₹8.00 | ₹8,000 |
| **Total Economic Value** | ₹2,91,000 + ₹8,000 | **₹2,99,000** |
| Net Economic Loss | (299 − 320) / 320 | **-6.25%** |

> Booking losses at ₹291.00 means forfeiting equity recovery while converting a temporary, tax-advantaged dividend payout into a crystallized short-term capital loss.

---

## 3. Value Analysis & Behavioral Traps (Parag Parikh Framework)

### A. Quality–Growth–Price (QGP) Analysis

**Quality (Q)**
ITC exhibits exceptional quality. Standalone Reserves and Surplus stand at **₹70,477.24 Cr** against an Equity Share Capital of only **₹1,248.47 Cr**, representing a virtually debt-free balance sheet. Operating Profit Margin stands at **37.42%** (OPM = ₹23,435 Cr on Sales of ₹62,628 Cr).

**Growth (G)**
Growth is stable and defensive. Q4 FY26 net profit rose **5% YoY** to **₹5,113 Cr**. The upcoming FMCG and hotel demerger provides additional structural tailwinds.

**Price (P)**
At a P/E of **17.79x** and a dividend yield of **4.97%** (FY24), the price is highly defensive.

---

### B. Valuation Margin of Safety (DDM)

Using the Dividend Discount Model:

```
Intrinsic Value (V₀) = D₁ / (r − g)

Where:
  D₁  = ₹16.36  (Expected Dividend, ~100% payout of EPS)
  r   = 9.50%   (Required Rate of Return / Cost of Equity)
  g   = 4.80%   (Terminal Growth Rate)

V₀ = 16.36 / (0.095 − 0.048) = 16.36 / 0.047 ≈ ₹348 → Rounded to ₹350.00

Margin of Safety = (350 − 291) / 350 × 100 = 16.86%
```

A **16.86% Margin of Safety** indicates the market has excessively discounted regulatory risks (e.g., cigarette taxation), offering a robust cushion for long-term holders.

---

### C. Behavioral Finance Pitfalls to Avoid

| Bias | Description | How It Applies Here |
|:---|:---|:---|
| **Anchoring Bias** | Over-relying on purchase price as a reference point | Decisions must be based on future value from ₹291, not ₹320 |
| **Loss Aversion** | Psychological pain of paper loss triggers irrational exit | Leading to the *Disposition Effect* — panic-selling a sound compounder |
| **Availability Heuristic** | Over-weighting recent negative headlines | Ignores structural reality of 36% ROCE and dominant market share |

---

## 4. Coffee Can Portfolio Evaluation (Saurabh Mukherjea Framework)

Saurabh Mukherjea's "Coffee Can Portfolio" (CCP) methodology filters for low-volatility, high-compounding structural franchises.

**Entry Hurdles (Non-Financial Firms):**
- Revenue Growth ≥ 10% per annum over 10 years
- ROCE ≥ 15% per annum over 10 years

### ITC's CCP Alignment

| Parameter | CCP Hurdle | ITC Actual | Status |
|:---|:---|:---|:---|
| ROCE | ≥ 15% | **36%** (+2,400 bps above hurdle) | ✅ Pass |
| ROE | Benchmark Cost of Capital | **29.60%** | ✅ Pass |

### Historical CCP Performance (ITC as Constituent)

| CCP Iteration | Period | ITC Included | Portfolio Return | Alpha vs. Sensex |
|:---|:---|:---|:---|:---|
| Iteration 12 | 2011–Present | Large-Cap | 14.2% p.a. | +3.9% |
| Iteration 14 | 2013–Present | Large-Cap | 24.4% CAGR | Beat Sensex (13.7%) |
| Iteration 15 | 2014–Present | Large-Cap | 26.4% absolute | +18.3% |

> Exiting a structural CCP compounder due to a temporary 9% nominal drawdown violates the core tenet: *buy high-quality businesses and do not touch them for 10 years.*

---

## 5. Synthesized Decision Matrix

| Decision Rule | Exit Condition | ITC Actual | Verdict | Action |
|:---|:---|:---|:---|:---|
| Capital Efficiency | ROCE < 15% | **36.00%** | ✅ Passed | Do Not Exit |
| Solvency Risk | Debt/Equity > 0.50x | **< 0.02x** | ✅ Passed | Do Not Exit |
| Valuation Bubble | Trailing P/E > 35x | **17.79x** | ✅ Passed | Do Not Exit |
| Dividend Coverage | OCF < Dividend Paid | **OCF ≈ ₹24,000 Cr** | ✅ Passed | Do Not Exit |
| Margin of Safety | Discount to IV < 0% | **+16.86% Discount** | ✅ Passed | Do Not Exit |

---

## Final Synthesis

ITC is trading at an attractive **17.79x P/E**, yielding **4.97%** in dividends, and generating **36% ROCE**. The drop to ₹291.00 is a mechanical ex-dividend adjustment — the ₹8.00 will be credited directly to your bank account.

Selling now:
- Crystallizes a non-existent economic loss
- Incurs transaction costs
- Triggers anchoring and loss aversion biases

**The mathematically precise and fundamentally sound decision is to HOLD your position.**

---
### Execution Logs
For a detailed look at the agent's step-by-step reasoning and tool execution, view the full logs here:
[View Detailed Agent Execution Log](https://gist.github.com/sunny9n9/25d795186bfd36e59541a2314bbba3f2)