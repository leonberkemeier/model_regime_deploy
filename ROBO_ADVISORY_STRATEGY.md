# Strategic Architecture: Robo-Advisory Portfolio Management

This document outlines the high-level strategic vision for the AI-driven Robo-Advisor application. To operate as an institutional-grade asset manager, the portfolio management logic is strictly divided into two distinct processes. 

By separating the "Alpha/Origination Model" from the "Execution/Management Model", we avoid massive portfolio churn, minimize tax implications, and reduce transaction fees.

---

## 🟢 Process 1: The "Greenfield" Setup (Current Focus)

**Goal: Determine the optimal portfolio *today* if we were investing raw cash.**

This is the absolute foundation of the system. Every single day, the AI PC evaluates the entire market universe completely objectively, with zero regard for existing user portfolios.

### How it works:
1. **Quantitative Evaluation:** The AI PC runs the Hidden Markov Model (HMM) to determine the macro market regime. It then runs Monte Carlo (MC) simulations to calculate exact risk metrics (VaR, Expected Shortfall) for every asset.
2. **Qualitative Scoring:** Using local LLMs (e.g., Gemma 4) and RAG (Retrieval-Augmented Generation), it reads SEC filings and earnings to score fundamental quality out of 1.0.
3. **Model Generation:** Based on these metrics, the AI outputs **5 completely new, optimal Model Portfolios** corresponding to 5 distinct Risk Profiles (1: Conservative  -> 5: Aggressive).
4. **Application:** If a new user signs up *today* with $10,000 and is assigned Risk Profile 3, their money is immediately deployed to perfectly mirror today's "Profile 3 Greenfield Model."

**Status:** *Active Development.* This is the immediate engineering priority for the `deploy_on_ai-pc` node.

---

## 🔄 Process 2: Tactical Reshifting & Capital Defense (Future Phase)

**Goal: Manage *existing* portfolios to minimize risk and evade opportunity costs, without excessive turnover.**

If a user invested 6 months ago, their portfolio has drifted. We **do not** want to indiscriminately force their 6-month-old portfolio to mirror today's Greenfield model. Doing so would trigger massive 100% turnover trades, realizing huge capital gains taxes and incurring massive slippage.

### How it works:
This process acts as a "Surgical Manager" for existing accounts:
1. **Capital Defense (Risk Reduction):** If the Greenfield process detects a regime shift (e.g., Bull Market -> High Volatility Bear), the Reshifting algorithm triggers a defensive rebalance to reduce equity exposure across existing portfolios.
2. **Evading Opportunity Cost:** If the Greenfield LLM score for a held asset (e.g., AAPL) suddenly plummets due to terrible earnings guidance, the Reshifting algorithm surgically removes or trims that failing asset and replaces it with a current Greenfield winner.
3. **Application:** It only executes trades when the mathematical benefit (alpha/protection) outweighs the friction (taxes/fees) of the trade.

**Status:** *Pending Team Discussion.* The specific rules, thresholds, and algorithms for this process will be evaluated by the team once the Greenfield engine is fully operational.

---

## 🛠️ Impact on the AI / ML Node (`deploy_on_ai-pc`)

Because we are focusing strictly on **Process 1**, the daily tasks inside this deployment folder will specifically target:
* `task_markov` & `task_monte_carlo`: Core universe evaluation.
* `task_llm_scoring`: Gemma 4 + MCP/RAG for asset insight.
* `task_greenfield_models`: The final daily orchestration that pushes the 5 ideal Model Portfolios to the Webserver database.