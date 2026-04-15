# Target System Architecture: Context & Overview

**Overall Project Summary:**
This project is a hybrid AI-driven Robo-Advisory and Portfolio Management system. It strictly separates high-frequency numerical data from unstructured qualitative intelligence to prevent LLM hallucinations. The architecture is divided into 6 core pillars:
1. **Nervous System (financial_data_aggregator):** Data aggregation via PostgreSQL, RAG, and an MCP layer.
2. **Regime Brain (model_regime_comparison):** HMM-based market state detection and Monte Carlo VaR simulations.
3. **Conviction Synthesis (model_regime_comparison):** LLM analysis fusing quantitative data and qualitative narratives.
4. **Risk-Factor Envelopes (model_regime_comparison):** Portfolio creation using strict Strategic Asset Allocation (SAA) and Factor filters.
5. **Gap-Filler Engine (model_regime_comparison):** Priority queue system handling practical recurring deposits (DCA) and fee-efficient rebalancing.
6. **Mirror Ledger (Trading_Simulator):** High-fidelity trading simulation app offering AI-driven monthly post-mortems.

---

## 2. Data Modelling: The "Regime Brain"
*(Housed in this module)*

The system utilizes a dual-engine approach to identify market phases and tail risks.

### Market Regime Detection (HMM)
*   **Logic:** A Hidden Markov Model (HMM) analyzes global benchmarks to infer the hidden state (e.g., Quiet Growth, Volatile Bear, Inflationary Shock).
*   **Hysteresis (Smoothing):** Regimes only switch if the probability holds above 70% for 3 days, preventing costly "whipsaw" trades.

### Forward Risk Modeling (Monte Carlo)
*   **Simulation:** 10,000 paths are projected per asset cluster.
*   **Hybrid VaR:** The system calculates Value-at-Risk using the maximum of a 60-day window and a 10-year historical window. This ensures the model doesn't become "blind" to tail-risk crashes during long periods of calm.



## 3. LLM Integration: The "Conviction Synthesis"
*(Housed in this module)*

The LLM acts as the senior analyst, refining the quantitative probability with qualitative context.

### The Synthesis Process
*   **Retrieve:** The LLM pulls numerical data (via MCP) and qualitative summaries (via RAG).
*   **Analyze:** Evaluates the "Narrative" vs. the "Numbers."
*   **The Modifier (Δp):** The LLM assigns a conviction score.
    *   *Formula:* `p_final = p_HMM + Δp_LLM`
*   **Audit:** A secondary script verifies that the LLM's explanation matches the numerical direction (the "Truth Checker").



## 4. Portfolio Creation: The "Risk-Factor Envelopes"
*(To be implemented in this module - e.g., `src/portfolio/`)*

Portfolios are constructed within a **Hard Constraint Strategic Asset Allocation (SAA) Matrix**. The system never drifts outside these boundaries; it only optimizes the internal composition of each bucket based on the user's risk profile.

### 4.1 The Strategic Asset Allocation (SAA) Matrix

| Risk Profile | Stocks | Bonds | Crypto | Commodities | Cash |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Conservative** | 20% | 53% | 0% | 12% *(Gold)* | 15% |
| **Mod-Conservative** | 40% | 35% | 2% | 8% *(Broad)* | 15% |
| **Balanced** | 60% | 25% | 5% | 5% *(Broad)* | 5% |
| **Growth** | 78% | 5% | 8% | 4% *(Industrial)* | 5% |
| **Aggressive** | 82% | 0% | 10% | 3% *(Speculative)*| 5% |

### 4.2 Tiered Asset Characterization (The "Factor Filter")

To make portfolios distinct, the 650-stock universe is filtered using Factor-Based Eligibility based on the user's profile:

*   **A. Equity Characterization:**
    *   *Conservative/Mod-Con:* Focus on **Quality & Low Volatility**. Metrics: Beta < 0.8, stable dividend history, high interest coverage ratios.
    *   *Balanced:* Focus on **Quality & Value**. Metrics: Strong Free Cash Flow (FCF) yields, P/E < Market Average.
    *   *Growth/Aggressive:* Focus on **Momentum & Growth**. Metrics: >15% YoY revenue growth, high RSI scores.
*   **B. Commodity Specialization:**
    *   *Gold (Conservative):* Pure tail-risk hedge.
    *   *Broad (Balanced):* Diversified mix of Energy, Metals, and Agriculture.
    *   *Industrial (Growth):* Copper, Lithium, Nickel (structural economic expansion).
    *   *Speculative (Aggressive):* Carbon credits, Uranium, or specific mining small-caps.
*   **C. Crypto Tiers:**
    *   *Tier 1 (Mod-Con/Balanced):* BTC and ETH only.
    *   *Tier 2 (Growth/Aggressive):* Top 10 by Market Cap + Layer 1s (e.g., Solana).

### 4.3 The "Two-Step" Portfolio Construction Logic

Every morning, the system runs a two-step optimization for each of the 5 profiles:

1.  **Bucket Selection (Factor Filtering):** The engine queries the SQL Database via the MCP Server for the Top 50 assets fitting the profile's specific factor filter.
2.  **Intra-Bucket Sizing (Fractional Kelly):** Calculates the weight of selected assets strictly within that bucket's envelope constraint. The result is scaled so the sum of weights exactly equals the SAA envelope constraint.

### 4.4 LLM Role in Risk-Profile Differentiation

The LLM "Analyst" uses divergent System Prompts depending on the target profile:
*   **Conservative Persona ("Cynical Auditor"):** Scores conviction based on debt levels, legal risks, and earnings stability.
*   **Aggressive Persona ("Venture Scout"):** Scores conviction based on Total Addressable Market (TAM), disruptive potential, and R&D velocity.

### 4.5 Tactical Re-Shifting (Regime-Aware Protection)

The HMM Regime can trigger a Tactical Retreat into Cash within a profile's constraints without changing the SAA matrix itself.
*   *Crisis Regime (HMM Trigger):* If an asset bucket's conviction probabilities drop (p < 0.5) due to a crash, the intra-bucket Kelly weights drop to zero, storing the capital within a temporary "Cash Buffer".

### 4.6 Summary of Construction by Profile

| Feature | Conservative | Balanced | Aggressive |
| :--- | :--- | :--- | :--- |
| **Core Goal** | Capital Preservation | Core Wealth Growth | Maximum Alpha |
| **Equity Filter** | Dividend Aristocrats, Utilities | S&P 500 Quality, Value | Tech, Biotech, Momentum |
| **Kelly Aggression** | 0.2x *(Extreme Caution)* | 0.5x *(Balanced)* | 0.7x *(Aggressive growth)* |
| **LLM Focus** | "Is this company safe?" | "Is this company fair value?" | "Is this company leading?" |
| **Rebalancing** | Infrequent (Quarterly) | Monthly | Monthly / Real-time triggers |




## 5. Portfolio Re-shifting: The "Gap-Filler" Engine
*(To be implemented in this module - e.g., `src/execution/`)*

Handles the operational reality of recurring (e.g., €500 monthly) contributions with high transaction costs.

### The Priority Queue Algorithm
*   **Calculate Target:** Determine the "Exemplary Portfolio" weights for the user's profile (as defined by the SAA matrix and Risk-Factor Envelopes).
*   **Identify Shortfall:** Find assets where `Actual Weight < Target Weight`.
*   **Execute Buys:**
    *   *Min Order:* €50 per ticker.
    *   *Priority:* Capital goes to the largest shortfall first.
    *   *Fee Efficiency:* Limit the number of trades per month to ensure total fees stay below 1-2% of the inflow.
*   **Natural Rebalancing:** Uses the monthly inflow to "buy the dip" in underweight assets rather than selling winners, maximizing tax efficiency and reducing commission drag.