# AI-PC Deployment & Testing Progress (Apr 18, 2026)

## 🎯 Current Project Status
We have successfully deployed the **AI-PC compute node** (running Arch Linux) and updated its local analytical scripts. We successfully migrated from relying solely on the central financial Webserver for data to an autonomous, direct-fetch capability using `yfinance`.

The primary milestone achieved is the complete automation, refinement, and validation of the purely mathematical Python code—specifically the **Hidden Markov Model (HMM)** simulation. The AI-PC now routinely syncs real-world data, applies advanced clustering (mean return + standard deviation volatility), and exports the tracked regimes to a local SQLite database for downstream AI and dashboard consumption.

### ✅ What We've Achieved
1. **Environment Initialization:**
   * Resolved strict `pip` environments on Arch Linux. Built a dedicated Python `venv`.
   * Added `yfinance` to `requirements.txt` to bypass webserver API data limits.
2. **Direct Market Data Ingestion (yfinance):**
   * Modified `WebserverClient` and task scripts to conditionally use `source="yfinance"`, pulling extensive multi-year historical datasets required for accurate matrix generation.
3. **Math & Pandas Data Debugging (Markov Chain):**
   * **Mathematical Accuracy:** Fixed a critical bug in `markov_chain_detector.py` where regime prediction confidence was stuck at 100% horizontally. Replaced point-in-time extraction with full-sequence `predict_proba(...)[-1]` analysis.
   * **Array Alignment:** Corrected index masking offset (`vol_window - 1`) in `filter_returns_by_regime` to accurately map historical returns to their identified states.
   * **Intelligent Regime Labeling:** Completely rewrote `_identify_regime_order`. HMM states are no longer strictly ranked by mean return. The model now actively classifies 4 states dynamically (Bull, Bear, Sideways / Quiet, High Volatility Chop) by cross-referencing directional returns against standard deviation (volatility).
4. **Daily SQLite Automation Batching (HMM & Monte Carlo):**
   * Created new autonomous scripts: `tasks/sqlite_regime_task.py` and `tasks/sqlite_monte_carlo_task.py`.
   * Built a robust SQLite schema (`regimes.db -> daily_regimes` and `daily_monte_carlo` tables).
   * **Dynamic Asset Discovery:** The runtime automatically connects to the server's `financial_data.db` to infer and process over 700 active stock tickers dynamically.
   * **Phase 1 (HMM):** Trains unique 60-day predictive HMM models for each stock and upserts the current state, confidence, mean return, and volatility to the database.
   * **Phase 2 (Monte Carlo):** Directly queries the target HMM properties from `regimes.db`, executing 10,000 forward paths over a 20-day horizon to calculate tail-risk metrics (Value at Risk 95/99, Expected Shortfall, Probability of Loss).
5. **Git Repository Hygiene & Filtering:**
   * Used `git filter-branch` to purge massive 100MB+ `.db` and trained `.pkl` model footprints from the repository's deep commit history (which were causing GitHub `push` rejections).
   * Automatically generated a `.gitignore` covering `*.db`, `*.pkl`, pycache, and virtual environments.

---

## 🔍 Key Findings
* **Webserver Port Split:** The Webserver exposes `FastMCP` separately from its API data payloads. Tools using `mcp.client.sse` must hit `9876`. Basic REST queries (`get_prices()`, `post_markov_state()`) must hit `9875`.
* **Limited Server History:** The `/api/data/latest` endpoint currently returns very sparse historical data (~30 price points). For highly complex ML logic like Markov distributions tracking 5 parameters (resulting in matrices with over 4,000,000 parameters behind the scenes), having small datasets leads to `degenerate solution` warnings. In the future, the backend Webserver ought to supply slightly more time steps per asset or pull larger lookback windows.

---

## ⏭️ Next Steps: Building Pillars 3, 4, and 5

With Phase 1 (HMM) and Phase 2 (Monte Carlo) fully completed, the rigorous statistical math pipeline is fully operational. The next phases transition entirely into AI-driven logic and portfolio construction:

1. **Pillar 3: The LLM "Conviction Synthesis" (`llm_task.py`)**
   * Wire up the local **Ollama** model (e.g., `deepseek-r1:1.5b`) to act as the Senior Analyst. *Note: Ensure the target model is dynamically configurable via the `.env` file for fast local testing.*
   * Build the SQLite loop: Read `daily_monte_carlo` records (VaR, Mean Return, Probability of Loss), parse that quantitative data into a prompt, and ask the LLM to output a modified **Conviction Score** (`p_final`).
   * *Phase 3B:* Integrate RAG (Retrieval-Augmented Generation) via the MCP connection to inject qualitative context (SEC filings, news) so the LLM can fuse the *Numbers* with the *Narrative*.

2. **Pillar 4: The "Risk-Factor Envelopes" (Portfolio Construction)**
   * Create a new execution script (e.g., `tasks/portfolio_builder.py`) to systematically filter the 700+ scored assets into 5 distinct Strategic Asset Allocation (SAA) profiles (Conservative -> Aggressive).
   * Enforce strict bucket constraints (e.g., matching the Conservative profile exclusively to low-beta stocks and bonds using the "Cynical Auditor" LLM conviction scores).

3. **Pillar 5: The "Gap-Filler" Priority Queue**
   * Develop the transactional algorithm to handle recurring monthly deposits (e.g., €500/month inflow).
   * Automatically calculate shortfalls within the target portfolio envelopes and generate optimal fractional buy-tickets to dynamically "buy the dip" and rebalance without triggering excessive commission fees.

4. **Verify the Daemon Loop in Production**
   * Link all phases end-to-end within the daily daemon schedule (`daemon.py` / `systemd`), ensuring Phase 3 (LLM) triggers strictly following the successful completion of Phase 1 and 2 database writes.