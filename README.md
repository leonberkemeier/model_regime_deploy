# AI/ML Compute Node (`deploy_on_ai-pc`)

## 🌍 The Grand Vision: The Full Financial Pipeline
This repository is one module of a larger **Distributed Financial Portfolio System**. The complete system is designed to autonomously scrape financial data, detect market regimes, simulate risk, use AI to score assets, and automatically execute mock trades in a portfolio simulator.

To maximize efficiency and minimize cloud hosting costs, the system is split into two main physical locations connected securely via **Tailscale** (Zero-Trust VPN):
1. **The Webserver (Always-On, Low Compute):** Hosts the central Database, Data Scrapers, REST API, and Trading Simulator web frontend.
2. **The AI Node (On-Demand, High Compute):** This folder (`deploy_on_ai-pc`). Runs on a local PC with a GPU to handle heavy math and AI inference without paying for expensive cloud GPUs.

---

## 🎯 Purpose of THIS Component (`deploy_on_ai-pc`)
The `deploy_on_ai-pc` folder is the **"Heavy-Lifting Brain"** of the architecture. 

Instead of keeping an expensive AI/ML server running 24/7, this lightweight Python daemon sleeps most of the day. When scheduled (e.g., 08:00 AM daily), it wakes up and executes the following sequence:

1. **Fetches Context:** Reaches out to the Webserver via REST API to download the latest aggregated OHLCV asset prices.
2. **Markov Chain Detection (Phase 1):** Uses `hmmlearn` to analyze the price data and determine the current overall "Market Regime" (e.g., Bull Market, High Volatility, Bear Market).
3. **Enhanced Monte Carlo (Phase 2):** Runs 10,000 simulations per asset using `numpy` and `pandas` to calculate maximum drawdown, Value at Risk (VaR), and Expected Shortfall based on the current regime.
4. **Local LLM Scoring & RAG (Phase 3):** The locally running **Ollama** model (e.g., gemma4:e4b) directly queries an **SQLite database** for structured financial metrics and performs Retrieval-Augmented Generation (**RAG**) against a **Vector Database** containing unstructured data (reports, news). Combining these sources, it calculates and assigns a 0.0 to 1.0 "Suitability Score".
5. **Data Push & Trigger (Phase 4):** Pushes all these scores and predictive states formatting as JSON back to the Webserver's database, then triggers the Webserver to build the final trading portfolio.
6. **Sleeps:** Shuts down active processing until the next day.

By isolating these tasks here, your web server never gets bogged down by heavy matrix multiplication or LLM token generation!

---

## 🛠️ The Tech Stack (AI Node)
- **Core Orchestration:** Pure Python 3.9+, `schedule` library for chronological execution.
- **Data & Knowledge:** SQLite for structured data querying, paired with a Vector Database for unstructured Retrieval-Augmented Generation (RAG).
- **Machine Learning / Statistics:** `scikit-learn`, `numpy`, `pandas`, `hmmlearn`.
- **Generative AI:** **Ollama** (Running 100% locally for zero-cost, private LLM inference).
- **Networking:** `requests` for standard HTTP, routed securely over **Tailscale**.

---

## 🚀 How to Run It

1. **Install Dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Copy the example config and add your Webserver's Tailscale MagicDNS or IP.
   ```bash
   cp .env.example .env
   # Edit .env with nano/vim to set WEBSERVER_URL
   ```

3. **Start the Local LLM (Separate Terminal):**
   Ensure Ollama is running in the background.
   ```bash
   ollama serve
   ```

4. **Start the Daemon:**
   ```bash
   python daemon.py
   ```
   *The daemon will now idle and automatically run the `markov -> monte_carlo -> llm` pipeline at the configured time.*