# AI-PC Deployment & Testing Progress (Apr 14, 2026)

## 🎯 Current Project Status
We have successfully deployed the **AI-PC compute node** (running Arch Linux) and connected its local analytical scripts to the central financial **Webserver (MCP + REST)** endpoints. 

The primary milestone achieved is the complete automation and validation of the purely mathematical Python code—specifically the **Hidden Markov Model (HMM)** simulation—against real-time Webserver data rather than synthetic mock data. The AI-PC now routinely syncs over the Zero-Trust Tailscale VPN and parses the real financial API feed successfully.

### ✅ What We've Achieved
1. **Environment Initialization:**
   * Resolved strict `pip` environments on Arch Linux. Built a dedicated Python `venv`.
   * Installed critical ML, Data, and AI libs (`numpy`, `pandas`, `hmmlearn`, `mcp`, `python-dotenv`, `loguru`).
2. **Daemonization:**
   * Converted the Python scheduler into a robust background `systemd` service (`model-agent.service`).
3. **Agentic MCP Implementation:**
   * Fully modernized `tasks/llm_task.py` to be a generic MCP client. It loops natively through Ollama (`gemma4:e4b`) to request remote tool execution via `mcp.client.sse`.
4. **Network & Routing Corrections:**
   * Diagnosed the 404 network failure. We discovered the Webserver splits its traffic across two ports:
     * `9876`: FastMCP SSE Server
     * `9875`: REST API (FastAPI/Uvicorn)
   * Re-routed all `WebserverClient` data fetching to the proper `:9875` REST interface.
5. **Math & Pandas Data Debugging (Markov Chain):**
   * **Data Ingestion:** Mapped FastAPI's JSON `list` responses seamlessly into Pandas Multi-Asset DataFrames.
   * **Missing Data Handling:** Real world OHLCV contains `NaN` gaps (e.g., from holiday/weekend discrepancies). We patched the pipeline with `ffill().bfill()` to avoid triggering division-by-zero crashes within the `sklearn` scalers.
   * **Dynamic Rolling Windows:** Handled tiny datasets (the Webserver only returned 30 days history) by applying a fallback `vol_window` to successfully compile historical features without raising "0 samples" exceptions.
   * **Regime Status Types:** Fixed aggregation crashes when evaluating multi-asset `pd.Series` into JSON serializable floats.
6. **REST Schema Alignment:**
   * Remapped the Markov detector's output `current_regime` from a String (e.g. "Bull") back to an Integer ID before submitting the `POST /api/analysis/markov` request to bypass the Webserver’s `422 Unprocessable Entity` Pydantic strict schemas.
7. **Git Repository Hygiene:**
   * Automatically generated a `.gitignore` to prevent 70MB+ trained model footprint blobs (`*.pkl`), pycache, and virtual environments from crashing GitHub pushes.
   * Removed pre-existing `models/markov_regime_model.pkl` from GitHub's index blob.

---

## 🔍 Key Findings
* **Webserver Port Split:** The Webserver exposes `FastMCP` separately from its API data payloads. Tools using `mcp.client.sse` must hit `9876`. Basic REST queries (`get_prices()`, `post_markov_state()`) must hit `9875`.
* **Limited Server History:** The `/api/data/latest` endpoint currently returns very sparse historical data (~30 price points). For highly complex ML logic like Markov distributions tracking 5 parameters (resulting in matrices with over 4,000,000 parameters behind the scenes), having small datasets leads to `degenerate solution` warnings. In the future, the backend Webserver ought to supply slightly more time steps per asset or pull larger lookback windows.

---

## ⏭️ Next Steps

1. **Validate the Monte Carlo Module (`monte_carlo_task.py`)**
   * Verify that the Monte Carlo task pulls the newly posted `MarkovRegimeState` properly.
   * Test if it evaluates the simulated asset future accurately.
   * Ensure that `POST /api/analysis/monte_carlo/all` validates the same strict Pydantic schemas.

2. **Validate the LLM Scoring Module (`llm_task.py`)**
   * Boot up the MCP loop over `http://100.69.76.20:9876`.
   * Watch the local `gemma4:e4b` Ollama node natively pick up the tools from the remote server, analyze the Markov/MC stats, and synthesize an asset sentiment score payload.

3. **Verify the Daemon Loop in Production**
   * Now that testing local manual tasks passed (`run_markov_task(...)`), allow the AI-PC daemon (`systemd`) to autonomously run the entire scheduled queue of tasks overnight.
   * Check `/var/log/syslog` tomorrow to see if jobs sequence accurately.

4. **Webserver Backend Expansion (Optional)**
   * Determine if the main FastAPI Webserver server requires modifications to serve longer historical datasets (e.g., 365 Days) to build more stable and comprehensive Regime matrices.