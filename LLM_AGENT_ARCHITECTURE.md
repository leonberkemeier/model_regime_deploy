# AI Agent Architecture: MCP + RAG Integration

This document outlines how the `deploy_on_ai-pc` node upgrades from a simple text-prompting script into an **Autonomous Financial Agent**. The LLM (Ollama) will use the **Model Context Protocol (MCP)** to actively query both structured math and unstructured text over the Tailscale network.

---

## 🏗️ 1. Architecture Updates Needed

To make this work, the responsibilities are split between our two main machines:

### The Central Webserver (The Data Hub)
1. **Vector Database:** Adds a Vector DB (like `pgvector` inside PostgreSQL or a lightweight `ChromaDB` instance). The Data Aggregator (cron 6 AM) will embed SEC 10-K/10-Q and earnings transcripts here.
2. **MCP Server:** Exposes a standard MCP endpoint over Tailscale. Instead of standard REST, it provides "Tools" that any MCP-compatible LLM agent can call.

### The AI / ML Node (`deploy_on_ai-pc`)
1. **MCP Client:** Translates the LLM's requests into MCP tool calls.
2. **Agentic Loop:** Updates `tasks/llm_task.py` to use a framework like `LangChain`, `LlamaIndex`, or `smolagents` to allow the local Ollama model to loop through thought -> action -> observation.

---

## 🛠️ 2. The MCP Tool Definitions

When Ollama starts analyzing an asset, we will provide it with these three specific tools exposed by the Webserver's MCP Server:

### Tool 1: `get_quantitative_risk` (Structured Data)
* **Description:** Retrieves the latest Phase 1 & Phase 2 mathematical risk metrics for a given asset.
* **Parameters:**
  * `ticker` (string): The stock or crypto ticker (e.g., "AAPL").
* **Returns (from SQL DB):**
  ```json
  {
    "current_regime": "High Volatility Bear",
    "monte_carlo_var_95": -0.045,
    "expected_return": 0.08,
    "probability_of_loss": 0.35
  }
  ```

### Tool 2: `search_filings_and_earnings` (Unstructured RAG Data)
* **Description:** Performs a semantic vector search over the company's recent SEC EDGAR filings, 10-Ks, and earnings call transcripts.
* **Parameters:**
  * `ticker` (string): The asset ticker.
  * `query` (string): What the LLM specifically wants to know (e.g., "AI revenue growth guidance", "supply chain risks", "CEO sentiment").
* **Returns (from Vector DB):**
  List of the Top 3 most relevant text chunks containing direct quotes from the company.

### Tool 3: `get_macro_economic_indicators` (Structured Data)
* **Description:** Retrieves current FRED macroeconomic data vital for context.
* **Parameters:** None (global context).
* **Returns:** US Interest Rates, Inflation (CPI), and GDP Growth trends.

---

## 🔄 3. The New `llm_task.py` Execution Flow

Here is exactly how the script will guide the LLM through scoring an asset:

1. **The System Prompt:**
   > "You are an expert quantitative hedge fund manager. Your job is to score assets from 0.0 to 1.0. You MUST use your tools to check the mathematical risk (VaR) and the fundamental sentiment (Earnings/SEC filings) before making a decision."
   
2. **The Agent Loop (Example: Analyzing NVDA):**
   * **Thought:** I need to analyze NVDA. First, I'll check its mathematical risk in the current market regime.
   * **Action:** Calls `get_quantitative_risk("NVDA")`.
   * **Observation:** MCP returns `{"current_regime": "Bull", "var_95": -0.08...}`
   * **Thought:** The VaR is slightly high, I need to see if their future guidance justifies this risk. I will search their recent earnings.
   * **Action:** Calls `search_filings_and_earnings("NVDA", "future revenue guidance data centers")`.
   * **Observation:** RAG returns quotes from Jensen Huang about next-gen Blackwell chip demand.
   * **Thought:** The fundamental RAG data is overwhelmingly positive and outweighs the Monte Carlo risk. I am ready to score.
   
3. **The Final Output:**
   The LLM generates the final payload:
   ```json
   {
     "ticker": "NVDA",
     "score": 0.92,
     "reasoning": "Despite a moderately high 95% VaR of 8% in the current macro regime, RAG text analysis of the recent 10-Q shows unprecedented backlog demand for next-gen silicon, heavily skewing the expected return positively."
   }
   ```

4. **Saving:** `llm_task.py` takes this JSON and POSTs it directly to the Webserver's database, then triggers portfolio creation.

---

## 🚀 Next Steps for Implementation

To make this a reality, we need to:
1. Choose an Agent framework for `deploy_on_ai-pc` (e.g., `LangChain` or just manual tool parsing with Ollama's native tool support).
2. Choose a Vector DB for the Webserver (e.g., `pgvector` inside the same PostgreSQL DB, meaning zero extra infrastructure).
3. Build the actual MCP Server Python file on the Webserver that defines these tools.