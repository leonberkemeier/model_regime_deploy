# 🚀 Next Steps: Building the Agentic MCP + RAG Pipeline

With the `deploy_on_ai-pc` folder structured and **Gemma 4** running via Ollama on the LLM Server, we are ready to upgrade the system from a static script into a true **Autonomous Financial Agent**.

Here is the exact step-by-step roadmap to implement the Model Context Protocol (MCP) and Retrieval-Augmented Generation (RAG) architecture over Tailscale:

---

## 🛠️ Step 1: Build the MCP Server (On the Webserver)
*The Webserver holds the data, so it must host the MCP Server to expose that data as Tools.*

1. **Create the Project:** Create a new `webserver_api/` folder on the Webserver.
2. **Setup the MCP Server:** Use the official Python `mcp` SDK to create an MCP Server that communicates over **SSE (Server-Sent Events) HTTP** (perfect for Tailscale).
3. **Define the Tools:**
   - `@mcp.tool() get_quantitative_risk(ticker: str)`: Queries the SQL database for the Phase 1 & 2 math (Markov Regime, Monte Carlo VaR).
   - `@mcp.tool() get_macro_indicators()`: Queries the database for the latest FRED macro data.
   - `@mcp.tool() search_filings(ticker: str, query: str)`: (Placeholder for Step 2) Queries the Vector DB for SEC earnings text.

## 📚 Step 2: Build the RAG Embeddings Pipeline (On the Webserver)
*Before Gemma 4 can search SEC filings, we need to embed them into a Vector Database.*

1. **Choose a Vector DB:** Integrate `pgvector` inside your existing PostgreSQL database, or spin up a lightweight `ChromaDB` instance on the Webserver.
2. **Create the Ingestion Script:** Write a python script for the Data Aggregator that takes raw SEC 10-K/10-Q text and Earnings Call transcripts.
3. **Chunk & Embed:** Use a fast, local embedding model (e.g., `all-MiniLM-L6-v2`) to turn paragraphs into vectors and save them.
4. **Connect to MCP:** Map the vector search logic into the `search_filings` MCP tool from Step 1.

## 🧠 Step 3: Upgrade the AI PC to an Agent (`deploy_on_ai-pc`)
*Now that the tools exist remotely, we need to teach Gemma 4 how to use them.*

1. **Install MCP Client:** Add the `mcp` client library to the AI PC's `requirements.txt`.
2. **Rewrite `tasks/llm_task.py`:**
   - Connect to the Webserver's MCP SSE URL over Tailscale.
   - Initialize an Agent loop (using a framework like `LangChain` or native tool-calling with Ollama).
   - Pass the MCP tool schemas directly to Gemma 4.
3. **The Agent Loop in Action:** 
   - Ask Gemma 4 to score an asset. 
   - Gemma 4 will autonomously halt, request to call the `get_quantitative_risk` tool, wait for the AI PC's MCP Client to fetch it from the Webserver, read the math, request to call `search_filings`, read the earnings, and then finally output the JSON `score` and `reasoning`.

---

## 🎯 Immediate Next Action
Create the `webserver_api/` directory on the Webserver and write the boilerplate FastMCP (SSE) Server to expose the first tool (`get_quantitative_risk`).