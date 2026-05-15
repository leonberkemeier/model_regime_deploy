"""
MCP Server — QuantAdvisor Financial Tools
==========================================
Exposes three tools via FastMCP over SSE transport.
Run this on the WEBSERVER so the AI PC can call tools over Tailscale.

    python mcp_server/server.py

The server listens on MCP_SERVER_HOST:MCP_SERVER_PORT (default 0.0.0.0:9876).
The AI PC connects via:
    MCP_SERVER_URL=http://your-server.tailnet.ts.net:9876

Tools exposed:
  get_quantitative_risk(ticker)       — HMM regime + Monte Carlo metrics
  search_filings(ticker, query)       — semantic search over SEC filings
  get_macro_indicators()              — latest FRED macro data
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from mcp.server.fastmcp import FastMCP
from mcp_server.tools import (
    get_quantitative_risk, search_filings, get_macro_indicators,
    get_trade_history, get_realignment_candidates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MCP_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_SERVER_PORT", "9876"))

# ── FastMCP app ───────────────────────────────────────────────────────────────

mcp = FastMCP(
    "QuantAdvisorServer",
    instructions=(
        "You are connected to the Quant Advisor financial data server. "
        "Use get_quantitative_risk to fetch HMM regime and Monte Carlo risk metrics, "
        "search_filings to retrieve relevant SEC filing passages, and "
        "get_macro_indicators for the current macroeconomic context."
    ),
)


@mcp.tool()
def get_quantitative_risk_tool(ticker: str) -> str:
    """
    Retrieve the latest Hidden Markov Model market regime and Monte Carlo
    risk metrics for a given stock ticker.

    Returns JSON with: hmm_state_14d, hmm_state_60d, var_95, es_95,
    mean_return, prob_loss, and the data date.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "NVDA")
    """
    result = get_quantitative_risk(ticker)
    return json.dumps(result, default=str)


@mcp.tool()
def search_filings_tool(ticker: str, query: str) -> str:
    """
    Perform a semantic search over SEC 10-K and 10-Q filings for a company.
    Returns the top 3 most relevant text passages from the filing database.

    Use this to answer qualitative questions like:
      - "What does management say about AI revenue guidance?"
      - "What are the main supply chain risks?"
      - "How is the company managing debt?"

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")
        query:  Natural language question or topic to search for
    """
    results = search_filings(ticker, query, n_results=3)
    return json.dumps(results, default=str)


@mcp.tool()
def get_macro_indicators_tool() -> str:
    """
    Retrieve current macroeconomic context from FRED data:
      - GDP growth
      - Unemployment rate (UNRATE)
      - Consumer Price Index / Inflation (CPIAUCSL)
      - Federal Funds Rate (FEDFUNDS)

    Use this to understand the broader economic environment when
    evaluating individual stock conviction.
    """
    result = get_macro_indicators()
    return json.dumps(result, default=str)


# ── Entry point ───────────────────────────────────────────────────────────────

@mcp.tool()
def get_realignment_candidates_tool(profile_id: int) -> str:
    """
    Portfolio Health Audit for a given risk profile.
    Returns two lists:
      - weakest_links: the held positions most at risk (ranked by weakness score
        combining low conviction, low win probability, high tail risk, low sentiment)
      - challengers: top 5 non-held assets by today's conviction score

    Use this before making swap or rebalancing decisions.

    Args:
        profile_id: Risk profile number (1=Conservative to 5=Aggressive)
    """
    return json.dumps(get_realignment_candidates(profile_id), default=str)


@mcp.tool()
def get_trade_history_tool(profile_id: int = None, days: int = 30) -> str:
    """
    Return the portfolio trade log — every buy, sell, swap, and rebuild
    decision made in the last `days` days, with a plain-English rationale
    for each one.

    Use this to understand WHY the portfolio changed, e.g.:
      - Why was NVDA sold?
      - What triggered the last portfolio rebuild?
      - What was bought as a replacement for a position that was exited?

    Args:
        profile_id: Risk profile 1-5. If omitted, returns history for all profiles.
        days:       How many days back to look (default: 30).
    """
    return json.dumps(get_trade_history(profile_id=profile_id, days=days), default=str)


if __name__ == "__main__":
    logger.info(f"Starting QuantAdvisor MCP Server on {MCP_HOST}:{MCP_PORT}")
    logger.info("Tools: get_quantitative_risk | search_filings | get_macro_indicators")
    mcp.run(transport="sse", host=MCP_HOST, port=MCP_PORT)
