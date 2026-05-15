import logging
import sqlite3
import datetime
import sys
import json
import requests
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load local environment settings
from dotenv import load_dotenv
import os

load_dotenv(Path(project_root) / ".env")

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

# Maximum number of candidates passed to the LLM.
TOP_N_CANDIDATES = int(os.getenv("LLM_TOP_N_CANDIDATES", "25"))

# Agentic mode: set both to enable ReAct tool-calling loop.
# Falls back to single-shot if MCP server is unreachable.
LLM_AGENTIC_MODE = os.getenv("LLM_AGENTIC_MODE", "false").lower() == "true"
MCP_SERVER_URL   = os.getenv("MCP_SERVER_URL", "")
AGENT_MAX_ITERS  = int(os.getenv("AGENT_MAX_ITERS", "5"))

# Ollama tool schemas for the 3 MCP tools
_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_quantitative_risk_tool",
            "description": (
                "Retrieve the latest HMM market regime and Monte Carlo risk metrics "
                "(VaR-95, ES-95, mean return, probability of loss) for a stock ticker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_filings_tool",
            "description": (
                "Semantic search over SEC 10-K/10-Q filings for a company. "
                "Use this to find qualitative evidence about revenue guidance, risks, or strategy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "query":  {"type": "string", "description": "What to search for in the filings"},
                },
                "required": ["ticker", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_indicators_tool",
            "description": (
                "Get current macroeconomic indicators: GDP growth, unemployment, "
                "CPI inflation, and Federal Funds Rate."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

logger = logging.getLogger(__name__)

def init_llm_table(conn):
    """Initialize SQLite table for storing daily fused LLM conviction scores."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_llm_conviction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            hmm_state TEXT,
            var_95 REAL,
            mean_expected_return REAL,
            prob_loss REAL,
            hmm_state_14d TEXT,
            hmm_state_60d TEXT,
            var_95_14d REAL,
            var_95_60d REAL,
            mean_expected_return_14d REAL,
            mean_expected_return_60d REAL,
            prob_loss_14d REAL,
            prob_loss_60d REAL,
            conviction_score REAL,
            reasoning TEXT,
            UNIQUE(date, ticker)
        )
    ''')

    # Backward-compatible migration for existing DBs
    cursor.execute("PRAGMA table_info(daily_llm_conviction)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    required_columns = {
        "hmm_state_14d": "TEXT",
        "hmm_state_60d": "TEXT",
        "var_95_14d": "REAL",
        "var_95_60d": "REAL",
        "mean_expected_return_14d": "REAL",
        "mean_expected_return_60d": "REAL",
        "prob_loss_14d": "REAL",
        "prob_loss_60d": "REAL",
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE daily_llm_conviction ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()


def _fmt_pct(value):
    """Format probability/return values for prompts."""
    if value is None:
        return "N/A"
    return f"{float(value):.2%}"


def _avg_available(*values):
    """Average non-null numeric values; return None if all values are null."""
    non_null = [float(v) for v in values if v is not None]
    if not non_null:
        return None
    return sum(non_null) / len(non_null)

def query_ollama(prompt: str) -> dict:
    """Send a prompt to the local Ollama instance and parse the JSON response."""
    url = f"{OLLAMA_HOST}/api/generate"
    
    # We enforce JSON output format
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result_text = response.json().get("response", "{}")
        return json.loads(result_text)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama connection error: {e}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama JSON: {e} - Raw: {result_text}")
        return {}

def _run_agentic_scoring(ticker: str, horizon_days: int) -> tuple[float, str]:
    """
    ReAct-style agentic loop: Ollama autonomously calls MCP tools before
    producing a final conviction score.

    Returns (conviction_score, reasoning).
    Falls back to (0.0, error_message) on failure.
    """
    from mcp_server.client import call_tool

    system_prompt = (
        f"You are a senior quantitative analyst at a hedge fund. "
        f"Your job is to produce a single conviction score for a stock ticker "
        f"on a scale from -1.0 (Strong Sell) to 1.0 (Strong Buy). "
        f"You MUST use your tools to gather evidence before scoring: "
        f"first check the quantitative risk metrics, then search for relevant "
        f"qualitative information in SEC filings, then check macro context if relevant. "
        f"Only after gathering evidence, output your final JSON verdict."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Analyze {ticker} and provide an investment conviction score "
                f"for a {horizon_days}-day horizon. Use your tools, then respond with:\n"
                f'{{"conviction_score": <float -1.0 to 1.0>, "reasoning": "<one sentence>"}}'   
            ),
        },
    ]

    for iteration in range(AGENT_MAX_ITERS):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages,
                      "tools": _AGENT_TOOLS, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            msg = resp.json().get("message", {})
        except Exception as exc:
            logger.error(f"[{ticker}] Ollama chat error: {exc}")
            return 0.0, f"Ollama error: {exc}"

        # ── Tool calls requested ──
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
            for tc in tool_calls:
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                logger.info(f"  [{ticker}] Tool call: {name}({args})")
                result = call_tool(name, args)
                logger.info(f"  [{ticker}] Tool result: {result[:120]}...")

                messages.append({"role": "tool", "content": result, "name": name})
            continue  # next iteration with tool results in context

        # ── Final response ──
        content = msg.get("content", "")
        if content:
            try:
                # Strip any markdown fences the model might add
                clean = content.strip().strip("```json").strip("```").strip()
                data  = json.loads(clean)
                score = data.get("conviction_score", 0.0)
                reason = data.get("reasoning", "No reasoning provided.")
                if isinstance(score, (int, float)):
                    return max(-1.0, min(1.0, float(score))), reason
            except (json.JSONDecodeError, AttributeError):
                pass  # model may still be reasoning; continue loop

        if iteration == AGENT_MAX_ITERS - 1:
            logger.warning(f"[{ticker}] Agent reached max iterations without a score.")

    return 0.0, "Agent did not produce a conviction score within iteration limit."


def _get_top_candidates(cursor, today_str: str, horizon_days: int, top_n: int) -> set[str]:
    """
    SQL pre-filter: rank all tickers by Calmar ratio and return the top N.

    Calmar ratio = avg_mean_expected_return / ABS(avg_var_95)
    Higher = better risk-adjusted return relative to downside.

    Hard gates applied before ranking:
      - mean_expected_return > 0   (positive expected return)
      - prob_loss < 0.55           (not more likely to lose than win)

    Averaging across 14d and 60d lookbacks gives a more stable signal
    than using either window alone.
    """
    cursor.execute('''
        SELECT ticker
        FROM (
            SELECT
                r.ticker,
                AVG(m.mean_expected_return)                         AS avg_return,
                AVG(m.var_95)                                       AS avg_var_95,
                AVG(m.prob_loss)                                    AS avg_prob_loss,
                CASE
                    WHEN AVG(ABS(m.var_95)) > 0
                    THEN AVG(m.mean_expected_return) / AVG(ABS(m.var_95))
                    ELSE -999
                END                                                 AS calmar_ratio
            FROM daily_regimes r
            INNER JOIN daily_monte_carlo m
                ON  r.ticker        = m.ticker
                AND r.date          = m.date
                AND r.lookback_days = m.lookback_days
                AND m.horizon_days  = ?
            WHERE r.date = ?
              AND m.mean_expected_return IS NOT NULL
              AND m.var_95               IS NOT NULL
            GROUP BY r.ticker
            HAVING AVG(m.mean_expected_return) > 0
               AND AVG(m.prob_loss)            < 0.55
            ORDER BY calmar_ratio DESC
            LIMIT ?
        ) ranked
    ''', (horizon_days, today_str, top_n))

    return {row[0] for row in cursor.fetchall()}


def run_sqlite_llm_task(db_path="regimes.db", horizon_days=20):
    """
    Fetch the HMM and Monte Carlo metrics for today from regimes.db.
    Feed them to the local Ollama LLM to generate a Conviction Score (-1.0 to 1.0).
    Save the results back to the database.
    """
    # Decide whether to use agentic mode (requires MCP server reachable)
    use_agentic = False
    if LLM_AGENTIC_MODE and MCP_SERVER_URL:
        from mcp_server.client import is_server_available
        use_agentic = is_server_available()
        if use_agentic:
            logger.info(f"Agentic mode ENABLED — MCP server reachable at {MCP_SERVER_URL}")
        else:
            logger.warning(f"Agentic mode requested but MCP server unreachable ({MCP_SERVER_URL}). Falling back to single-shot.")

    logger.info(f"=== Starting Daily SQLite LLM Conviction Task ({OLLAMA_MODEL}) ===")
    
    today_str = datetime.date.today().isoformat()
    
    conn = sqlite3.connect(db_path)
    init_llm_table(conn)
    cursor = conn.cursor()
    
    # 1. Fetch 14d and 60d HMM + MC properties for today and fuse them per ticker
    cursor.execute('''
        SELECT
            r.ticker,
            MAX(CASE WHEN r.lookback_days = 14 THEN r.current_state END) AS hmm_state_14d,
            MAX(CASE WHEN r.lookback_days = 60 THEN r.current_state END) AS hmm_state_60d,
            MAX(CASE WHEN m.lookback_days = 14 THEN m.var_95 END) AS var_95_14d,
            MAX(CASE WHEN m.lookback_days = 60 THEN m.var_95 END) AS var_95_60d,
            MAX(CASE WHEN m.lookback_days = 14 THEN m.mean_expected_return END) AS mean_expected_return_14d,
            MAX(CASE WHEN m.lookback_days = 60 THEN m.mean_expected_return END) AS mean_expected_return_60d,
            MAX(CASE WHEN m.lookback_days = 14 THEN m.prob_loss END) AS prob_loss_14d,
            MAX(CASE WHEN m.lookback_days = 60 THEN m.prob_loss END) AS prob_loss_60d
        FROM daily_regimes r
        LEFT JOIN daily_monte_carlo m
            ON r.ticker = m.ticker
            AND r.date = m.date
            AND r.lookback_days = m.lookback_days
            AND m.horizon_days = ?
        WHERE r.date = ?
        GROUP BY r.ticker
        ORDER BY r.ticker
    ''', (horizon_days, today_str))
    
    assets = cursor.fetchall()
    
    if not assets:
        logger.warning(f"No joined HMM/MC data found for {today_str}. Did you run Phase 1 & 2?")
        return

    logger.info(f"Universe: {len(assets)} assets with HMM+MC data for {today_str}.")

    # ── Pre-filter: rank by Calmar ratio, keep only top N ──────────────────
    top_candidates = _get_top_candidates(cursor, today_str, horizon_days, TOP_N_CANDIDATES)

    if not top_candidates:
        logger.warning(
            "Pre-filter returned 0 candidates. "
            "Possible causes: no positive-return assets today, or Phase 2 hasn't run yet."
        )
        return

    # Always include currently held portfolio tickers so they get a fresh
    # score today — even if they've drifted below the Calmar rank threshold.
    # Without this, a degrading held position never triggers a violation.
    try:
        cursor.execute("""
            SELECT DISTINCT ticker FROM model_portfolio_positions
            WHERE build_date = (SELECT MAX(build_date) FROM model_portfolio_positions)
              AND ticker != 'CASH'
        """)
        held_in_portfolio = {row[0] for row in cursor.fetchall()}
    except Exception:
        held_in_portfolio = set()  # table may not exist on first run

    forced_extra = held_in_portfolio - top_candidates
    if forced_extra:
        logger.info(
            f"{len(forced_extra)} held ticker(s) outside top {TOP_N_CANDIDATES} "
            f"forced into scoring: {sorted(forced_extra)}"
        )
        top_candidates |= forced_extra

    assets = [a for a in assets if a[0] in top_candidates]
    logger.info(
        f"Pre-filter: {len(assets)} candidates "
        f"(top {TOP_N_CANDIDATES} by Calmar + {len(forced_extra)} held tickers)."
    )
    # ───────────────────────────────────────────────────────────────────────
    
    for (
        ticker,
        state_14d,
        state_60d,
        var_95_14d,
        var_95_60d,
        mean_ret_14d,
        mean_ret_60d,
        prob_loss_14d,
        prob_loss_60d,
    ) in assets:
        logger.info(f"Analyzing {ticker}...")

        if all(v is None for v in [var_95_14d, var_95_60d, mean_ret_14d, mean_ret_60d, prob_loss_14d, prob_loss_60d]):
            logger.warning(f"Skipping {ticker}: no Monte Carlo metrics found for either 14d or 60d lookback.")
            continue

        state_14d_display = state_14d or "N/A"
        state_60d_display = state_60d or "N/A"

        fused_hmm_state = f"14d={state_14d_display} | 60d={state_60d_display}"
        fused_var_95 = _avg_available(var_95_14d, var_95_60d)
        fused_mean_ret = _avg_available(mean_ret_14d, mean_ret_60d)
        fused_prob_loss = _avg_available(prob_loss_14d, prob_loss_60d)
        
        # 2. Score the ticker ─ agentic (MCP tools) or single-shot
        if use_agentic:
            score, reasoning = _run_agentic_scoring(ticker, horizon_days)
        else:
            prompt = f"""You are a quantitative financial analyst evaluating the asset {ticker}.
Based on two horizon-aware model views for today, produce one consolidated investment conviction.

Short-Term View (14-day regime):
- Market Regime State: {state_14d_display}
- {horizon_days}-Day 95% Value at Risk (VaR-95): {_fmt_pct(var_95_14d)}
- Expected Mean Return: {_fmt_pct(mean_ret_14d)}
- Probability of Loss: {_fmt_pct(prob_loss_14d)}

Long-Term View (60-day regime):
- Market Regime State: {state_60d_display}
- {horizon_days}-Day 95% Value at Risk (VaR-95): {_fmt_pct(var_95_60d)}
- Expected Mean Return: {_fmt_pct(mean_ret_60d)}
- Probability of Loss: {_fmt_pct(prob_loss_60d)}

Evaluate this asset's risk/reward profile by balancing short-term and long-term signals.
Return a single Conviction Score between -1.0 (Strong Sell) and 1.0 (Strong Buy).

Respond ONLY with a valid JSON object in this exact format, with no markdown formatting or extra text:
{{
  "conviction_score": <float between -1.0 and 1.0>,
  "reasoning": "<short 1 sentence explanation>"
}}
"""
            llm_response = query_ollama(prompt)
            score    = llm_response.get("conviction_score", 0.0)
            reasoning = llm_response.get("reasoning", "LLM failed to provide reasoning.")
            if not isinstance(score, (int, float)):
                logger.warning(f"LLM returned invalid score for {ticker}: {score}. Defaulting to 0.0")
                score = 0.0
            score = max(-1.0, min(1.0, float(score)))

        logger.info(f"✅ {ticker} -> Score: {score} | Reason: {reasoning}")
        
        # 4. Save to Database
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_llm_conviction 
                (date, ticker, hmm_state, var_95, mean_expected_return, prob_loss,
                 hmm_state_14d, hmm_state_60d, var_95_14d, var_95_60d,
                 mean_expected_return_14d, mean_expected_return_60d,
                 prob_loss_14d, prob_loss_60d,
                 conviction_score, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today_str,
                ticker,
                fused_hmm_state,
                fused_var_95,
                fused_mean_ret,
                fused_prob_loss,
                state_14d,
                state_60d,
                var_95_14d,
                var_95_60d,
                mean_ret_14d,
                mean_ret_60d,
                prob_loss_14d,
                prob_loss_60d,
                score,
                reasoning,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save {ticker} to DB: {e}")
            
    conn.close()
    logger.info("=== LLM Conviction Task Completed ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_sqlite_llm_task()
