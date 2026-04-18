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

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

logger = logging.getLogger(__name__)

def init_llm_table(conn):
    """Initialize SQLite table for storing daily LLM conviction scores."""
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
            conviction_score REAL,
            reasoning TEXT,
            UNIQUE(date, ticker)
        )
    ''')
    conn.commit()

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

def run_sqlite_llm_task(db_path="regimes.db"):
    """
    Fetch the HMM and Monte Carlo metrics for today from regimes.db.
    Feed them to the local Ollama LLM to generate a Conviction Score (-1.0 to 1.0).
    Save the results back to the database.
    """
    logger.info(f"=== Starting Daily SQLite LLM Conviction Task ({OLLAMA_MODEL}) ===")
    
    today_str = datetime.date.today().isoformat()
    
    conn = sqlite3.connect(db_path)
    init_llm_table(conn)
    cursor = conn.cursor()
    
    # 1. Fetch latest HMM + MC properties for today
    # JOIN daily_regimes and daily_monte_carlo
    cursor.execute('''
        SELECT r.ticker, r.current_state, m.var_95, m.mean_expected_return, m.prob_loss
        FROM daily_regimes r
        JOIN daily_monte_carlo m ON r.ticker = m.ticker AND r.date = m.date
        WHERE r.date = ?
    ''', (today_str,))
    
    assets = cursor.fetchall()
    
    if not assets:
        logger.warning(f"No joined HMM/MC data found for {today_str}. Did you run Phase 1 & 2?")
        return

    logger.info(f"Found {len(assets)} assets to analyze with {OLLAMA_MODEL}...")
    
    for ticker, state, var_95, mean_ret, prob_loss in assets:
        logger.info(f"Analyzing {ticker}...")
        
        # 2. Formulate the strictly constrained Prompt
        prompt = f"""You are a quantitative financial analyst evaluating the asset {ticker}.
Based on our math models for today:
- Market Regime State: {state}
- 20-Day 95% Value at Risk (VaR-95): {var_95:.2%}
- Expected Mean Return: {mean_ret:.2%}
- Probability of Loss: {prob_loss:.1%}

Evaluate this asset's risk/reward profile. Convert this quantitative risk profile into a single Conviction Score between -1.0 (Strong Sell) and 1.0 (Strong Buy).

Respond ONLY with a valid JSON object in this exact format, with no markdown formatting or extra text:
{{
  "conviction_score": <float between -1.0 and 1.0>,
  "reasoning": "<short 1 sentence explanation>"
}}
"""
        
        # 3. Query local Ollama
        llm_response = query_ollama(prompt)
        
        score = llm_response.get("conviction_score")
        reasoning = llm_response.get("reasoning", "LLM failed to provide reasoning.")
        
        # Fallback if the LLM hallucinated a non-float
        if not isinstance(score, (int, float)):
            logger.warning(f"LLM returned invalid score type for {ticker}: {score}. Defaulting to 0.0")
            score = 0.0
            
        # Clamp score between -1 and 1
        score = max(-1.0, min(1.0, float(score)))

        logger.info(f"✅ {ticker} -> Score: {score} | Reason: {reasoning}")
        
        # 4. Save to Database
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_llm_conviction 
                (date, ticker, hmm_state, var_95, mean_expected_return, prob_loss, conviction_score, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (today_str, ticker, state, float(var_95), float(mean_ret), float(prob_loss), score, reasoning))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save {ticker} to DB: {e}")
            
    conn.close()
    logger.info("=== LLM Conviction Task Completed ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_sqlite_llm_task()
