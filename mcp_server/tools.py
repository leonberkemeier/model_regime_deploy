"""
MCP Tool Implementations
========================
Pure Python functions — no MCP dependencies.
These are wrapped by server.py into MCP tools, and called directly
by client.py when the server is remote.

All three tools are designed to run on the SERVER where:
  - regimes.db    ← synced from AI PC after each daily run
  - financial_data.db ← the primary webserver database
  - chroma_db/    ← ChromaDB populated by sec_ingestion_task.py
"""

import json
import logging
import os
import sqlite3
import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_base = Path(__file__).parent.parent

REGIMES_DB_PATH  = os.getenv("REGIMES_DB_PATH",  str(_base / "regimes.db"))
FINANCIAL_DB_PATH = os.getenv("FINANCIAL_DB_PATH", str(_base / "financial_data.db"))
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH",   str(_base / "chroma_db"))
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")

# Lazy-loaded ChromaDB client (only initialised if search_filings is called)
_chroma_client   = None
_raw_collection  = None


def _get_chroma_collection():
    """Lazy-init ChromaDB so the server can start without it."""
    global _chroma_client, _raw_collection
    if _raw_collection is not None:
        return _raw_collection
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _chroma_client  = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _raw_collection = _chroma_client.get_or_create_collection(
            name="sec_raw_chunks", embedding_function=ef
        )
        return _raw_collection
    except Exception as exc:
        logger.error(f"ChromaDB unavailable: {exc}")
        return None


# ── Tool 1: Quantitative Risk ─────────────────────────────────────────────────

def get_quantitative_risk(ticker: str) -> dict:
    """
    Return the latest HMM regime + Monte Carlo risk metrics for a ticker.

    Reads from regimes.db (daily_regimes + daily_monte_carlo).
    Returns the most recent date available (today or last run date).
    """
    ticker = ticker.upper().strip()
    result = {
        "ticker":           ticker,
        "date":             None,
        "hmm_state_14d":    None,
        "hmm_state_60d":    None,
        "var_95":           None,
        "es_95":            None,
        "mean_return":      None,
        "prob_loss":        None,
        "error":            None,
    }

    if not Path(REGIMES_DB_PATH).exists():
        result["error"] = f"regimes.db not found at {REGIMES_DB_PATH}"
        return result

    try:
        conn   = sqlite3.connect(REGIMES_DB_PATH)
        cursor = conn.cursor()

        # Get latest date available for this ticker
        cursor.execute(
            "SELECT MAX(date) FROM daily_regimes WHERE ticker = ?", (ticker,)
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            result["error"] = f"No regime data found for {ticker}"
            conn.close()
            return result

        latest_date = row[0]
        result["date"] = latest_date

        # Fetch 14d and 60d HMM states
        cursor.execute("""
            SELECT lookback_days, current_state, mean_return, volatility
            FROM daily_regimes
            WHERE ticker = ? AND date = ?
        """, (ticker, latest_date))
        for lookback, state, mean_ret, vol in cursor.fetchall():
            if lookback == 14:
                result["hmm_state_14d"] = state
            elif lookback == 60:
                result["hmm_state_60d"] = state

        # Fetch averaged Monte Carlo metrics (20-day horizon, across lookbacks)
        cursor.execute("""
            SELECT
                AVG(var_95)               AS var_95,
                AVG(es_95)                AS es_95,
                AVG(mean_expected_return) AS mean_return,
                AVG(prob_loss)            AS prob_loss
            FROM daily_monte_carlo
            WHERE ticker = ? AND date = ? AND horizon_days = 20
        """, (ticker, latest_date))
        mc = cursor.fetchone()
        if mc and mc[0] is not None:
            result["var_95"]      = round(mc[0], 4)
            result["es_95"]       = round(mc[1], 4)
            result["mean_return"] = round(mc[2], 4)
            result["prob_loss"]   = round(mc[3], 4)

        conn.close()
        return result

    except Exception as exc:
        result["error"] = str(exc)
        logger.error(f"get_quantitative_risk({ticker}): {exc}")
        return result


# ── Tool 2: Filing Search ─────────────────────────────────────────────────────

def search_filings(ticker: str, query: str, n_results: int = 3) -> list[dict]:
    """
    Semantic search over SEC 10-K/10-Q raw text chunks for a specific ticker.

    Returns the top n_results most relevant text passages.
    Each result: { section, filing_date, form_type, text, relevance_score }
    """
    ticker = ticker.upper().strip()
    collection = _get_chroma_collection()

    if collection is None:
        return [{"error": "ChromaDB unavailable — run sec_ingestion_task.py first"}]

    if collection.count() == 0:
        return [{"error": "sec_raw_chunks collection is empty — run sec_ingestion_task.py first"}]

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where={"ticker": ticker},
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "section":       meta.get("section"),
                "filing_date":   meta.get("filing_date"),
                "form_type":     meta.get("form_type"),
                "text":          doc,
                "relevance_score": round(1 - dist, 3),  # cosine distance → similarity
            })

        if not hits:
            return [{"info": f"No SEC filings found for {ticker} — may not be ingested yet"}]

        return hits

    except Exception as exc:
        logger.error(f"search_filings({ticker}, '{query}'): {exc}")
        return [{"error": str(exc)}]


# ── Tool 3: Macro Indicators ──────────────────────────────────────────────────

def get_realignment_candidates(profile_id: int) -> dict:
    """
    Portfolio Health Audit: compare a profile's weakest held positions
    against the top market challengers not currently in the portfolio.

    Weakest Links   — held tickers ranked by a weakness score:
                      low win_probability + high expected_shortfall + low sentiment
    Market Challengers — top 5 non-held tickers by today's LLM conviction score

    Returns:
        {
          "profile_id": int,
          "profile_name": str,
          "evaluation_date": str,
          "weakest_links": [ { ticker, conviction, prob_positive, es_95, sentiment, weakness_score } ],
          "challengers":   [ { ticker, conviction, prob_positive, es_95, expected_return } ]
        }
    """
    import datetime
    today = datetime.date.today().isoformat()

    if not Path(REGIMES_DB).exists():
        return {"error": f"regimes.db not found at {REGIMES_DB}"}

    try:
        conn   = sqlite3.connect(REGIMES_DB)
        cursor = conn.cursor()

        # 1. Held tickers for this profile (most recent build)
        cursor.execute("""
            SELECT ticker, profile_name FROM model_portfolio_positions
            WHERE profile_id = ?
              AND build_date = (SELECT MAX(build_date) FROM model_portfolio_positions
                                WHERE profile_id = ?)
              AND ticker != 'CASH'
        """, (profile_id, profile_id))
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return {"error": f"No portfolio found for profile_id={profile_id}"}

        held_tickers = [r[0] for r in rows]
        profile_name = rows[0][1]
        held_set     = set(held_tickers)
        ticker_list  = ",".join(f"'{t}'" for t in held_tickers)

        # 2. Today's LLM conviction for held tickers
        cursor.execute(f"""
            SELECT ticker, conviction_score, var_95, mean_expected_return
            FROM daily_llm_conviction
            WHERE date = ? AND ticker IN ({ticker_list})
        """, (today,))
        llm_data = {r[0]: {"conviction": r[1], "var_95": r[2], "mean_return": r[3]}
                    for r in cursor.fetchall()}

        # 3. Today's MC risk for held tickers
        cursor.execute(f"""
            SELECT ticker, AVG(prob_positive), AVG(es_95)
            FROM daily_monte_carlo
            WHERE date = ? AND horizon_days = 20 AND ticker IN ({ticker_list})
            GROUP BY ticker
        """, (today,))
        mc_data = {r[0]: {"prob_positive": r[1], "es_95": r[2]} for r in cursor.fetchall()}

        # 4. Latest sentiment for held tickers (from financial_data.db if available)
        sentiment_data: dict = {}
        if Path(FINANCIAL_DB_PATH).exists():
            try:
                fconn  = sqlite3.connect(FINANCIAL_DB_PATH)
                fcursor = fconn.cursor()
                fcursor.execute(f"""
                    SELECT c.ticker, AVG(s.sentiment_score)
                    FROM fact_sentiment s
                    JOIN dim_company c ON s.company_id = c.company_id
                    WHERE c.ticker IN ({ticker_list})
                    GROUP BY c.ticker
                """)
                sentiment_data = {r[0]: r[1] for r in fcursor.fetchall()}
                fconn.close()
            except Exception:
                pass

        # 5. Build Weakest Links — composite weakness score (higher = weaker)
        weakest_links = []
        for t in held_tickers:
            conviction   = (llm_data.get(t, {}).get("conviction") or 0.0)
            prob_pos     = (mc_data.get(t, {}).get("prob_positive") or 0.5)
            es           = (mc_data.get(t, {}).get("es_95") or 0.0)
            sentiment    = (sentiment_data.get(t) or 0.0)
            # weakness: low conviction + low win probability + bad ES + negative sentiment
            weakness = (-conviction) + (1 - prob_pos) + abs(min(es, 0)) + (-sentiment)
            weakest_links.append({
                "ticker":         t,
                "conviction":     round(conviction, 3),
                "prob_positive":  round(prob_pos, 3),
                "es_95":          round(es, 4) if es else None,
                "sentiment":      round(sentiment, 3) if sentiment else None,
                "weakness_score": round(weakness, 3),
            })
        weakest_links.sort(key=lambda x: x["weakness_score"], reverse=True)

        # 6. Market Challengers — top 5 non-held by conviction
        cursor.execute("""
            SELECT lc.ticker, lc.conviction_score, lc.mean_expected_return,
                   AVG(mc.prob_positive), AVG(mc.es_95)
            FROM daily_llm_conviction lc
            LEFT JOIN daily_monte_carlo mc
                ON lc.ticker = mc.ticker AND lc.date = mc.date AND mc.horizon_days = 20
            WHERE lc.date = ?
            GROUP BY lc.ticker
            ORDER BY lc.conviction_score DESC
        """, (today,))
        challengers = []
        for row in cursor.fetchall():
            if row[0] not in held_set:
                challengers.append({
                    "ticker":          row[0],
                    "conviction":      round(row[1], 3) if row[1] else None,
                    "expected_return": round(row[2], 4) if row[2] else None,
                    "prob_positive":   round(row[3], 3) if row[3] else None,
                    "es_95":           round(row[4], 4) if row[4] else None,
                })
            if len(challengers) >= 5:
                break

        conn.close()
        return {
            "profile_id":       profile_id,
            "profile_name":     profile_name,
            "evaluation_date":  today,
            "weakest_links":    weakest_links,
            "challengers":      challengers,
        }

    except Exception as exc:
        logger.error(f"get_realignment_candidates({profile_id}): {exc}")
        return {"error": str(exc)}


def get_trade_history(profile_id: int = None, days: int = 30) -> list[dict]:
    """
    Return the trade log for the last `days` days.
    If profile_id is None, returns entries for all profiles.

    Each entry explains WHY a trade was made:
      action, ticker, trigger_type, rationale, swap_ticker, executed
    """
    from tasks.trade_log import get_trade_history as _get
    return _get(profile_id=profile_id, days=days, db_path=REGIMES_DB)


def get_macro_indicators() -> dict:
    """
    Return the latest values for key FRED macroeconomic indicators:
      GDP, UNRATE (unemployment), CPIAUCSL (CPI), FEDFUNDS (Fed rate)

    Reads from financial_data.db (fact_economic_indicator).
    """
    indicators = {}

    if not Path(FINANCIAL_DB_PATH).exists():
        return {"error": f"financial_data.db not found at {FINANCIAL_DB_PATH}"}

    try:
        conn   = sqlite3.connect(FINANCIAL_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                d.indicator_code,
                d.indicator_name,
                d.unit,
                f.value,
                f.change_percent,
                dim_d.full_date
            FROM fact_economic_indicator f
            JOIN dim_economic_indicator d ON f.indicator_id = d.indicator_id
            JOIN dim_date dim_d           ON f.date_id      = dim_d.date_id
            WHERE d.indicator_code IN ('GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS')
              AND f.value IS NOT NULL
            ORDER BY d.indicator_code, dim_d.full_date DESC
        """)

        seen = set()
        for code, name, unit, value, chg_pct, date in cursor.fetchall():
            if code not in seen:  # take only the most recent per indicator
                indicators[code] = {
                    "name":           name,
                    "unit":           unit,
                    "value":          round(float(value), 4) if value is not None else None,
                    "change_percent": round(float(chg_pct), 4) if chg_pct is not None else None,
                    "as_of":          date,
                }
                seen.add(code)

        conn.close()

        if not indicators:
            return {"warning": "No macro data found — check if data scrapers have run"}

        return indicators

    except Exception as exc:
        logger.error(f"get_macro_indicators: {exc}")
        return {"error": str(exc)}
