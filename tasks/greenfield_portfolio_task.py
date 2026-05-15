import logging
import sqlite3
import datetime
from pathlib import Path
from connectors.webserver_client import WebserverClient
from typing import List, Dict

logger = logging.getLogger(__name__)

REGIMES_DB = str(Path(__file__).parent.parent / "regimes.db")

# Basic bounds mapped to the 5 standard Risk Profiles (1: Conservative to 5: Aggressive)
RISK_PROFILES = {
    1: {"name": "Conservative", "max_var_95": -0.02, "min_assets": 15},
    2: {"name": "Moderately Conservative", "max_var_95": -0.04, "min_assets": 12},
    3: {"name": "Moderate", "max_var_95": -0.06, "min_assets": 10},
    4: {"name": "Moderately Aggressive", "max_var_95": -0.10, "min_assets": 10},
    5: {"name": "Aggressive", "max_var_95": -0.15, "min_assets": 10}
}

def run_greenfield_models_task(api_client: WebserverClient):
    """
    Constructs the 5 daily Model Portfolios purely based on today's math and AI scores.
    """
    logger.info("=== Starting Phase 4: Constructing Greenfield Model Portfolios ===")
    
    try:
        # 1. Fetch Today's Universe Data
        mc_results_list = api_client.get_monte_carlo_results()
        llm_scores_data = api_client.get_llm_scores()
        
        execution_date = llm_scores_data.get("execution_date")
        llm_scores = llm_scores_data.get("scores", {})
        
        # Merge MC math and LLM qualitative scores into a single view
        universe = []
        for asset in mc_results_list:
            ticker = asset.get("ticker")
            universe.append({
                "ticker": ticker,
                "var_95": asset.get("var_95", -0.99),       # Negative number (e.g. -0.05 is a 5% loss)
                "expected_return": asset.get("mean_return", 0),
                "llm_score": llm_scores.get(ticker, {}).get("score", 0.0)
            })

        greenfield_models = []

        # 2. Construct the 5 Portfolios
        for profile_id, constraints in RISK_PROFILES.items():
            logger.info(f"Building Model for Risk Profile {profile_id}: {constraints['name']}")
            
            # Filter assets strictly by VaR constraint (Math acts as the "Brakes")
            # e.g., Conservative (Profile 1) drops anything where 95% VaR is worse than -2%
            eligible_assets = [
                a for a in universe 
                if a["var_95"] >= constraints["max_var_95"]  # e.g., -0.01 >= -0.02 is True
            ]
            
            # Rank eligible assets strictly by LLM Fundamental Score (AI acts as the "Gas Pedal")
            ranked_assets = sorted(eligible_assets, key=lambda x: x["llm_score"], reverse=True)
            
            # Pick the top N assets
            n_assets = constraints["min_assets"]
            selected = ranked_assets[:n_assets]
            
            # Simple Equal Weighting for the Greenfield Model definition
            # (A true optimizer could be added here later to maximize Sharpe Ratio)
            if not selected:
                logger.warning(f"No assets qualified for Profile {profile_id} bounds. Falling back to cash.")
                positions = [{"ticker": "CASH", "weight": 1.0}]
            else:
                weight = 1.0 / len(selected)
                positions = [{"ticker": asset["ticker"], "weight": round(weight, 4)} for asset in selected]
                
            greenfield_models.append({
                "profile_id": profile_id,
                "profile_name": constraints["name"],
                "execution_date": execution_date,
                "positions": positions
            })
            
            logger.info(f" -> Selected {len(positions)} assets for Profile {profile_id}")

        # 3. Push the 5 Models to the Webserver
        payload = {
            "execution_date": execution_date,
            "models": greenfield_models
        }
        
        api_client.post_greenfield_models(payload)
        logger.info(f"✅ Success! 5 Greenfield Model Portfolios pushed to Webserver.")

        # Save snapshot for realignment tracking
        try:
            from tasks.realignment_task import (
                init_snapshot_tables, save_conviction_snapshot, save_portfolio_positions
            )
            from tasks.trade_log import init_trade_log_table, log_rebuild
            today_str = execution_date or datetime.date.today().isoformat()
            snap_conn = sqlite3.connect(REGIMES_DB)
            init_snapshot_tables(snap_conn)
            init_trade_log_table(snap_conn)
            for model in greenfield_models:
                pid       = model["profile_id"]
                pname     = model["profile_name"]
                positions = model["positions"]
                save_portfolio_positions(snap_conn, pid, pname, positions, today_str)
                save_conviction_snapshot(snap_conn, pid, positions, today_str)
                # Log the greenfield construction as a REBUILD event
                tickers = [p["ticker"] for p in positions if p["ticker"] != "CASH"]
                log_rebuild(
                    snap_conn,
                    log_date=today_str,
                    profile_id=pid,
                    profile_name=pname,
                    n_violations=len(tickers),
                    violation_summary=(
                        f"Greenfield construction: {len(tickers)} positions selected "
                        f"({', '.join(tickers[:5])}{', ...' if len(tickers) > 5 else ''})"
                    ),
                )
            snap_conn.close()
            logger.info("✅ Conviction snapshot and trade log saved.")
        except Exception as snap_exc:
            logger.warning(f"Snapshot/trade-log save failed (non-critical): {snap_exc}")
        
    except Exception as e:
        logger.error(f"❌ Greenfield Models Task Failed: {e}", exc_info=True)
