import logging
import sqlite3
import datetime
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

def init_mc_table(conn):
    """Initialize SQLite table for storing daily Monte Carlo results."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_monte_carlo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            horizon_days INTEGER,
            simulations INTEGER,
            mean_expected_return REAL,
            median_expected_return REAL,
            var_95 REAL,
            var_99 REAL,
            es_95 REAL,
            es_99 REAL,
            prob_loss REAL,
            prob_positive REAL,
            UNIQUE(date, ticker, horizon_days)
        )
    ''')
    conn.commit()

def run_sqlite_monte_carlo_task(db_path="regimes.db", horizon_days=20, n_simulations=10000):
    """
    Fetch the HMM regime properties (mean return, volatility) from regimes.db 
    and run a Monte Carlo simulation. Save the VaR & Expected Shortfall risk metrics.
    """
    logger.info("=== Starting Daily SQLite Monte Carlo Risk Task ===")
    
    today_str = datetime.date.today().isoformat()
    
    conn = sqlite3.connect(db_path)
    init_mc_table(conn)
    cursor = conn.cursor()
    
    # 1. Fetch latest HMM properties for today
    cursor.execute('''
        SELECT ticker, mean_return, volatility, current_state
        FROM daily_regimes 
        WHERE date = ?
    ''', (today_str,))
    
    daily_regimes = cursor.fetchall()
    
    if not daily_regimes:
        logger.warning(f"No HMM regime data found for {today_str}. Did you run sqlite_regime_task first?")
        return

    logger.info(f"Found {len(daily_regimes)} assets to simulate...")
    
    np.random.seed(42) # For reproducibility
    
    for ticker, mean, vol, state in daily_regimes:
        try:
            # 2. Simulate Forward Paths directly from the HMM's exact state parameters
            daily_returns = np.random.normal(
                loc=mean,
                scale=vol,
                size=(n_simulations, horizon_days)
            )
            
            # Compound returns over horizon
            compounded = np.prod(1 + daily_returns, axis=1) - 1
            
            # 3. Calculate Risk Metrics
            mean_ret = float(np.mean(compounded))
            median_ret = float(np.median(compounded))
            
            var_95 = float(np.percentile(compounded, 5))
            var_99 = float(np.percentile(compounded, 1))
            
            es_95 = float(compounded[compounded <= var_95].mean())
            es_99 = float(compounded[compounded <= var_99].mean())
            
            prob_loss = float(np.mean(compounded < 0))
            prob_pos = float(np.mean(compounded > 0))
            
            # 4. Save to Database
            cursor.execute('''
                INSERT OR REPLACE INTO daily_monte_carlo 
                (date, ticker, horizon_days, simulations, mean_expected_return, median_expected_return, 
                 var_95, var_99, es_95, es_99, prob_loss, prob_positive)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (today_str, ticker, horizon_days, n_simulations, 
                  mean_ret, median_ret, var_95, var_99, es_95, es_99, prob_loss, prob_pos))
            
            logger.info(f"✅ {ticker} (State: {state}) -> horizon: {horizon_days}d | VaR-95: {var_95:.2%} | Prob Loss: {prob_loss:.1%}")
            
        except Exception as e:
            logger.error(f"❌ Failed simulating {ticker}: {e}")
            
    conn.commit()
    conn.close()
    logger.info("=== Monte Carlo Task Completed ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_sqlite_monte_carlo_task()
