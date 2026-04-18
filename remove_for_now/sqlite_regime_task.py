import logging
import sqlite3
import datetime
import sys
from pathlib import Path

# Add project root to sys.path so we can import core_math
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yfinance as yf
import pandas as pd

from core_math.markov_chain_detector import MarkovChainRegimeDetector

logger = logging.getLogger(__name__)

def init_db(db_path="regimes.db"):
    """Initialize SQLite database for storing daily regime data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create table if it doesn't exist. UNIQUE constraint ensures we don't get duplicates per day/ticker.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_regimes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            lookback_days INTEGER,
            current_state TEXT,
            confidence REAL,
            mean_return REAL,
            volatility REAL,
            UNIQUE(date, ticker, lookback_days)
        )
    ''')
    conn.commit()
    return conn

def run_sqlite_regime_task(tickers=None, db_path="regimes.db"):
    """Fetch data, build HMM for each stock, and save results to SQLite."""
    if tickers is None:
        tickers = ["SPY", "TSLA", "QQQ", "AAPL", "MSFT"]
        
    logger.info("=== Starting Daily SQLite Regime Builder Task ===")
    conn = init_db(db_path)
    cursor = conn.cursor()
    
    today_str = datetime.date.today().isoformat()
    
    # Subdue internal logs for the hmmlearn library so the console loop isn't too messy
    hmm_logger = logging.getLogger('core_math.markov_chain_detector')
    hmm_logger.setLevel(logging.WARNING)
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        try:
            # 1. Fetch data from yfinance
            df = yf.download(ticker, period="5y", progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
            
            # yfinance occasionally returns a MultiIndex if multiple tickers are passed or due to version differences
            if isinstance(df.columns, pd.MultiIndex):
                prices = df['Close'][ticker].squeeze()
            else:
                prices = df['Close'].squeeze()
                
            prices.index.name = 'date'
                
            # 2. Build and Train Model
            model_out_path = Path(f"models/hmm_regime_{ticker}_60d.pkl")
            detector = MarkovChainRegimeDetector(
                n_states=3,
                n_iter=100,
                lookback_days=60,
                model_path=model_out_path
            )
            
            detector.fit(prices)
            state_info = detector.detect_current_regime(prices)
            
            current_state = state_info.current_regime
            confidence = state_info.regime_probability
            
            # 3. Get Regime Statistics
            stats = detector.get_regime_statistics(prices, current_state)
            mean_return = stats.get("mean_return", 0.0)
            volatility = stats.get("std_return", 0.0) # Extract standard deviation for volatility
            
            # 4. Save to SQLite (INSERT OR REPLACE to handle re-running on the same day safely)
            cursor.execute('''
                INSERT OR REPLACE INTO daily_regimes 
                (date, ticker, lookback_days, current_state, confidence, mean_return, volatility)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (today_str, ticker, 60, current_state, confidence, mean_return, volatility))
            conn.commit()
            
            logger.info(f"✅ Saved {ticker} -> State: {current_state} | Conf: {confidence:.1%} | Ret: {mean_return:.4f} | Vol: {volatility:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed processing {ticker}: {e}")
            
    conn.close()
    logger.info("=== Daily SQLite Regime Builder Task Completed ===")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # Load your list of stocks to train from the local financial_data.db
    my_stocks = ["SPY", "TSLA", "QQQ", "AAPL", "MSFT"] # Fallback list
    fin_db_path = Path(project_root) / "financial_data.db"
    
    if fin_db_path.exists():
        try:
            fin_conn = sqlite3.connect(fin_db_path)
            fin_cursor = fin_conn.cursor()
            fin_cursor.execute("SELECT DISTINCT ticker FROM dim_company")
            db_stocks = [row[0] for row in fin_cursor.fetchall()]
            
            # Filter out index tickers (like ^GSPC) if preferred, or keep them all. Let's keep valid tickers.
            # You can filter by specific conditions if needed.
            if db_stocks:
                my_stocks = db_stocks
                logger.info(f"Loaded {len(my_stocks)} stocks from financial_data.db")
                
            fin_conn.close()
        except Exception as e:
            logger.error(f"Failed to fetch tickers from financial_data.db: {e}")
    else:
        logger.warning(f"financial_data.db not found at {fin_db_path}. Using fallback list.")
    
    run_sqlite_regime_task(tickers=my_stocks)

