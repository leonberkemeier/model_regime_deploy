import logging
import os
import sqlite3
import datetime
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(Path(project_root) / ".env")

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Set MC_USE_GARCH=false in .env to revert to plain Gaussian (for debugging)
MC_USE_GARCH = os.getenv("MC_USE_GARCH", "true").lower() == "true"

# Standard equity GARCH(1,1) parameters (Engle 2002)
# alpha + beta < 1.0 guarantees variance stationarity
_GARCH_ALPHA = float(os.getenv("GARCH_ALPHA", "0.09"))  # ARCH term: shock sensitivity
_GARCH_BETA  = float(os.getenv("GARCH_BETA",  "0.90"))  # GARCH term: variance persistence


# ── GARCH(1,1) simulation ─────────────────────────────────────────────────────

def _fit_garch_params(returns: np.ndarray) -> tuple[float, float, float]:
    """
    Fit GARCH(1,1) to a return series using the arch library.
    Returns (omega, alpha, beta) in daily return scale.

    Falls back to (None, _GARCH_ALPHA, _GARCH_BETA) when fitting fails
    so the caller can use the HMM-derived omega with calibrated dynamics.
    """
    try:
        from arch import arch_model
        # Scale to % for numerical stability, then rescale params back
        r_pct   = returns * 100.0
        model   = arch_model(r_pct, vol="Garch", p=1, q=1, rescale=False)
        result  = model.fit(disp="off", show_warning=False)
        omega   = float(result.params["omega"]) / 10_000   # back to daily return scale
        alpha   = float(result.params["alpha[1]"])
        beta    = float(result.params["beta[1]"])
        # Reject degenerate solutions
        if alpha > 0 and beta > 0 and (alpha + beta) < 0.9995:
            return omega, alpha, beta
    except Exception:
        pass
    return None, _GARCH_ALPHA, _GARCH_BETA


def _batch_fetch_returns(tickers: list[str], period: str = "1y") -> dict[str, np.ndarray]:
    """
    Batch-download 1 year of daily close prices via yfinance and compute
    log-returns for each ticker. Returns {ticker: returns_array}.
    Tickers with insufficient data are silently omitted.
    """
    if not tickers:
        return {}
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=period, auto_adjust=True,
                          progress=False, threads=True)
        # yfinance returns MultiIndex columns for multiple tickers
        if hasattr(raw.columns, "levels"):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})

        result = {}
        for ticker in tickers:
            if ticker not in close.columns:
                continue
            prices = close[ticker].dropna()
            if len(prices) < 60:
                continue
            returns = np.log(prices / prices.shift(1)).dropna().values
            result[ticker] = returns
        return result
    except Exception as exc:
        logger.warning(f"yfinance batch fetch failed: {exc}")
        return {}


def _simulate_garch_paths(
    mean: float,
    vol: float,
    horizon_days: int,
    n_simulations: int,
    fitted_omega: float = None,
    fitted_alpha: float = None,
    fitted_beta:  float = None,
) -> np.ndarray:
    """
    GARCH(1,1) Monte Carlo — simulates n_simulations forward paths of
    `horizon_days` each with time-varying, mean-reverting volatility.

    Model:
        r_t     = mean + sqrt(h_t) * z_t         z_t ~ N(0,1)
        h_{t+1} = omega + alpha*(r_t - mean)^2 + beta*h_t

    omega is derived from the HMM state volatility so that the long-run
    unconditional variance equals the HMM regime variance:
        E[h] = omega / (1 - alpha - beta)  =>  omega = vol^2 * (1 - alpha - beta)

    This produces volatility clustering: a large shock raises h_t, which
    raises volatility for subsequent days before mean-reverting — exactly
    what equity markets exhibit but plain Gaussian ignores.

    Returns:
        Array of shape (n_simulations,) — compounded horizon returns.
    """
    # Use fitted params if provided; fall back to literature defaults
    alpha = fitted_alpha if fitted_alpha is not None else _GARCH_ALPHA
    beta  = fitted_beta  if fitted_beta  is not None else _GARCH_BETA
    if fitted_omega is not None:
        omega = fitted_omega
    else:
        omega = (vol ** 2) * (1.0 - alpha - beta)  # anchor to HMM variance
    omega = max(omega, 1e-8)                        # numerical floor

    # Draw all innovations at once — shape (n_simulations, horizon_days)
    innovations = np.random.standard_normal((n_simulations, horizon_days))

    daily_returns = np.empty((n_simulations, horizon_days))
    h = np.full(n_simulations, vol ** 2, dtype=np.float64)  # initial conditional variance

    for t in range(horizon_days):
        z   = innovations[:, t]              # (n_simulations,)
        r_t = mean + np.sqrt(h) * z          # daily returns this step
        daily_returns[:, t] = r_t
        # GARCH(1,1) variance update
        h = omega + alpha * (r_t - mean) ** 2 + beta * h
        h = np.maximum(h, 1e-8)              # keep variance positive

    # Compound over the horizon: (1+r1)(1+r2)...(1+rT) - 1
    return np.prod(1.0 + daily_returns, axis=1) - 1.0

def init_mc_table(conn):
    """Initialize or migrate SQLite table for storing daily Monte Carlo results."""
    cursor = conn.cursor()

    # Ensure table exists (latest schema)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_monte_carlo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            lookback_days INTEGER,
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
            simulation_method TEXT,
            UNIQUE(date, ticker, lookback_days, horizon_days)
        )
    ''')

    # Add simulation_method column to existing DBs (safe migration)
    cursor.execute("PRAGMA table_info(daily_monte_carlo)")
    cols = {row[1] for row in cursor.fetchall()}
    if "simulation_method" not in cols:
        cursor.execute("ALTER TABLE daily_monte_carlo ADD COLUMN simulation_method TEXT")

    # Backward-compatible migration from old schema
    cursor.execute("PRAGMA table_info(daily_monte_carlo)")
    existing_columns = [row[1] for row in cursor.fetchall()]

    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='daily_monte_carlo'")
    create_sql_row = cursor.fetchone()
    create_sql = (create_sql_row[0] if create_sql_row and create_sql_row[0] else "").lower().replace(" ", "")

    has_old_unique_key = "unique(date,ticker,horizon_days)" in create_sql
    needs_migration = ("lookback_days" not in existing_columns) or has_old_unique_key

    if needs_migration:
        logger.info("Migrating daily_monte_carlo schema to include lookback_days in uniqueness key...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_monte_carlo_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                ticker TEXT,
                lookback_days INTEGER,
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
                UNIQUE(date, ticker, lookback_days, horizon_days)
            )
        ''')

        if "lookback_days" in existing_columns:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_monte_carlo_new
                (date, ticker, lookback_days, horizon_days, simulations, mean_expected_return, median_expected_return,
                 var_95, var_99, es_95, es_99, prob_loss, prob_positive)
                SELECT date, ticker, lookback_days, horizon_days, simulations, mean_expected_return, median_expected_return,
                       var_95, var_99, es_95, es_99, prob_loss, prob_positive
                FROM daily_monte_carlo
            ''')
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_monte_carlo_new
                (date, ticker, lookback_days, horizon_days, simulations, mean_expected_return, median_expected_return,
                 var_95, var_99, es_95, es_99, prob_loss, prob_positive)
                SELECT date, ticker, 60, horizon_days, simulations, mean_expected_return, median_expected_return,
                       var_95, var_99, es_95, es_99, prob_loss, prob_positive
                FROM daily_monte_carlo
            ''')

        cursor.execute("DROP TABLE daily_monte_carlo")
        cursor.execute("ALTER TABLE daily_monte_carlo_new RENAME TO daily_monte_carlo")

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
        SELECT ticker, lookback_days, mean_return, volatility, current_state
        FROM daily_regimes 
        WHERE date = ?
    ''', (today_str,))
    
    daily_regimes = cursor.fetchall()
    
    if not daily_regimes:
        logger.warning(f"No HMM regime data found for {today_str}. Did you run sqlite_regime_task first?")
        return

    method = "GARCH(1,1)" if MC_USE_GARCH else "Gaussian"
    logger.info(f"Found {len(daily_regimes)} assets to simulate... [method: {method}]")

    # Pre-fit GARCH parameters per ticker using 1 year of actual returns.
    # This calibrates the volatility dynamics to each stock's own history
    # rather than relying on generic equity defaults.
    garch_params: dict[str, tuple] = {}  # ticker -> (omega, alpha, beta)
    if MC_USE_GARCH:
        all_tickers = list({r[0] for r in daily_regimes})
        logger.info(f"Batch downloading 1y returns for GARCH fitting ({len(all_tickers)} tickers)...")
        returns_map = _batch_fetch_returns(all_tickers)
        fitted_count = 0
        for tkr, ret in returns_map.items():
            omega, alpha, beta = _fit_garch_params(ret)
            garch_params[tkr] = (omega, alpha, beta)
            if omega is not None:
                fitted_count += 1
        logger.info(
            f"GARCH fitted for {fitted_count}/{len(all_tickers)} tickers; "
            f"{len(all_tickers) - fitted_count} use literature defaults."
        )

    np.random.seed(42)  # For reproducibility

    for ticker, lookback_days, mean, vol, state in daily_regimes:
        try:
            # 2. Simulate forward paths
            if MC_USE_GARCH:
                fitted = garch_params.get(ticker, (None, None, None))
                compounded = _simulate_garch_paths(
                    mean, vol, horizon_days, n_simulations,
                    fitted_omega=fitted[0], fitted_alpha=fitted[1], fitted_beta=fitted[2]
                )
            else:
                # Fallback: plain Gaussian (constant volatility)
                daily_returns = np.random.normal(
                    loc=mean, scale=vol, size=(n_simulations, horizon_days)
                )
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
                (date, ticker, lookback_days, horizon_days, simulations, mean_expected_return, median_expected_return,
                 var_95, var_99, es_95, es_99, prob_loss, prob_positive, simulation_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (today_str, ticker, int(lookback_days), horizon_days, n_simulations,
                  mean_ret, median_ret, var_95, var_99, es_95, es_99, prob_loss, prob_pos, method))

            logger.info(
                f"✅ {ticker} [{lookback_days}d | {state}] "
                f"VaR-95: {var_95:.2%} | ES-95: {es_95:.2%} | Prob Loss: {prob_loss:.1%}"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed simulating {ticker}: {e}")
            
    conn.commit()
    conn.close()
    logger.info("=== Monte Carlo Task Completed ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_sqlite_monte_carlo_task()
