import logging
from connectors.webserver_client import WebserverClient
from core_math.enhanced_monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)

def run_monte_carlo_task(api_client: WebserverClient):
    """Fetches data and current regime, simulates future paths, posts back risk metrics."""
    logger.info("=== Starting Phase 2: Enhanced Monte Carlo ===")
    
    try:
        # 1. Fetch dependencies from webserver
        # Using yfinance for direct stock fetches, but you can switch to source="webserver"
        df = api_client.get_prices(
            source="yfinance", 
            yfinance_tickers="AAPL MSFT GOOG AMZN", 
            period="5y"
        )
        regime_state = api_client.get_markov_state() # Need current regime state
        
        # 2. Run simulation
        simulator = MonteCarloSimulator(n_simulations=10000)
        
        results_payload = []
        for ticker in df.columns:
            logger.info(f"Simulating expected paths for {ticker}...")
            mc_metrics = simulator.simulate_asset(
                # Ensure df[ticker] matches series structure
                df[ticker], 
                ticker,
                # Assuming regime_state deserializes to matching object type
                regime_state
            )
            
            results_payload.append({
                "ticker": ticker,
                "execution_date": regime_state.get('execution_date'),
                "mean_return": mc_metrics.mean_return,
                "var_95": mc_metrics.var_95,
                "var_99": mc_metrics.var_99,
                "es_95": mc_metrics.es_95,
                "es_99": mc_metrics.es_99,
                "prob_loss": mc_metrics.prob_loss,
                "regime_suitability": mc_metrics.regime_suitability
            })
            
        # 3. Post back metrics
        api_client.post_monte_carlo_result(results_payload)
        logger.info(f"✅ Success! Monte Carlo results saved to DB for {len(results_payload)} assets")
        
    except Exception as e:
        logger.error(f"❌ Monte Carlo Task Failed: {e}", exc_info=True)
