import logging
from connectors.webserver_client import WebserverClient
from core_math.markov_chain_detector import MarkovChainRegimeDetector
import os
import datetime

logger = logging.getLogger(__name__)

def run_markov_task(api_client: WebserverClient):
    """Fetches data from webserver, calculates current market regime, posts back state."""
    logger.info("=== Starting Phase 1: Markov Detection ===")
    
    try:
        # 1. Fetch data
        df = api_client.get_prices()
        
        # 2. Run core math (Requires clean dataframe)
        detector = MarkovChainRegimeDetector(n_states=5)
        
        # NOTE: df structure needs to match what your markov chain expects!
        detector.fit(df)
        regime_state = detector.detect_current_regime(df)
        
        # 3. Create POST payload
        # Find integer ID for the current regime string
        regime_id_map = {v: k for k, v in detector.regime_order.items()}
        current_regime_id = int(regime_id_map.get(regime_state.current_regime, 0))

        payload = {
            "execution_date": datetime.date.today().isoformat(),
            "current_regime": current_regime_id,
            "regime_probability": float(regime_state.regime_probability),
            # transition_matrix will need `.tolist()` for JSON serialization
            "transition_matrix": regime_state.transition_matrix.tolist() if hasattr(regime_state.transition_matrix, "tolist") else regime_state.transition_matrix,
            "probability_next_regime": regime_state.probability_next_regime
        }
        
        api_client.post_markov_state(payload)
        logger.info(f"✅ Success! Markov State saved: {regime_state.current_regime} ({regime_state.regime_probability:.2%})")

    except Exception as e:
        logger.error(f"❌ Markov Task Failed: {e}", exc_info=True)
