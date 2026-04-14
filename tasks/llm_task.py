import logging
from connectors.webserver_client import WebserverClient
from connectors.ollama_client import OllamaClient
from config.settings import OLLAMA_MODEL

logger = logging.getLogger(__name__)

def run_llm_task(api_client: WebserverClient, ollama_client: OllamaClient):
    """Fetch MC data, use local LLM to score suitability, post back to DB."""
    logger.info("=== Starting Phase 3: LLM Asset Scoring ===")
    
    try:
        # 1. Retrieve prior data
        regime = api_client.get_markov_state()
        mc_results_list = api_client.get_monte_carlo_results()
        
        # 2. Iterate each asset through local Ollama LLM
        llm_scores = {}
        
        for asset in mc_results_list:
            ticker = asset.get('ticker')
            prompt = f"""
            Analyze this asset for suitability in the current {regime.get('current_regime')} regime.
            Metrics:
            - Mean Return: {asset.get('mean_return', 0):.2%}
            - Value at Risk (95%): {asset.get('var_95', 0):.2%}
            - Expected Shortfall (95%): {asset.get('es_95', 0):.2%}
            - Probability of Loss: {asset.get('prob_loss', 0):.0%}
            
            Based on these metrics and the regime, score between 0.0 (Worst) and 1.0 (Best).
            You MUST return a JSON object with only a 'score' and 'reasoning'. e.g. {{"score": 0.8, "reasoning": "Strong expected return despite VaR..."}}
            """
            
            logger.info(f"Scoring {ticker} via Ollama ({OLLAMA_MODEL})...")
            # Wait for inference
            score = ollama_client.generate_score(prompt=prompt, model=OLLAMA_MODEL)
            
            llm_scores[ticker] = {
                "score": score,
                "reasoning": f"Generated score based on ML Server local {OLLAMA_MODEL}"
            }
        
        # 3. Post LLM metrics back
        payload = {
            "execution_date": regime.get('execution_date'),
            "scores": llm_scores
        }
        api_client.post_llm_scores(payload)
        logger.info(f"✅ Success! Ollama processed {len(mc_results_list)} assets.")
        
    except Exception as e:
        logger.error(f"❌ LLM Task Failed: {e}", exc_info=True)
