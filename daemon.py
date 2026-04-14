import logging
import schedule
import time
import os
import sys
from datetime import datetime

# Adjust module path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import WEBSERVER_URL, OLLAMA_URL, WORK_HOUR, WORK_MINUTE_MARKOV, WORK_MINUTE_MC, WORK_MINUTE_LLM, WORK_MINUTE_TRIGGER
from connectors.webserver_client import WebserverClient
from connectors.ollama_client import OllamaClient

from tasks.markov_task import run_markov_task
from tasks.monte_carlo_task import run_monte_carlo_task
from tasks.llm_task import run_llm_task
from tasks.greenfield_portfolio_task import run_greenfield_models_task

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

api_client = WebserverClient(WEBSERVER_URL)
ollama_client = OllamaClient(OLLAMA_URL)

def execute_daily_pipeline():
    """Manually invoke the entire sequence back-to-back if desired!"""
    logger.info("=== SPUN UP: ML Pipeline Starting ===")
    
    run_markov_task(api_client)
    run_monte_carlo_task(api_client)
    run_llm_task(api_client, ollama_client)
    run_greenfield_models_task(api_client)
    
    logger.info("=== ML Pipeline Finished -> Going back to sleep. ===")

def main():
    logger.info(f"AI Server Daemon Starting. Connected to Webserver @ {WEBSERVER_URL}")
    
    # We can schedule them as discrete tasks, or as one giant pipeline at 08:05:
    daily_schedule = f"{WORK_HOUR:02d}:{WORK_MINUTE_MARKOV:02d}"
    schedule.every().day.at(daily_schedule).do(execute_daily_pipeline)
    
    logger.info(f"Scheduled daily pipeline start at {daily_schedule}.")

    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Daemon stopping...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Scheduler Exception: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
