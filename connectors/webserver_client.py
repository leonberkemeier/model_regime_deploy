import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class WebserverClient:
    """Connects to the central Webserver via Tailscale/REST API"""
    def __init__(self, base_url):
        self.base_url = base_url

    def get_prices(self, source="webserver", yfinance_tickers="^GSPC", period="1y") -> pd.DataFrame:
        if source == "yfinance":
            logger.info(f"Fetching latest prices directly from yfinance for {yfinance_tickers}...")
            try:
                import yfinance as yf
            except ImportError:
                raise ImportError("yfinance not installed. Please install with: pip install yfinance")
            
            df = yf.download(yfinance_tickers, period=period)
            # Ensure index name is date
            df.index.name = 'date'
            
            # If multiple tickers, df.Close is a DataFrame. If single ticker, df['Close'] might be a Series in older yfinance,
            # but in newer yfinance, df['Close'] is a DataFrame if multiple tickers are passed.
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten or select just the Close prices
                return df['Close']
            else:
                return df[['Close']].rename(columns={'Close': yfinance_tickers})
            
        logger.info(f"Fetching latest prices from {self.base_url}/api/data/latest...")
        response = requests.get(f"{self.base_url}/api/data/latest")
        response.raise_for_status()
        data = response.json()
        
        # Reconstruct DataFrame from JSON format
        # If it's a list directly, use it. If dict, extract 'prices'.
        prices_data = data.get('prices', {}) if isinstance(data, dict) else data
        df = pd.DataFrame(prices_data)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        return df

    def get_markov_state(self):
        response = requests.get(f"{self.base_url}/api/analysis/markov")
        response.raise_for_status()
        return response.json()
    
    def get_monte_carlo_results(self):
        response = requests.get(f"{self.base_url}/api/analysis/monte_carlo/all")
        response.raise_for_status()
        return response.json()

    def post_markov_state(self, payload: dict):
        response = requests.post(f"{self.base_url}/api/analysis/markov", json=payload)
        response.raise_for_status()
        return response.json()
    
    def post_monte_carlo_result(self, payload: list):
        response = requests.post(f"{self.base_url}/api/analysis/monte_carlo", json=payload)
        response.raise_for_status()
        return response.json()

    def post_llm_scores(self, payload: dict):
        response = requests.post(f"{self.base_url}/api/analysis/llm_scores", json=payload)
        response.raise_for_status()
        return response.json()

    def get_llm_scores(self):
        response = requests.get(f"{self.base_url}/api/analysis/llm_scores/latest")
        response.raise_for_status()
        return response.json()

    def post_greenfield_models(self, payload: list):
        """Pushes the 5 generated Greenfield Model Portfolios to the Webserver"""
        response = requests.post(f"{self.base_url}/api/portfolio/greenfield_models", json=payload)
        response.raise_for_status()
        return response.json()
