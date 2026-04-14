"""
Enhanced Monte Carlo Simulator with Regime Awareness

Simulates possible future outcomes based on current market regime to estimate:
- Mean and Median returns
- Value at Risk (VaR) - 95% and 99% confidence levels
- Expected Shortfall (CVaR) - tail risk metric
- Confidence intervals
- Regime suitability scoring

Used to inform portfolio construction and risk assessment.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
from datetime import datetime

from .markov_chain_detector import MarkovRegimeState, MarkovChainRegimeDetector


@dataclass
class MonteCarloMetrics:
    """Results from Monte Carlo simulation for a single asset."""
    ticker: str
    regime: str
    n_simulations: int
    
    # Central tendency metrics
    mean_return: float
    median_return: float
    std_return: float
    
    # Risk metrics (percentiles of simulated returns)
    var_95: float           # 5th percentile (95% don't lose this much)
    var_99: float           # 1st percentile (99% don't lose this much)
    
    # Expected Shortfall (mean of worst X%)
    es_95: float            # Mean of worst 5% outcomes
    es_99: float            # Mean of worst 1% outcomes
    
    # Probability metrics
    prob_loss: float        # P(return < 0)
    prob_positive: float    # P(return > 0)
    
    # Confidence intervals
    ci_95_lower: float      # 2.5th percentile
    ci_95_upper: float      # 97.5th percentile
    ci_99_lower: float      # 0.5th percentile
    ci_99_upper: float      # 99.5th percentile
    
    # Distribution shape
    skewness: float
    kurtosis: float
    
    # Regime-specific metrics
    regime_suitability: Dict[str, float] = field(default_factory=dict)  # Regime → score
    frequency_in_regime: int = 0  # Observations from current regime
    
    # Raw data
    simulated_returns: np.ndarray = field(default_factory=lambda: np.array([]))


class MonteCarloSimulator:
    """
    Monte Carlo simulator for assets with regime awareness.
    
    Process:
    1. Current regime identified by Markov chain
    2. Filter historical returns to only current regime type
    3. Simulate forward returns using regime-specific distribution
    4. Calculate risk metrics from simulated paths
    5. Output to portfolio construction and risk assessment
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        horizon_days: int = 20,  # 1 trading month
        use_regime_filtering: bool = True,
        random_state: int = 42,
        markov_detector: Optional[MarkovChainRegimeDetector] = None,
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of Monte Carlo paths to simulate
            horizon_days: Forward-looking period (days)
            use_regime_filtering: Filter returns by current regime (recommended: True)
            random_state: Random seed for reproducibility
            markov_detector: Markov chain detector for regime filtering
        """
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.use_regime_filtering = use_regime_filtering
        self.random_state = random_state
        self.markov_detector = markov_detector
        
        np.random.seed(random_state)
        self.logger = logger.bind(module="monte_carlo_simulator")
        
        self.logger.info(
            f"MonteCarloSimulator initialized: "
            f"n_sims={n_simulations}, horizon={horizon_days}d, "
            f"regime_filtering={use_regime_filtering}"
        )
    
    def simulate_asset(
        self,
        prices: pd.Series,
        ticker: str,
        current_regime: MarkovRegimeState,
    ) -> MonteCarloMetrics:
        """
        Run Monte Carlo simulation for a single asset.
        
        Args:
            prices: Historical price series
            ticker: Asset ticker
            current_regime: Current market regime from Markov chain
            
        Returns:
            MonteCarloMetrics with simulation results
        """
        # Filter returns by regime if detector available
        if self.use_regime_filtering and self.markov_detector is not None:
            try:
                returns = self.markov_detector.filter_returns_by_regime(
                    prices, 
                    current_regime.current_regime
                )
                freq_in_regime = len(returns)
                
                self.logger.debug(
                    f"{ticker}: Using {freq_in_regime} regime-specific returns "
                    f"from '{current_regime.current_regime}' regime"
                )
            except Exception as e:
                self.logger.warning(f"Failed to filter {ticker} by regime: {e}. Using all returns.")
                returns = prices.pct_change().dropna()
                freq_in_regime = len(returns)
        else:
            returns = prices.pct_change().dropna()
            freq_in_regime = len(returns)
        
        if len(returns) < 10:
            self.logger.warning(
                f"{ticker}: Only {len(returns)} return observations. Using defaults."
            )
            return self._create_default_metrics(ticker, current_regime.current_regime)
        
        # Simulate forward returns
        simulated_returns = self._simulate_paths(returns)
        
        # Compute metrics from simulated returns
        metrics = self._compute_metrics(
            ticker=ticker,
            regime=current_regime.current_regime,
            simulated_returns=simulated_returns,
            freq_in_regime=freq_in_regime,
        )
        
        # Compute regime suitability scores
        metrics.regime_suitability = self._compute_regime_suitability(metrics, returns)
        
        self.logger.debug(
            f"{ticker}: MC results - "
            f"mean={metrics.mean_return:.2%}, "
            f"VaR95={metrics.var_95:.2%}, "
            f"ES95={metrics.es_95:.2%}"
        )
        
        return metrics
    
    def _simulate_paths(self, returns: pd.Series) -> np.ndarray:
        """
        Simulate forward return paths using historical distribution.
        
        Args:
            returns: Historical returns to sample from
            
        Returns:
            Array of shape (n_simulations,) with simulated returns
        """
        # Use historical moments for simulations
        mean = returns.mean()
        std = returns.std()
        
        # For each day in horizon, sample from distribution
        daily_returns = np.random.normal(
            loc=mean,
            scale=std,
            size=(self.n_simulations, self.horizon_days)
        )
        
        # Compound returns over horizon
        # (1 + r1) * (1 + r2) * ... - 1 = horizon return
        compounded = np.prod(1 + daily_returns, axis=1) - 1
        
        return compounded
    
    def _compute_metrics(
        self,
        ticker: str,
        regime: str,
        simulated_returns: np.ndarray,
        freq_in_regime: int,
    ) -> MonteCarloMetrics:
        """
        Compute all risk metrics from simulated returns.
        
        Args:
            ticker: Asset ticker
            regime: Current regime
            simulated_returns: Array of simulated returns
            freq_in_regime: Number of historical observations
            
        Returns:
            MonteCarloMetrics object
        """
        returns = simulated_returns
        
        # Central tendency
        mean_return = float(np.mean(returns))
        median_return = float(np.median(returns))
        std_return = float(np.std(returns))
        
        # Percentiles for VaR
        var_95_pct = np.percentile(returns, 5)    # 5th percentile
        var_99_pct = np.percentile(returns, 1)    # 1st percentile
        
        # Expected Shortfall (mean of tail)
        es_95 = float(returns[returns <= np.percentile(returns, 5)].mean())
        es_99 = float(returns[returns <= np.percentile(returns, 1)].mean())
        
        # Probability metrics
        prob_loss = float(np.mean(returns < 0))
        prob_positive = float(np.mean(returns > 0))
        
        # Confidence intervals
        ci_95_lower = float(np.percentile(returns, 2.5))
        ci_95_upper = float(np.percentile(returns, 97.5))
        ci_99_lower = float(np.percentile(returns, 0.5))
        ci_99_upper = float(np.percentile(returns, 99.5))
        
        # Distribution shape
        skewness = float(((returns - mean_return) ** 3).mean() / (std_return ** 3)) if std_return > 0 else 0
        kurtosis = float(((returns - mean_return) ** 4).mean() / (std_return ** 4)) if std_return > 0 else 0
        
        metrics = MonteCarloMetrics(
            ticker=ticker,
            regime=regime,
            n_simulations=len(returns),
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            var_95=float(var_95_pct),
            var_99=float(var_99_pct),
            es_95=es_95,
            es_99=es_99,
            prob_loss=prob_loss,
            prob_positive=prob_positive,
            ci_95_lower=ci_95_lower,
            ci_95_upper=ci_95_upper,
            ci_99_lower=ci_99_lower,
            ci_99_upper=ci_99_upper,
            skewness=skewness,
            kurtosis=kurtosis,
            frequency_in_regime=freq_in_regime,
            simulated_returns=returns,
        )
        
        return metrics
    
    def _compute_regime_suitability(
        self,
        metrics: MonteCarloMetrics,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """
        Score how suitable an asset is for different regimes.
        
        Scoring logic:
        - Bull regime: High positive return + manageable risk
        - Bear regime: Positive return + low downside risk (bonds, low-beta)
        - Sideways: Positive return + low volatility
        
        Args:
            metrics: Computed metrics
            returns: Historical returns
            
        Returns:
            Dict mapping regime name to suitability score (0-1)
        """
        scores = {}
        
        # Return momentum score
        recent_returns = returns.iloc[-20:].mean() if len(returns) >= 20 else returns.mean()
        return_score = min(1.0, max(0.0, (recent_returns + 0.05) / 0.10))  # 0-1 scale
        
        # Risk score (lower risk = better for cautious regimes)
        risk_score = 1.0 - min(1.0, abs(metrics.es_95) / 0.10)  # 0-1 scale
        
        # Upside potential score
        upside_score = min(1.0, metrics.mean_return / 0.05) if metrics.mean_return > 0 else 0
        
        # Downside protection score
        downside_score = 1.0 - min(1.0, metrics.prob_loss / 0.5)
        
        # Stability score (low volatility)
        stability_score = 1.0 - min(1.0, metrics.std_return / 0.15)
        
        # Bull regime: High return + reasonable risk
        scores["Bull"] = (return_score * 0.4 + upside_score * 0.4 + risk_score * 0.2)
        
        # Bear regime: Low downside risk + stability
        scores["Bear"] = (downside_score * 0.4 + stability_score * 0.3 + risk_score * 0.3)
        
        # Sideways: Consistency + low vol
        scores["Sideways"] = (stability_score * 0.5 + downside_score * 0.5)
        
        # Volatility Spike: Quick recovery + resilience
        scores["Volatility Spike"] = downside_score * 0.7 + risk_score * 0.3
        
        # Recovery: Positive expected value + upside potential
        scores["Recovery"] = (upside_score * 0.4 + return_score * 0.4 + downside_score * 0.2)
        
        return scores
    
    def _create_default_metrics(self, ticker: str, regime: str) -> MonteCarloMetrics:
        """
        Create default metrics when insufficient data.
        
        Args:
            ticker: Asset ticker
            regime: Current regime
            
        Returns:
            MonteCarloMetrics with conservative defaults
        """
        return MonteCarloMetrics(
            ticker=ticker,
            regime=regime,
            n_simulations=0,
            mean_return=0.0,
            median_return=0.0,
            std_return=0.15,  # Assume 15% volatility
            var_95=-0.10,     # Conservative 10% downside
            var_99=-0.15,     # Conservative 15% tail
            es_95=-0.12,
            es_99=-0.18,
            prob_loss=0.5,
            prob_positive=0.5,
            ci_95_lower=-0.10,
            ci_95_upper=0.10,
            ci_99_lower=-0.15,
            ci_99_upper=0.15,
            skewness=0.0,
            kurtosis=3.0,
            frequency_in_regime=0,
        )
    
    def simulate_portfolio(
        self,
        asset_prices: Dict[str, pd.Series],
        asset_weights: Dict[str, float],
        current_regime: MarkovRegimeState,
    ) -> MonteCarloMetrics:
        """
        Simulate a portfolio combining multiple assets.
        
        Args:
            asset_prices: Dict of ticker → price series
            asset_weights: Dict of ticker → weight (should sum to 1)
            current_regime: Current market regime
            
        Returns:
            MonteCarloMetrics for the portfolio
        """
        # Simulate each asset
        asset_metrics = {}
        for ticker, prices in asset_prices.items():
            metrics = self.simulate_asset(prices, ticker, current_regime)
            asset_metrics[ticker] = metrics
        
        # Combine simulations weighted
        portfolio_returns = np.zeros(self.n_simulations)
        
        for ticker, weight in asset_weights.items():
            if ticker in asset_metrics:
                portfolio_returns += weight * asset_metrics[ticker].simulated_returns
        
        # Compute portfolio metrics
        portfolio_metrics = MonteCarloMetrics(
            ticker="PORTFOLIO",
            regime=current_regime.current_regime,
            n_simulations=self.n_simulations,
            mean_return=float(np.mean(portfolio_returns)),
            median_return=float(np.median(portfolio_returns)),
            std_return=float(np.std(portfolio_returns)),
            var_95=float(np.percentile(portfolio_returns, 5)),
            var_99=float(np.percentile(portfolio_returns, 1)),
            es_95=float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()),
            es_99=float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean()),
            prob_loss=float(np.mean(portfolio_returns < 0)),
            prob_positive=float(np.mean(portfolio_returns > 0)),
            ci_95_lower=float(np.percentile(portfolio_returns, 2.5)),
            ci_95_upper=float(np.percentile(portfolio_returns, 97.5)),
            ci_99_lower=float(np.percentile(portfolio_returns, 0.5)),
            ci_99_upper=float(np.percentile(portfolio_returns, 99.5)),
            skewness=float(((portfolio_returns - np.mean(portfolio_returns)) ** 3).mean() / (np.std(portfolio_returns) ** 3)),
            kurtosis=float(((portfolio_returns - np.mean(portfolio_returns)) ** 4).mean() / (np.std(portfolio_returns) ** 4)),
            simulated_returns=portfolio_returns,
        )
        
        self.logger.info(
            f"Portfolio simulation: mean={portfolio_metrics.mean_return:.2%}, "
            f"VaR95={portfolio_metrics.var_95:.2%}, ES95={portfolio_metrics.es_95:.2%}"
        )
        
        return portfolio_metrics
