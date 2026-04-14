"""
Markov Chain Regime Detection (Enhanced)

Uses Hidden Markov Model to detect market regimes and provide:
- Current regime identification
- Transition probability matrix
- Confidence levels
- Regime-specific statistics

Regimes detected:
- Bull: High positive returns, moderate volatility
- Bear: Negative returns, high volatility
- Sideways: Near-zero returns, low volatility
- Volatility Spike: Any direction, extreme volatility
- Recovery: Mean reversion after shock
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
import pickle

try:
    from hmmlearn import hmm
except ImportError:
    logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
    hmm = None


@dataclass
class MarkovRegimeState:
    """Current market regime state from Markov chain analysis."""
    current_regime: str                          # "Bull", "Bear", "Sideways", etc.
    regime_probability: float                    # 0-1 confidence in current regime
    transition_matrix: np.ndarray                # n_states × n_states probabilities
    probability_next_regime: Dict[str, float]    # Next regime probabilities
    time_in_regime: timedelta                    # How long in current regime
    regime_features: Dict[str, float] = field(default_factory=dict)  # Returns, vol, etc.
    
    # Metadata
    detection_date: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0"
    confidence_score: float = 0.0  # Overall confidence


class MarkovChainRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    Process:
    1. Extract features from price data (returns, volatility)
    2. Fit HMM to learn hidden market states
    3. Identify current regime based on latest data
    4. Compute transition probabilities
    5. Return regime state for downstream use
    """
    
    # Standard regime names (ordered by expected return)
    REGIME_NAMES = ["Bear", "Sideways", "Bull", "Volatility Spike", "Recovery"]
    
    def __init__(
        self,
        n_states: int = 5,
        n_iter: int = 100,
        random_state: int = 42,
        model_path: Optional[Path] = None,
        lookback_days: int = 252  # 1 year of data
    ):
        """
        Initialize Markov Chain Regime Detector.
        
        Args:
            n_states: Number of hidden states (default 5)
            n_iter: Number of EM iterations for HMM training
            random_state: Random seed for reproducibility
            model_path: Path to save/load trained model
            lookback_days: Days of historical data for feature extraction
        """
        if hmm is None:
            raise ImportError("hmmlearn required. Install with: pip install hmmlearn")
        
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_path = model_path or Path("models/markov_regime_model.pkl")
        self.lookback_days = lookback_days
        
        self.model: Optional[hmm.GaussianHMM] = None
        self.scaler_mean = None
        self.scaler_std = None
        self.fitted = False
        self.regime_order = None  # Maps state indices to regime names
        
        self.logger = logger.bind(module="markov_chain_detector")
    
    def _extract_features(
        self,
        prices: pd.Series,
        vol_window: int = 20
    ) -> np.ndarray:
        """
        Extract features for HMM from price series.
        
        Features:
        1. Daily returns (normalized)
        2. Rolling volatility (normalized)
        
        Args:
            prices: Series of closing prices
            vol_window: Window for rolling volatility
            
        Returns:
            Array of shape (n_samples, 2) with [returns, volatility]
        """
        if len(prices) < vol_window + 1:
            raise ValueError(f"Need at least {vol_window + 1} price points")
        
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=vol_window).std()
        
        # Combine into feature matrix
        features = np.column_stack([returns.values[vol_window-1:], volatility.values[vol_window:]])
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        self.logger.debug(f"Extracted {len(features)} feature vectors")
        
        return features
    
    def fit(self, prices: pd.Series) -> None:
        """
        Fit HMM to historical price data.
        
        Args:
            prices: Series of closing prices (date index, price values)
        """
        self.logger.info(f"Fitting HMM with {self.n_states} states to {len(prices)} price points")
        
        # Extract features
        features = self._extract_features(prices)
        
        # Standardize features for HMM
        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0  # Avoid division by zero
        
        features_scaled = (features - self.scaler_mean) / self.scaler_std
        
        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.model.fit(features_scaled)
        
        # Identify regime ordering based on mean returns
        # (state with highest return = Bull, etc.)
        self._identify_regime_order()
        
        self.fitted = True
        self.logger.info(f"✓ HMM fitted. Transition matrix shape: {self.model.transmat_.shape}")
        
        # Save model
        self._save_model()
    
    def _identify_regime_order(self) -> None:
        """
        Identify which state index corresponds to which regime.
        
        Orders states by their mean return (higher return = Bull, lower = Bear)
        """
        if self.model is None or self.model.means_ is None:
            self.regime_order = list(range(self.n_states))
            return
        
        # Mean return is first feature
        mean_returns = self.model.means_[:, 0]
        
        # Sort state indices by mean return (descending = Bull first)
        sorted_indices = np.argsort(mean_returns)[::-1]
        
        # Ensure we have enough regime names
        regime_names = self.REGIME_NAMES[:self.n_states]
        
        # Create mapping: state_index → regime_name
        self.regime_order = {}
        for rank, state_idx in enumerate(sorted_indices):
            self.regime_order[state_idx] = regime_names[rank]
        
        self.logger.debug(f"Regime order: {self.regime_order}")
    
    def detect_current_regime(
        self,
        prices: pd.Series,
        lookback: Optional[int] = None
    ) -> MarkovRegimeState:
        """
        Detect current market regime based on latest price data.
        
        Args:
            prices: Series of closing prices
            lookback: Number of recent days to use (default: lookback_days)
            
        Returns:
            MarkovRegimeState with current regime info
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted before calling detect_current_regime")
        
        lookback = lookback or self.lookback_days
        
        # Use recent data
        recent_prices = prices.tail(lookback)
        
        # Extract features
        features = self._extract_features(recent_prices)
        
        # Standardize using fitted scalers
        features_scaled = (features - self.scaler_mean) / self.scaler_std
        
        # Get hidden state sequence
        hidden_states = self.model.predict(features_scaled)
        current_state_idx = hidden_states[-1]
        
        # Get state probabilities for latest data point
        state_probs = self.model.predict_proba(features_scaled[-1:]).flatten()
        current_regime_prob = state_probs[current_state_idx]
        
        # Get current regime name
        current_regime = self.regime_order.get(current_state_idx, f"State_{current_state_idx}")
        
        # Compute how long in current regime
        time_in_regime = self._compute_time_in_regime(hidden_states)
        
        # Get next regime probabilities
        next_probs = self.model.transmat_[current_state_idx]
        probability_next_regime = {
            self.regime_order.get(i, f"State_{i}"): float(prob)
            for i, prob in enumerate(next_probs)
        }
        
        # Extract regime features from latest data
        latest_returns = recent_prices.pct_change().iloc[-20:].mean()
        latest_volatility = recent_prices.pct_change().iloc[-20:].std()
        
        regime_features = {
            "mean_return": float(latest_returns),
            "volatility": float(latest_volatility),
            "return_trend": float(recent_prices.pct_change().iloc[-5:].mean()),
            "volatility_trend": float(recent_prices.pct_change().iloc[-5:].std()),
        }
        
        state = MarkovRegimeState(
            current_regime=current_regime,
            regime_probability=float(current_regime_prob),
            transition_matrix=self.model.transmat_.copy(),
            probability_next_regime=probability_next_regime,
            time_in_regime=time_in_regime,
            regime_features=regime_features,
            detection_date=datetime.now(),
            model_version="1.0"
        )
        
        self.logger.info(
            f"Detected regime: {current_regime} "
            f"(confidence: {current_regime_prob:.2%}, time: {time_in_regime.days} days)"
        )
        
        return state
    
    def _compute_time_in_regime(self, hidden_states: np.ndarray) -> timedelta:
        """
        Compute how long the market has been in current regime.
        
        Args:
            hidden_states: Array of state indices over time
            
        Returns:
            Timedelta representing duration in current regime
        """
        if len(hidden_states) == 0:
            return timedelta(days=0)
        
        current_state = hidden_states[-1]
        
        # Count consecutive days in current state (backward)
        days_in_state = 1
        for i in range(len(hidden_states) - 2, -1, -1):
            if hidden_states[i] == current_state:
                days_in_state += 1
            else:
                break
        
        return timedelta(days=days_in_state)
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the transition probability matrix.
        
        Returns:
            Array of shape (n_states, n_states) where element [i,j]
            is probability of transitioning from state i to state j
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted first")
        return self.model.transmat_.copy()
    
    def get_transition_probabilities(self, from_regime: str) -> Dict[str, float]:
        """
        Get transition probabilities from a specific regime.
        
        Args:
            from_regime: Regime name (e.g., "Bull", "Bear")
            
        Returns:
            Dict mapping target regime to probability
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted first")
        
        # Find state index for from_regime
        from_state = None
        for state_idx, regime_name in self.regime_order.items():
            if regime_name == from_regime:
                from_state = state_idx
                break
        
        if from_state is None:
            raise ValueError(f"Regime '{from_regime}' not found")
        
        probs = self.model.transmat_[from_state]
        return {
            self.regime_order.get(i, f"State_{i}"): float(prob)
            for i, prob in enumerate(probs)
        }
    
    def filter_returns_by_regime(
        self,
        prices: pd.Series,
        target_regime: str
    ) -> pd.Series:
        """
        Filter historical returns to only those from a specific regime.
        
        Useful for Monte Carlo: simulate only returns from current regime type.
        
        Args:
            prices: Historical price series
            target_regime: Regime to filter for (e.g., "Bull")
            
        Returns:
            Series of returns that occurred during target regime
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted first")
        
        # Extract features
        features = self._extract_features(prices)
        
        # Standardize
        features_scaled = (features - self.scaler_mean) / self.scaler_std
        
        # Get hidden states
        hidden_states = self.model.predict(features_scaled)
        
        # Find target state index
        target_state = None
        for state_idx, regime_name in self.regime_order.items():
            if regime_name == target_regime:
                target_state = state_idx
                break
        
        if target_state is None:
            raise ValueError(f"Regime '{target_regime}' not found")
        
        # Get returns corresponding to target regime
        returns = prices.pct_change().dropna()
        mask = hidden_states == target_state
        
        # Align mask with returns (feature extraction offsets)
        vol_window = 20
        mask = np.append(np.zeros(vol_window, dtype=bool), mask)[:len(returns)]
        
        filtered_returns = returns[mask]
        
        self.logger.debug(
            f"Filtered {len(filtered_returns)}/{len(returns)} returns for regime '{target_regime}'"
        )
        
        return filtered_returns
    
    def _save_model(self) -> None:
        """Save trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            'model': self.model,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'regime_order': self.regime_order,
            'n_states': self.n_states,
            'fitted': self.fitted,
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        self.logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Load trained model from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.model_path.exists():
            self.logger.warning(f"Model file not found: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                state_dict = pickle.load(f)
            
            self.model = state_dict['model']
            self.scaler_mean = state_dict['scaler_mean']
            self.scaler_std = state_dict['scaler_std']
            self.regime_order = state_dict['regime_order']
            self.n_states = state_dict['n_states']
            self.fitted = state_dict['fitted']
            
            self.logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_regime_statistics(self, prices: pd.Series, regime: str) -> Dict[str, float]:
        """
        Get statistics for a specific regime (mean return, volatility, etc.).
        
        Args:
            prices: Historical price series
            regime: Regime name
            
        Returns:
            Dict of statistics
        """
        filtered_returns = self.filter_returns_by_regime(prices, regime)
        
        if len(filtered_returns) == 0:
            return {}
        
        return {
            "mean_return": float(filtered_returns.mean()),
            "median_return": float(filtered_returns.median()),
            "std_return": float(filtered_returns.std()),
            "min_return": float(filtered_returns.min()),
            "max_return": float(filtered_returns.max()),
            "skewness": float(filtered_returns.skew()),
            "kurtosis": float(filtered_returns.kurtosis()),
            "n_observations": len(filtered_returns),
        }
