"""
VaR Model Module
================
Historical Simulation VaR implementation.

Author: Risk Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VaRModel:
    """
    Historical Simulation VaR Model.
    
    Implements non-parametric VaR calculation using historical return distribution.
    Supports portfolio-level VaR calculation with custom weights.
    
    Attributes:
        returns (pd.DataFrame): Historical returns data
        weights (np.ndarray): Portfolio weights
        portfolio_value (float): Total portfolio value
    """
    
    def __init__(self, returns: pd.DataFrame, weights: List[float], portfolio_value: float = 10_000_000):
        """
        Initialize VaR model.
        
        Args:
            returns (pd.DataFrame): Historical returns for each asset
            weights (List[float]): Portfolio weights (must sum to 1.0)
            portfolio_value (float): Total portfolio value in currency units
        """
        self.returns = returns
        self.weights = np.array(weights)
        self.portfolio_value = portfolio_value
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate portfolio returns
        self.portfolio_returns = self._calculate_portfolio_returns()
        
        logger.info(f"VaR Model initialized: {len(returns)} observations, {len(weights)} assets")
    
    def _validate_inputs(self) -> None:
        """Validate model inputs."""
        # Check weights
        if len(self.weights) != self.returns.shape[1]:
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of assets ({self.returns.shape[1]})")
        
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0 (current sum: {self.weights.sum():.6f})")
        
        if (self.weights < 0).any():
            logger.warning("Negative weights detected (short positions)")
        
        # Check for missing values
        if self.returns.isna().any().any():
            raise ValueError("Returns contain NaN values. Please clean data first.")
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate portfolio returns from individual asset returns.
        
        Returns:
            pd.Series: Portfolio returns
        """
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        return portfolio_returns
    
    def historical_var(self, confidence: float = 0.95, time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Historical Simulation VaR.
        
        Args:
            confidence (float): Confidence level (e.g., 0.95, 0.99)
            time_horizon (int): Time horizon in days
            
        Returns:
            Dict[str, float]: VaR results including absolute and percentage values
        """
        logger.info(f"Calculating Historical VaR at {confidence*100}% confidence")
        
        # Get portfolio returns
        portfolio_returns = self.portfolio_returns
        
        # Scale returns for time horizon (square root rule for volatility)
        if time_horizon > 1:
            portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # Calculate VaR as percentile
        var_percentile = 1 - confidence
        var_return = np.percentile(portfolio_returns, var_percentile * 100)
        
        # Convert to absolute loss
        var_absolute = -var_return * self.portfolio_value
        
        # Additional statistics
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        results = {
            'confidence': confidence,
            'time_horizon': time_horizon,
            'var_return': var_return,
            'var_absolute': var_absolute,
            'var_percentage': -var_return * 100,
            'mean_return': mean_return,
            'volatility': std_return,
            'observations': len(portfolio_returns)
        }
        
        logger.info(f"VaR ({confidence*100}%): ${var_absolute:,.2f} ({-var_return*100:.2f}%)")
        
        return results
    
    def calculate_multiple_vars(self, confidences: List[float] = [0.95, 0.99]) -> pd.DataFrame:
        """
        Calculate VaR at multiple confidence levels.
        
        Args:
            confidences (List[float]): List of confidence levels
            
        Returns:
            pd.DataFrame: VaR results for all confidence levels
        """
        results = []
        for conf in confidences:
            var_result = self.historical_var(confidence=conf)
            results.append({
                'Confidence': f"{conf*100:.0f}%",
                'VaR (Currency)': var_result['var_absolute'],
                'VaR (%)': var_result['var_percentage'],
                'VaR (Return)': var_result['var_return']
            })
        
        return pd.DataFrame(results)
    
    def component_var(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Calculate Component VaR (contribution of each asset to portfolio VaR).
        
        Args:
            confidence (float): Confidence level
            
        Returns:
            pd.DataFrame: Component VaR for each asset
        """
        logger.info("Calculating Component VaR")
        
        # Portfolio VaR
        portfolio_var = self.historical_var(confidence=confidence)
        
        # Calculate marginal VaR for each asset
        components = []
        
        for i, asset in enumerate(self.returns.columns):
            # Correlation between asset and portfolio returns
            corr = self.returns.iloc[:, i].corr(self.portfolio_returns)
            
            # Asset volatility
            asset_vol = self.returns.iloc[:, i].std()
            
            # Portfolio volatility
            portfolio_vol = self.portfolio_returns.std()
            
            # Marginal VaR = (correlation * asset_vol * portfolio_var) / portfolio_vol
            if portfolio_vol > 0:
                marginal_var = (corr * asset_vol / portfolio_vol)
            else:
                marginal_var = 0
            
            # Component VaR = weight * marginal_var * portfolio_var
            component_var = self.weights[i] * marginal_var * portfolio_var['var_absolute']
            
            # Percentage contribution
            if portfolio_var['var_absolute'] > 0:
                pct_contribution = (component_var / portfolio_var['var_absolute']) * 100
            else:
                pct_contribution = 0
            
            components.append({
                'Asset': asset,
                'Weight': self.weights[i],
                'Marginal VaR': marginal_var,
                'Component VaR': component_var,
                'Contribution (%)': pct_contribution
            })
        
        return pd.DataFrame(components)
    
    def calculate_expected_shortfall(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate Expected Shortfall (CVaR) - average loss beyond VaR.
        
        Args:
            confidence (float): Confidence level
            
        Returns:
            Dict[str, float]: Expected Shortfall results
        """
        logger.info(f"Calculating Expected Shortfall at {confidence*100}% confidence")
        
        # Get VaR threshold
        var_result = self.historical_var(confidence=confidence)
        var_return = var_result['var_return']
        
        # Get returns worse than VaR
        tail_returns = self.portfolio_returns[self.portfolio_returns <= var_return]
        
        # Calculate average of tail
        es_return = tail_returns.mean()
        es_absolute = -es_return * self.portfolio_value
        
        results = {
            'confidence': confidence,
            'es_return': es_return,
            'es_absolute': es_absolute,
            'es_percentage': -es_return * 100,
            'var_absolute': var_result['var_absolute'],
            'es_var_ratio': es_absolute / var_result['var_absolute'] if var_result['var_absolute'] > 0 else 0
        }
        
        logger.info(f"Expected Shortfall ({confidence*100}%): ${es_absolute:,.2f}")
        
        return results
    
    def get_portfolio_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio statistics.
        
        Returns:
            Dict[str, float]: Portfolio statistics
        """
        portfolio_returns = self.portfolio_returns
        
        # Annualization factors
        trading_days = 252
        
        # Calculate metrics
        stats = {
            'Mean Return (Daily)': portfolio_returns.mean(),
            'Mean Return (Annual)': portfolio_returns.mean() * trading_days,
            'Volatility (Daily)': portfolio_returns.std(),
            'Volatility (Annual)': portfolio_returns.std() * np.sqrt(trading_days),
            'Skewness': portfolio_returns.skew(),
            'Kurtosis': portfolio_returns.kurtosis(),
            'Min Return': portfolio_returns.min(),
            'Max Return': portfolio_returns.max(),
            '5th Percentile': portfolio_returns.quantile(0.05),
            '95th Percentile': portfolio_returns.quantile(0.95),
            'Sharpe Ratio (Annual)': (portfolio_returns.mean() * trading_days) / (portfolio_returns.std() * np.sqrt(trading_days)) if portfolio_returns.std() > 0 else 0,
            'Portfolio Value': self.portfolio_value,
            'Observations': len(portfolio_returns)
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    returns = pd.DataFrame(
        np.random.randn(500, 5) * 0.015,
        index=dates,
        columns=[f'Asset_{i}' for i in range(1, 6)]
    )
    
    # Equal weights
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Initialize VaR model
    var_model = VaRModel(returns, weights)
    
    # Calculate VaR at multiple confidence levels
    print("\n=== VaR Results ===")
    print(var_model.calculate_multiple_vars([0.95, 0.99]))
    
    # Component VaR
    print("\n=== Component VaR ===")
    print(var_model.component_var(0.99))
    
    # Expected Shortfall
    print("\n=== Expected Shortfall ===")
    es = var_model.calculate_expected_shortfall(0.99)
    print(f"Expected Shortfall (99%): ${es['es_absolute']:,.2f}")
    
    # Portfolio statistics
    print("\n=== Portfolio Statistics ===")
    for key, value in var_model.get_portfolio_statistics().items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
