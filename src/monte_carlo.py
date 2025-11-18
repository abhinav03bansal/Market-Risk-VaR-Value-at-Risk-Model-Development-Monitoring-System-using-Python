"""
Monte Carlo VaR Module
======================
Parametric Monte Carlo VaR simulation.

Author: Risk Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MonteCarloVaR:
    """
    Monte Carlo VaR simulation engine.
    
    Generates random return scenarios based on historical statistics
    to estimate VaR using parametric assumptions.
    
    Attributes:
        returns (pd.DataFrame): Historical returns data
        weights (np.ndarray): Portfolio weights
        portfolio_value (float): Total portfolio value
        mean_returns (np.ndarray): Mean returns for each asset
        cov_matrix (np.ndarray): Covariance matrix
    """
    
    def __init__(self, returns: pd.DataFrame, weights: List[float], portfolio_value: float = 10_000_000):
        """
        Initialize Monte Carlo VaR model.
        
        Args:
            returns (pd.DataFrame): Historical returns for each asset
            weights (List[float]): Portfolio weights (must sum to 1.0)
            portfolio_value (float): Total portfolio value in currency units
        """
        self.returns = returns
        self.weights = np.array(weights)
        self.portfolio_value = portfolio_value
        
        # Calculate parameters from historical data
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        
        logger.info(f"Monte Carlo VaR initialized: {len(returns)} historical observations")
    
    def simulate_returns(self, n_simulations: int = 10000, time_horizon: int = 1, seed: int = None) -> np.ndarray:
        """
        Generate random return scenarios.
        
        Args:
            n_simulations (int): Number of Monte Carlo simulations
            time_horizon (int): Time horizon in days
            seed (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: Simulated portfolio returns
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Running {n_simulations:,} Monte Carlo simulations...")
        
        # Generate random returns for each asset
        # Assumes multivariate normal distribution
        simulated_asset_returns = np.random.multivariate_normal(
            mean=self.mean_returns,
            cov=self.cov_matrix,
            size=n_simulations
        )
        
        # Scale for time horizon (square root rule)
        if time_horizon > 1:
            simulated_asset_returns = simulated_asset_returns * np.sqrt(time_horizon)
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(simulated_asset_returns, self.weights)
        
        return portfolio_returns
    
    def calculate_var(self, confidence: float = 0.95, n_simulations: int = 10000, 
                     time_horizon: int = 1, seed: int = 42) -> Dict[str, float]:
        """
        Calculate Monte Carlo VaR.
        
        Args:
            confidence (float): Confidence level (e.g., 0.95, 0.99)
            n_simulations (int): Number of simulations
            time_horizon (int): Time horizon in days
            seed (int): Random seed
            
        Returns:
            Dict[str, float]: VaR results
        """
        logger.info(f"Calculating Monte Carlo VaR at {confidence*100}% confidence")
        
        # Run simulations
        portfolio_returns = self.simulate_returns(n_simulations, time_horizon, seed)
        
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
            'n_simulations': n_simulations,
            'var_return': var_return,
            'var_absolute': var_absolute,
            'var_percentage': -var_return * 100,
            'mean_simulated_return': mean_return,
            'std_simulated_return': std_return,
            'min_simulated_return': portfolio_returns.min(),
            'max_simulated_return': portfolio_returns.max()
        }
        
        logger.info(f"Monte Carlo VaR ({confidence*100}%): ${var_absolute:,.2f} ({-var_return*100:.2f}%)")
        
        return results
    
    def calculate_expected_shortfall(self, confidence: float = 0.95, 
                                    n_simulations: int = 10000, 
                                    time_horizon: int = 1, 
                                    seed: int = 42) -> Dict[str, float]:
        """
        Calculate Expected Shortfall using Monte Carlo simulation.
        
        Args:
            confidence (float): Confidence level
            n_simulations (int): Number of simulations
            time_horizon (int): Time horizon in days
            seed (int): Random seed
            
        Returns:
            Dict[str, float]: Expected Shortfall results
        """
        logger.info(f"Calculating Monte Carlo Expected Shortfall at {confidence*100}% confidence")
        
        # Run simulations
        portfolio_returns = self.simulate_returns(n_simulations, time_horizon, seed)
        
        # Get VaR threshold
        var_percentile = 1 - confidence
        var_return = np.percentile(portfolio_returns, var_percentile * 100)
        
        # Calculate Expected Shortfall (average of returns worse than VaR)
        tail_returns = portfolio_returns[portfolio_returns <= var_return]
        es_return = tail_returns.mean()
        es_absolute = -es_return * self.portfolio_value
        
        results = {
            'confidence': confidence,
            'n_simulations': n_simulations,
            'es_return': es_return,
            'es_absolute': es_absolute,
            'es_percentage': -es_return * 100,
            'var_return': var_return,
            'var_absolute': -var_return * self.portfolio_value,
            'tail_observations': len(tail_returns)
        }
        
        logger.info(f"Monte Carlo ES ({confidence*100}%): ${es_absolute:,.2f}")
        
        return results
    
    def calculate_multiple_vars(self, confidences: List[float] = [0.95, 0.99], 
                              n_simulations: int = 10000) -> pd.DataFrame:
        """
        Calculate VaR at multiple confidence levels using Monte Carlo.
        
        Args:
            confidences (List[float]): List of confidence levels
            n_simulations (int): Number of simulations
            
        Returns:
            pd.DataFrame: VaR results for all confidence levels
        """
        results = []
        for conf in confidences:
            var_result = self.calculate_var(confidence=conf, n_simulations=n_simulations)
            results.append({
                'Confidence': f"{conf*100:.0f}%",
                'VaR (Currency)': var_result['var_absolute'],
                'VaR (%)': var_result['var_percentage'],
                'VaR (Return)': var_result['var_return'],
                'Simulations': n_simulations
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(self, confidence: float = 0.99, 
                           n_simulations_list: List[int] = [1000, 5000, 10000, 50000]) -> pd.DataFrame:
        """
        Analyze VaR sensitivity to number of simulations.
        
        Args:
            confidence (float): Confidence level
            n_simulations_list (List[int]): List of simulation counts to test
            
        Returns:
            pd.DataFrame: Sensitivity analysis results
        """
        logger.info("Running Monte Carlo sensitivity analysis...")
        
        results = []
        for n_sims in n_simulations_list:
            var_result = self.calculate_var(confidence=confidence, n_simulations=n_sims)
            results.append({
                'Simulations': n_sims,
                'VaR (Currency)': var_result['var_absolute'],
                'VaR (%)': var_result['var_percentage'],
                'Std Dev': var_result['std_simulated_return']
            })
        
        return pd.DataFrame(results)
    
    def compare_distributions(self, confidence: float = 0.99, n_simulations: int = 10000) -> Dict:
        """
        Compare normal vs t-distribution for VaR calculation.
        
        Args:
            confidence (float): Confidence level
            n_simulations (int): Number of simulations
            
        Returns:
            Dict: Comparison results
        """
        logger.info("Comparing distribution assumptions...")
        
        # Normal distribution (default)
        normal_var = self.calculate_var(confidence=confidence, n_simulations=n_simulations)
        
        # Student's t-distribution (heavier tails)
        np.random.seed(42)
        df = 5  # degrees of freedom (lower = heavier tails)
        
        # Generate t-distributed returns
        t_returns = np.random.standard_t(df, size=(n_simulations, len(self.mean_returns)))
        
        # Scale by historical volatility
        std_devs = np.sqrt(np.diag(self.cov_matrix))
        scaled_returns = t_returns * std_devs + self.mean_returns
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(scaled_returns, self.weights)
        
        # Calculate t-distribution VaR
        var_return_t = np.percentile(portfolio_returns, (1 - confidence) * 100)
        var_absolute_t = -var_return_t * self.portfolio_value
        
        comparison = {
            'normal_var': normal_var['var_absolute'],
            't_distribution_var': var_absolute_t,
            'difference': var_absolute_t - normal_var['var_absolute'],
            'difference_pct': ((var_absolute_t / normal_var['var_absolute']) - 1) * 100 if normal_var['var_absolute'] > 0 else 0
        }
        
        logger.info(f"Distribution comparison - Normal: ${normal_var['var_absolute']:,.2f}, t-dist: ${var_absolute_t:,.2f}")
        
        return comparison
    
    def get_simulation_statistics(self, n_simulations: int = 10000) -> Dict[str, float]:
        """
        Get statistics from Monte Carlo simulations.
        
        Args:
            n_simulations (int): Number of simulations
            
        Returns:
            Dict[str, float]: Simulation statistics
        """
        portfolio_returns = self.simulate_returns(n_simulations=n_simulations)
        
        stats = {
            'Mean': portfolio_returns.mean(),
            'Std Dev': portfolio_returns.std(),
            'Skewness': stats.skew(portfolio_returns),
            'Kurtosis': stats.kurtosis(portfolio_returns),
            'Min': portfolio_returns.min(),
            'Max': portfolio_returns.max(),
            '1st Percentile': np.percentile(portfolio_returns, 1),
            '5th Percentile': np.percentile(portfolio_returns, 5),
            '95th Percentile': np.percentile(portfolio_returns, 95),
            '99th Percentile': np.percentile(portfolio_returns, 99),
            'Simulations': n_simulations
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
    
    # Initialize Monte Carlo VaR
    mc_var = MonteCarloVaR(returns, weights)
    
    # Calculate VaR
    print("\n=== Monte Carlo VaR Results ===")
    print(mc_var.calculate_multiple_vars([0.95, 0.99], n_simulations=10000))
    
    # Expected Shortfall
    print("\n=== Expected Shortfall ===")
    es = mc_var.calculate_expected_shortfall(0.99, n_simulations=10000)
    print(f"Expected Shortfall (99%): ${es['es_absolute']:,.2f}")
    
    # Sensitivity analysis
    print("\n=== Sensitivity Analysis ===")
    print(mc_var.sensitivity_analysis())
    
    # Distribution comparison
    print("\n=== Distribution Comparison ===")
    comparison = mc_var.compare_distributions()
    print(f"Normal VaR: ${comparison['normal_var']:,.2f}")
    print(f"t-distribution VaR: ${comparison['t_distribution_var']:,.2f}")
    print(f"Difference: ${comparison['difference']:,.2f} ({comparison['difference_pct']:.2f}%)")
