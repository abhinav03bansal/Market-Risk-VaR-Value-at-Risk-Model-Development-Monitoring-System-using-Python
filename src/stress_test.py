"""
Stress Testing Module
=====================
Scenario analysis and shock testing for portfolio risk assessment.

Author: Risk Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StressTester:
    """
    Portfolio stress testing engine.
    
    Applies various shock scenarios to assess portfolio resilience
    under extreme market conditions.
    
    Attributes:
        returns (pd.DataFrame): Historical returns data
        weights (np.ndarray): Portfolio weights
        portfolio_value (float): Total portfolio value
    """
    
    def __init__(self, returns: pd.DataFrame, weights: List[float], portfolio_value: float = 10_000_000):
        """
        Initialize stress tester.
        
        Args:
            returns (pd.DataFrame): Historical returns for each asset
            weights (List[float]): Portfolio weights
            portfolio_value (float): Total portfolio value
        """
        self.returns = returns
        self.weights = np.array(weights)
        self.portfolio_value = portfolio_value
        self.asset_names = returns.columns.tolist()
        
        logger.info(f"Stress Tester initialized: {len(self.asset_names)} assets")
    
    def apply_shock(self, shocks: List[float], scenario_name: str = "Custom") -> Dict:
        """
        Apply shock scenario to portfolio.
        
        Args:
            shocks (List[float]): Shock magnitudes for each asset (e.g., -0.10 for -10%)
            scenario_name (str): Name of the scenario
            
        Returns:
            Dict: Shock results including losses and risk metrics
        """
        if len(shocks) != len(self.weights):
            raise ValueError(f"Number of shocks ({len(shocks)}) must match number of assets ({len(self.weights)})")
        
        shocks_array = np.array(shocks)
        
        # Calculate portfolio return under shock
        portfolio_shocked_return = np.dot(shocks_array, self.weights)
        
        # Calculate portfolio loss
        portfolio_loss = -portfolio_shocked_return * self.portfolio_value
        
        # Calculate individual asset impacts
        asset_impacts = []
        for i, asset in enumerate(self.asset_names):
            asset_contribution = shocks_array[i] * self.weights[i] * self.portfolio_value
            asset_impacts.append({
                'Asset': asset,
                'Weight': self.weights[i],
                'Shock': shocks_array[i],
                'Shock (%)': shocks_array[i] * 100,
                'Contribution to Loss': -asset_contribution
            })
        
        results = {
            'scenario_name': scenario_name,
            'portfolio_return': portfolio_shocked_return,
            'portfolio_loss': portfolio_loss,
            'portfolio_loss_pct': -portfolio_shocked_return * 100,
            'asset_impacts': pd.DataFrame(asset_impacts)
        }
        
        logger.info(f"Scenario '{scenario_name}': Loss = ${portfolio_loss:,.2f} ({-portfolio_shocked_return*100:.2f}%)")
        
        return results
    
    def run_standard_scenarios(self) -> Dict[str, Dict]:
        """
        Run predefined standard stress scenarios.
        
        Returns:
            Dict[str, Dict]: Results for each scenario
        """
        logger.info("Running standard stress scenarios...")
        
        scenarios = {
            'Market Crash (-10%)': [-0.10] * len(self.weights),
            'Moderate Decline (-5%)': [-0.05] * len(self.weights),
            'Market Rally (+5%)': [0.05] * len(self.weights),
            'Strong Rally (+10%)': [0.10] * len(self.weights),
            'Severe Crash (-20%)': [-0.20] * len(self.weights),
            'Flash Crash (-15%)': [-0.15] * len(self.weights)
        }
        
        results = {}
        for scenario_name, shocks in scenarios.items():
            results[scenario_name] = self.apply_shock(shocks, scenario_name)
        
        return results
    
    def run_individual_shocks(self, shock_size: float = -0.10) -> Dict[str, Dict]:
        """
        Apply individual shocks to each asset while keeping others constant.
        
        Args:
            shock_size (float): Shock magnitude (e.g., -0.10 for -10%)
            
        Returns:
            Dict[str, Dict]: Results for each individual asset shock
        """
        logger.info(f"Running individual asset shocks ({shock_size*100:.0f}%)...")
        
        results = {}
        for i, asset in enumerate(self.asset_names):
            shocks = [0.0] * len(self.weights)
            shocks[i] = shock_size
            
            scenario_name = f"{asset} Shock ({shock_size*100:.0f}%)"
            results[scenario_name] = self.apply_shock(shocks, scenario_name)
        
        return results
    
    def run_correlation_breakdown(self) -> Dict:
        """
        Stress test assuming all correlations go to 1 (perfect correlation).
        
        Returns:
            Dict: Results under correlation breakdown scenario
        """
        logger.info("Running correlation breakdown scenario...")
        
        # Calculate portfolio volatility under perfect correlation
        asset_vols = self.returns.std().values
        
        # Under perfect correlation, portfolio vol = weighted sum of individual vols
        portfolio_vol_perfect = np.dot(asset_vols, np.abs(self.weights))
        
        # Current portfolio volatility
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        current_vol = portfolio_returns.std()
        
        # Volatility increase
        vol_increase = portfolio_vol_perfect - current_vol
        vol_increase_pct = (vol_increase / current_vol) * 100 if current_vol > 0 else 0
        
        # Estimate VaR impact (simplified)
        # Assume VaR scales with volatility
        var_increase_pct = vol_increase_pct
        
        results = {
            'scenario_name': 'Correlation Breakdown (All correlations → 1)',
            'current_volatility': current_vol,
            'stressed_volatility': portfolio_vol_perfect,
            'volatility_increase': vol_increase,
            'volatility_increase_pct': vol_increase_pct,
            'var_increase_pct': var_increase_pct,
            'interpretation': 'Diversification benefit is lost'
        }
        
        logger.info(f"Correlation breakdown: Volatility increases by {vol_increase_pct:.2f}%")
        
        return results
    
    def run_historical_scenarios(self, n_worst: int = 10) -> pd.DataFrame:
        """
        Identify worst historical scenarios.
        
        Args:
            n_worst (int): Number of worst scenarios to return
            
        Returns:
            pd.DataFrame: Worst historical scenarios
        """
        logger.info(f"Identifying {n_worst} worst historical scenarios...")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        # Get worst returns
        worst_returns = portfolio_returns.nsmallest(n_worst)
        
        # Create results DataFrame
        worst_scenarios = pd.DataFrame({
            'Date': worst_returns.index,
            'Return': worst_returns.values,
            'Return (%)': worst_returns.values * 100,
            'Loss': -worst_returns.values * self.portfolio_value
        })
        
        worst_scenarios.reset_index(drop=True, inplace=True)
        worst_scenarios.index = worst_scenarios.index + 1
        worst_scenarios.index.name = 'Rank'
        
        return worst_scenarios
    
    def run_var_multiple_scenarios(self, multiples: List[float] = [1.5, 2.0, 3.0]) -> pd.DataFrame:
        """
        Apply shocks as multiples of historical VaR.
        
        Args:
            multiples (List[float]): VaR multiples to test
            
        Returns:
            pd.DataFrame: Results for VaR multiple scenarios
        """
        logger.info("Running VaR multiple scenarios...")
        
        # Calculate historical 99% VaR
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        var_99 = np.percentile(portfolio_returns, 1)
        var_99_absolute = -var_99 * self.portfolio_value
        
        results = []
        for multiple in multiples:
            shocked_return = var_99 * multiple
            shocked_loss = -shocked_return * self.portfolio_value
            
            results.append({
                'Scenario': f'{multiple}x VaR',
                'Multiple': multiple,
                'Shocked Return': shocked_return,
                'Shocked Return (%)': shocked_return * 100,
                'Loss': shocked_loss,
                'VaR_99': var_99_absolute
            })
        
        return pd.DataFrame(results)
    
    def generate_stress_report(self) -> Dict:
        """
        Generate comprehensive stress testing report.
        
        Returns:
            Dict: Complete stress test results
        """
        logger.info("Generating comprehensive stress test report...")
        
        report = {
            'standard_scenarios': self.run_standard_scenarios(),
            'individual_shocks': self.run_individual_shocks(-0.10),
            'correlation_breakdown': self.run_correlation_breakdown(),
            'worst_historical': self.run_historical_scenarios(10),
            'var_multiples': self.run_var_multiple_scenarios([1.5, 2.0, 3.0])
        }
        
        return report
    
    def create_summary_table(self, stress_results: Dict) -> pd.DataFrame:
        """
        Create summary table from stress test results.
        
        Args:
            stress_results (Dict): Results from run_standard_scenarios() or similar
            
        Returns:
            pd.DataFrame: Summary table
        """
        summary_data = []
        
        for scenario_name, result in stress_results.items():
            if 'portfolio_loss' in result:
                summary_data.append({
                    'Scenario': scenario_name,
                    'Portfolio Loss': result['portfolio_loss'],
                    'Loss (%)': result['portfolio_loss_pct'],
                    'Portfolio Return': result['portfolio_return']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Portfolio Loss', ascending=False)
        
        return summary_df
    
    def compare_to_var(self, stress_results: Dict, var_99: float) -> pd.DataFrame:
        """
        Compare stress losses to VaR estimate.
        
        Args:
            stress_results (Dict): Stress test results
            var_99 (float): 99% VaR estimate
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for scenario_name, result in stress_results.items():
            if 'portfolio_loss' in result:
                loss = result['portfolio_loss']
                ratio = loss / var_99 if var_99 > 0 else 0
                exceeds = loss > var_99
                
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Stress Loss': loss,
                    'VaR (99%)': var_99,
                    'Loss / VaR': ratio,
                    'Exceeds VaR': exceeds
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Loss / VaR', ascending=False)
        
        return comparison_df


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
    
    # Initialize stress tester
    stress_tester = StressTester(returns, weights)
    
    # Run standard scenarios
    print("\n=== Standard Stress Scenarios ===")
    standard_results = stress_tester.run_standard_scenarios()
    summary = stress_tester.create_summary_table(standard_results)
    print(summary)
    
    # Individual shocks
    print("\n=== Individual Asset Shocks ===")
    individual_results = stress_tester.run_individual_shocks(-0.10)
    individual_summary = stress_tester.create_summary_table(individual_results)
    print(individual_summary)
    
    # Correlation breakdown
    print("\n=== Correlation Breakdown ===")
    corr_breakdown = stress_tester.run_correlation_breakdown()
    print(f"Volatility increase: {corr_breakdown['volatility_increase_pct']:.2f}%")
    
    # Worst historical scenarios
    print("\n=== Worst Historical Scenarios ===")
    print(stress_tester.run_historical_scenarios(5))
