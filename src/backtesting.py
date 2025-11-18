"""
Backtesting Module
==================
VaR model validation and Basel traffic-light classification.

Author: Risk Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Backtester:
    """
    VaR backtesting engine with Basel traffic-light classification.
    
    Validates VaR model accuracy by comparing predicted VaR against
    actual realized portfolio losses.
    
    Attributes:
        returns (pd.DataFrame): Historical returns data
        weights (np.ndarray): Portfolio weights
        portfolio_value (float): Total portfolio value
    """
    
    # Basel traffic-light thresholds (for 250 trading days)
    BASEL_THRESHOLDS = {
        0.99: {'green': 4, 'yellow': 9},
        0.95: {'green': 13, 'yellow': 19}
    }
    
    def __init__(self, returns: pd.DataFrame, weights: List[float], portfolio_value: float = 10_000_000):
        """
        Initialize backtester.
        
        Args:
            returns (pd.DataFrame): Historical returns for each asset
            weights (List[float]): Portfolio weights
            portfolio_value (float): Total portfolio value
        """
        self.returns = returns
        self.weights = np.array(weights)
        self.portfolio_value = portfolio_value
        
        # Calculate portfolio returns
        self.portfolio_returns = (returns * self.weights).sum(axis=1)
        
        logger.info(f"Backtester initialized: {len(returns)} observations")
    
    def run_backtest(self, window: int = 250, confidence: float = 0.99, 
                    start_date: str = None) -> Dict:
        """
        Run rolling window backtest.
        
        Args:
            window (int): Rolling window size for VaR calculation
            confidence (float): Confidence level
            start_date (str): Start date for backtesting (YYYY-MM-DD)
            
        Returns:
            Dict: Comprehensive backtest results
        """
        logger.info(f"Running backtest: window={window}, confidence={confidence*100}%")
        
        # Prepare data
        if start_date:
            start_idx = self.portfolio_returns.index.get_loc(pd.Timestamp(start_date))
        else:
            start_idx = window
        
        # Initialize results storage
        dates = []
        var_estimates = []
        actual_returns = []
        actual_pnl = []
        exceptions = []
        
        # Rolling window backtest
        for i in range(start_idx, len(self.portfolio_returns)):
            # Get historical window
            hist_returns = self.portfolio_returns.iloc[i-window:i]
            
            # Calculate VaR using historical simulation
            var_percentile = 1 - confidence
            var_return = np.percentile(hist_returns, var_percentile * 100)
            var_absolute = -var_return * self.portfolio_value
            
            # Get actual return for this day
            actual_return = self.portfolio_returns.iloc[i]
            actual_loss = -actual_return * self.portfolio_value
            
            # Check if exception occurred
            is_exception = actual_return < var_return
            
            # Store results
            dates.append(self.portfolio_returns.index[i])
            var_estimates.append(var_absolute)
            actual_returns.append(actual_return)
            actual_pnl.append(actual_loss)
            exceptions.append(is_exception)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': dates,
            'VaR': var_estimates,
            'Actual_Return': actual_returns,
            'Actual_PnL': actual_pnl,
            'Exception': exceptions
        })
        results_df.set_index('Date', inplace=True)
        
        # Calculate statistics
        n_exceptions = sum(exceptions)
        n_observations = len(exceptions)
        exception_rate = n_exceptions / n_observations if n_observations > 0 else 0
        expected_exceptions = n_observations * (1 - confidence)
        
        # Traffic-light classification
        traffic_light = self._classify_traffic_light(n_exceptions, confidence, n_observations)
        
        # Exception details
        exception_dates = results_df[results_df['Exception']].index.tolist()
        exception_losses = results_df[results_df['Exception']]['Actual_PnL'].values
        
        # Calculate excess losses
        excess_losses = []
        for idx in exception_dates:
            excess = results_df.loc[idx, 'Actual_PnL'] - results_df.loc[idx, 'VaR']
            excess_losses.append(excess)
        
        # Summary statistics
        summary = {
            'window': window,
            'confidence': confidence,
            'n_observations': n_observations,
            'n_exceptions': n_exceptions,
            'expected_exceptions': expected_exceptions,
            'exception_rate': exception_rate,
            'expected_rate': 1 - confidence,
            'traffic_light': traffic_light,
            'exception_dates': exception_dates,
            'exception_losses': exception_losses,
            'excess_losses': excess_losses,
            'avg_var': np.mean(var_estimates),
            'avg_actual_loss': np.mean([x for x in actual_pnl if x > 0]),
            'max_var': np.max(var_estimates),
            'max_actual_loss': np.max(actual_pnl),
            'avg_excess': np.mean(excess_losses) if excess_losses else 0,
            'max_excess': np.max(excess_losses) if excess_losses else 0,
            'start_date': dates[0],
            'end_date': dates[-1]
        }
        
        logger.info(f"Backtest complete: {n_exceptions} exceptions ({exception_rate*100:.2f}%), Traffic Light: {traffic_light}")
        
        return {
            'summary': summary,
            'results_df': results_df
        }
    
    def _classify_traffic_light(self, n_exceptions: int, confidence: float, n_observations: int) -> str:
        """
        Classify model using Basel traffic-light approach.
        
        Args:
            n_exceptions (int): Number of exceptions
            confidence (float): Confidence level
            n_observations (int): Number of observations
            
        Returns:
            str: Traffic light color (GREEN, YELLOW, RED)
        """
        # Scale thresholds based on actual observation period
        if n_observations < 250:
            scale_factor = n_observations / 250
        else:
            scale_factor = 1.0
        
        # Get thresholds for confidence level
        if confidence in self.BASEL_THRESHOLDS:
            green_threshold = self.BASEL_THRESHOLDS[confidence]['green'] * scale_factor
            yellow_threshold = self.BASEL_THRESHOLDS[confidence]['yellow'] * scale_factor
        else:
            # Use proportional thresholds for other confidence levels
            expected = n_observations * (1 - confidence)
            green_threshold = expected + 2 * np.sqrt(expected)
            yellow_threshold = expected + 4 * np.sqrt(expected)
        
        if n_exceptions <= green_threshold:
            return 'GREEN'
        elif n_exceptions <= yellow_threshold:
            return 'YELLOW'
        else:
            return 'RED'
    
    def kupiec_test(self, backtest_results: Dict) -> Dict[str, float]:
        """
        Perform Kupiec's Proportion of Failures (POF) test.
        
        Tests whether the exception rate is statistically different from expected rate.
        
        Args:
            backtest_results (Dict): Results from run_backtest()
            
        Returns:
            Dict[str, float]: Test statistics and p-value
        """
        summary = backtest_results['summary']
        n = summary['n_observations']
        x = summary['n_exceptions']
        p = 1 - summary['confidence']
        
        # Kupiec likelihood ratio statistic
        if x == 0:
            lr = -2 * np.log((1-p)**n)
        elif x == n:
            lr = -2 * np.log(p**n)
        else:
            lr = -2 * (np.log((1-p)**(n-x) * p**x) - np.log((1-x/n)**(n-x) * (x/n)**x))
        
        # P-value from chi-square distribution (1 degree of freedom)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr, df=1)
        
        # Reject null hypothesis if p < 0.05
        reject = p_value < 0.05
        
        results = {
            'lr_statistic': lr,
            'p_value': p_value,
            'reject_null': reject,
            'interpretation': 'Model is INADEQUATE' if reject else 'Model is ADEQUATE'
        }
        
        logger.info(f"Kupiec Test: LR={lr:.4f}, p-value={p_value:.4f}, {results['interpretation']}")
        
        return results
    
    def christoffersen_test(self, backtest_results: Dict) -> Dict[str, float]:
        """
        Perform Christoffersen's independence test.
        
        Tests whether exceptions are independent (not clustered).
        
        Args:
            backtest_results (Dict): Results from run_backtest()
            
        Returns:
            Dict[str, float]: Test statistics
        """
        results_df = backtest_results['results_df']
        exceptions = results_df['Exception'].values.astype(int)
        
        # Count transitions
        n00 = np.sum((exceptions[:-1] == 0) & (exceptions[1:] == 0))
        n01 = np.sum((exceptions[:-1] == 0) & (exceptions[1:] == 1))
        n10 = np.sum((exceptions[:-1] == 1) & (exceptions[1:] == 0))
        n11 = np.sum((exceptions[:-1] == 1) & (exceptions[1:] == 1))
        
        # Transition probabilities
        if (n00 + n01) > 0:
            p01 = n01 / (n00 + n01)
        else:
            p01 = 0
            
        if (n10 + n11) > 0:
            p11 = n11 / (n10 + n11)
        else:
            p11 = 0
        
        # Likelihood ratio test
        n = len(exceptions)
        p = np.sum(exceptions) / n if n > 0 else 0
        
        if p01 == 0 or p11 == 0 or p == 0 or p == 1:
            lr_ind = 0
        else:
            lr_ind = -2 * (
                np.log((1-p)**(n00+n10) * p**(n01+n11)) -
                np.log((1-p01)**n00 * p01**n01 * (1-p11)**n10 * p11**n11)
            )
        
        # P-value
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_ind, df=1)
        reject = p_value < 0.05
        
        results = {
            'lr_independence': lr_ind,
            'p_value': p_value,
            'reject_null': reject,
            'p01': p01,
            'p11': p11,
            'interpretation': 'Exceptions are CLUSTERED' if reject else 'Exceptions are INDEPENDENT'
        }
        
        logger.info(f"Christoffersen Test: LR={lr_ind:.4f}, p-value={p_value:.4f}, {results['interpretation']}")
        
        return results
    
    def compare_multiple_windows(self, windows: List[int] = [125, 250, 500], 
                                confidence: float = 0.99) -> pd.DataFrame:
        """
        Compare backtest results across different window sizes.
        
        Args:
            windows (List[int]): List of window sizes
            confidence (float): Confidence level
            
        Returns:
            pd.DataFrame: Comparison results
        """
        logger.info("Comparing multiple window sizes...")
        
        results = []
        for window in windows:
            try:
                backtest = self.run_backtest(window=window, confidence=confidence)
                summary = backtest['summary']
                
                results.append({
                    'Window': window,
                    'Observations': summary['n_observations'],
                    'Exceptions': summary['n_exceptions'],
                    'Exception Rate': summary['exception_rate'],
                    'Expected Rate': summary['expected_rate'],
                    'Traffic Light': summary['traffic_light'],
                    'Avg VaR': summary['avg_var'],
                    'Max VaR': summary['max_var']
                })
            except Exception as e:
                logger.warning(f"Could not backtest window {window}: {str(e)}")
        
        return pd.DataFrame(results)
    
    def get_exception_analysis(self, backtest_results: Dict) -> pd.DataFrame:
        """
        Detailed analysis of exception dates.
        
        Args:
            backtest_results (Dict): Results from run_backtest()
            
        Returns:
            pd.DataFrame: Exception details
        """
        summary = backtest_results['summary']
        
        if not summary['exception_dates']:
            logger.info("No exceptions found")
            return pd.DataFrame()
        
        exception_df = pd.DataFrame({
            'Date': summary['exception_dates'],
            'Actual Loss': summary['exception_losses'],
            'Excess Loss': summary['excess_losses']
        })
        
        exception_df['Loss (%)'] = (exception_df['Actual Loss'] / self.portfolio_value) * 100
        exception_df['Excess (%)'] = (exception_df['Excess Loss'] / self.portfolio_value) * 100
        
        return exception_df


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
    
    # Initialize backtester
    backtester = Backtester(returns, weights)
    
    # Run backtest
    print("\n=== Backtesting Results ===")
    results = backtester.run_backtest(window=250, confidence=0.99)
    
    summary = results['summary']
    print(f"Observations: {summary['n_observations']}")
    print(f"Exceptions: {summary['n_exceptions']}")
    print(f"Exception Rate: {summary['exception_rate']*100:.2f}%")
    print(f"Traffic Light: {summary['traffic_light']}")
    
    # Statistical tests
    print("\n=== Kupiec Test ===")
    kupiec = backtester.kupiec_test(results)
    print(f"LR Statistic: {kupiec['lr_statistic']:.4f}")
    print(f"P-value: {kupiec['p_value']:.4f}")
    print(f"Result: {kupiec['interpretation']}")
    
    print("\n=== Christoffersen Test ===")
    chris = backtester.christoffersen_test(results)
    print(f"LR Statistic: {chris['lr_independence']:.4f}")
    print(f"P-value: {chris['p_value']:.4f}")
    print(f"Result: {chris['interpretation']}")
