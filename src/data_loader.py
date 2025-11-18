"""
Data Loader Module
==================
Handles data ingestion, preprocessing, and validation for VaR analysis.

Author: Risk Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader class for market price data.
    
    Handles loading, validation, and preprocessing of historical price data
    for VaR model calculations.
    
    Attributes:
        filepath (str): Path to the CSV file containing price data
        prices (pd.DataFrame): Raw price data
        returns (pd.DataFrame): Calculated returns
    """
    
    def __init__(self, filepath: str):
        """
        Initialize DataLoader with file path.
        
        Args:
            filepath (str): Path to CSV file containing historical prices
        """
        self.filepath = filepath
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        
    def load_and_preprocess(self, generate_if_missing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess price data.
        
        Args:
            generate_if_missing (bool): If True, generates synthetic data if file not found
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (prices, returns)
        """
        try:
            # Check if file exists
            if not Path(self.filepath).exists():
                if generate_if_missing:
                    logger.warning(f"File {self.filepath} not found. Generating synthetic data...")
                    self._generate_synthetic_data()
                else:
                    raise FileNotFoundError(f"Data file not found: {self.filepath}")
            
            # Load data
            logger.info(f"Loading data from {self.filepath}")
            self.prices = pd.read_csv(self.filepath, index_col=0, parse_dates=True)
            
            # Validate data
            self._validate_data()
            
            # Calculate returns
            self.returns = self._calculate_returns()
            
            # Remove any remaining NaN values
            self.returns = self.returns.dropna()
            
            logger.info(f"Data loaded successfully: {self.returns.shape[0]} observations, {self.returns.shape[1]} assets")
            logger.info(f"Date range: {self.returns.index[0]} to {self.returns.index[-1]}")
            
            return self.prices, self.returns
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self) -> None:
        """Validate loaded price data."""
        if self.prices is None:
            raise ValueError("No price data loaded")
        
        # Check for missing values
        missing_pct = (self.prices.isna().sum() / len(self.prices)) * 100
        if missing_pct.max() > 5:
            logger.warning(f"High percentage of missing values detected: {missing_pct.max():.2f}%")
        
        # Check for non-positive prices
        if (self.prices <= 0).any().any():
            logger.warning("Non-positive prices detected. Forward-filling missing values...")
            self.prices = self.prices.replace(0, np.nan).fillna(method='ffill')
        
        # Check for outliers (>50% daily change)
        daily_changes = self.prices.pct_change()
        outliers = (daily_changes.abs() > 0.5).sum()
        if outliers.sum() > 0:
            logger.warning(f"Potential outliers detected: {outliers.sum()} observations")
    
    def _calculate_returns(self) -> pd.DataFrame:
        """
        Calculate log returns from prices.
        
        Returns:
            pd.DataFrame: Log returns
        """
        logger.info("Calculating log returns...")
        returns = np.log(self.prices / self.prices.shift(1))
        return returns.dropna()
    
    def _generate_synthetic_data(self, n_assets: int = 5, n_days: int = 756) -> None:
        """
        Generate synthetic price data for testing (3 years of daily data).
        
        Args:
            n_assets (int): Number of assets to generate
            n_days (int): Number of trading days (252 * 3 = 756)
        """
        logger.info(f"Generating synthetic data: {n_assets} assets, {n_days} days")
        
        # Create date range
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate correlated returns
        # Define correlation matrix
        correlation = np.array([
            [1.00, 0.65, 0.45, 0.30, 0.25],
            [0.65, 1.00, 0.55, 0.35, 0.30],
            [0.45, 0.55, 1.00, 0.40, 0.35],
            [0.30, 0.35, 0.40, 1.00, 0.45],
            [0.25, 0.30, 0.35, 0.45, 1.00]
        ])
        
        # Asset parameters (annual)
        annual_returns = [0.08, 0.10, 0.12, 0.09, 0.11]  # 8-12% annual
        annual_vols = [0.15, 0.20, 0.25, 0.18, 0.22]      # 15-25% annual vol
        
        # Convert to daily
        daily_returns = [r / 252 for r in annual_returns]
        daily_vols = [v / np.sqrt(252) for v in annual_vols]
        
        # Generate correlated random returns
        cov_matrix = np.outer(daily_vols, daily_vols) * correlation
        returns = np.random.multivariate_normal(daily_returns, cov_matrix, n_days)
        
        # Convert returns to prices (start at 100)
        prices = np.zeros((n_days, n_assets))
        prices[0] = 100
        
        for i in range(1, n_days):
            prices[i] = prices[i-1] * np.exp(returns[i])
        
        # Create DataFrame
        asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
        self.prices = pd.DataFrame(prices, index=dates, columns=asset_names)
        
        # Save to file
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        self.prices.to_csv(self.filepath)
        logger.info(f"Synthetic data saved to {self.filepath}")
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics for returns.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        if self.returns is None:
            raise ValueError("No returns data available. Call load_and_preprocess() first.")
        
        stats = pd.DataFrame({
            'Mean (Daily)': self.returns.mean(),
            'Std Dev (Daily)': self.returns.std(),
            'Mean (Annual)': self.returns.mean() * 252,
            'Vol (Annual)': self.returns.std() * np.sqrt(252),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis(),
            'Min': self.returns.min(),
            'Max': self.returns.max()
        })
        
        return stats
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns.
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if self.returns is None:
            raise ValueError("No returns data available. Call load_and_preprocess() first.")
        
        return self.returns.corr()


# Example usage
if __name__ == "__main__":
    loader = DataLoader('data/sample_prices.csv')
    prices, returns = loader.load_and_preprocess()
    
    print("\n=== Summary Statistics ===")
    print(loader.get_summary_statistics())
    
    print("\n=== Correlation Matrix ===")
    print(loader.get_correlation_matrix())
