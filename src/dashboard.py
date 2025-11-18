"""
Dashboard Module
================
Visualization and monitoring for VaR analysis.

Author: Risk Analytics Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class RiskDashboard:
    """
    Risk monitoring dashboard with comprehensive visualizations.
    
    Creates publication-quality charts for VaR analysis, backtesting,
    and stress testing results.
    
    Attributes:
        output_dir (str): Directory for saving charts
    """
    
    def __init__(self, output_dir: str = 'reports/charts'):
        """
        Initialize dashboard.
        
        Args:
            output_dir (str): Directory to save generated charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dashboard initialized: output directory = {output_dir}")
    
    def plot_return_distribution(self, returns: pd.Series, save_path: str = None) -> None:
        """
        Plot return distribution with VaR markers.
        
        Args:
            returns (pd.Series): Portfolio returns
            save_path (str): Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add VaR lines
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR ({var_95*100:.2f}%)')
        ax1.axvline(var_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR ({var_99*100:.2f}%)')
        
        # Fit normal distribution
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        from scipy.stats import norm
        ax1.plot(x, norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal Distribution')
        
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Density')
        ax1.set_title('Portfolio Return Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved return distribution plot to {save_path}")
        
        plt.close()
    
    def plot_var_comparison(self, var_results: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot VaR comparison across confidence levels and methods.
        
        Args:
            var_results (pd.DataFrame): VaR results from multiple methods
            save_path (str): Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Assume var_results has 'Method', 'Confidence', 'VaR' columns
        if 'Method' in var_results.columns:
            methods = var_results['Method'].unique()
            x = np.arange(len(var_results['Confidence'].unique()))
            width = 0.35
            
            for i, method in enumerate(methods):
                data = var_results[var_results['Method'] == method]
                ax.bar(x + i*width, data['VaR (Currency)'], width, label=method)
        else:
            # Simple bar chart
            ax.bar(range(len(var_results)), var_results['VaR (Currency)'], color='steelblue')
            ax.set_xticks(range(len(var_results)))
            ax.set_xticklabels(var_results['Confidence'])
        
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('VaR (Currency Units)')
        ax.set_title('Value-at-Risk Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved VaR comparison plot to {save_path}")
        
        plt.close()
    
    def plot_backtest_results(self, backtest_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot backtesting results: P&L vs VaR overlay.
        
        Args:
            backtest_df (pd.DataFrame): Backtest results with Date, VaR, Actual_PnL, Exception
            save_path (str): Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: P&L vs VaR
        ax1.plot(backtest_df.index, backtest_df['Actual_PnL'], label='Actual P&L', 
                color='steelblue', linewidth=1, alpha=0.7)
        ax1.plot(backtest_df.index, backtest_df['VaR'], label='VaR Estimate', 
                color='red', linestyle='--', linewidth=2)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        # Highlight exceptions
        exceptions = backtest_df[backtest_df['Exception']]
        if len(exceptions) > 0:
            ax1.scatter(exceptions.index, exceptions['Actual_PnL'], 
                       color='red', s=100, marker='x', linewidths=3, 
                       label=f'Exceptions ({len(exceptions)})', zorder=5)
        
        ax1.set_ylabel('P&L (Currency Units)')
        ax1.set_title('Backtesting: Actual P&L vs VaR')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Exception timeline
        exception_series = backtest_df['Exception'].astype(int)
        ax2.fill_between(backtest_df.index, 0, exception_series, 
                         where=exception_series>0, color='red', alpha=0.5, label='Exception Days')
        ax2.set_ylabel('Exception Indicator')
        ax2.set_xlabel('Date')
        ax2.set_title('Exception Timeline')
        ax2.set_ylim(0, 1.2)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved backtest plot to {save_path}")
        
        plt.close()
    
    def plot_traffic_light(self, n_exceptions: int, confidence: float, 
                          n_observations: int, save_path: str = None) -> None:
        """
        Plot Basel traffic-light classification.
        
        Args:
            n_exceptions (int): Number of exceptions
            confidence (float): Confidence level
            n_observations (int): Number of observations
            save_path (str): Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define zones (scaled to observation period)
        scale_factor = n_observations / 250
        
        if confidence == 0.99:
            green_max = 4 * scale_factor
            yellow_max = 9 * scale_factor
        elif confidence == 0.95:
            green_max = 13 * scale_factor
            yellow_max = 19 * scale_factor
        else:
            expected = n_observations * (1 - confidence)
            green_max = expected + 2 * np.sqrt(expected)
            yellow_max = expected + 4 * np.sqrt(expected)
        
        # Create zones
        zones = [
            ('GREEN', 0, green_max, 'green', 0.3),
            ('YELLOW', green_max, yellow_max, 'yellow', 0.3),
            ('RED', yellow_max, n_observations*0.2, 'red', 0.3)
        ]
        
        for zone_name, start, end, color, alpha in zones:
            ax.axhspan(start, end, alpha=alpha, color=color, label=f'{zone_name} Zone')
        
        # Plot actual exceptions
        ax.axhline(n_exceptions, color='black', linestyle='--', linewidth=3, 
                  label=f'Actual Exceptions: {n_exceptions}')
        
        # Expected exceptions
        expected = n_observations * (1 - confidence)
        ax.axhline(expected, color='blue', linestyle=':', linewidth=2, 
                  label=f'Expected: {expected:.1f}')
        
        ax.set_ylabel('Number of Exceptions')
        ax.set_title(f'Basel Traffic-Light Classification ({confidence*100:.0f}% VaR, {n_observations} days)')
        ax.legend(loc='best')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved traffic-light plot to {save_path}")
        
        plt.close()
    
    def plot_rolling_volatility(self, returns: pd.Series, window: int = 30, save_path: str = None) -> None:
        """
        Plot rolling volatility over time.
        
        Args:
            returns (pd.Series): Portfolio returns
            window (int): Rolling window size
            save_path (str): Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Calculate rolling volatility (annualized)
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        ax.plot(rolling_vol.index, rolling_vol * 100, color='steelblue', linewidth=2)
        ax.fill_between(rolling_vol.index, 0, rolling_vol * 100, alpha=0.3, color='steelblue')
        
        # Add mean line
        mean_vol = rolling_vol.mean() * 100
        ax.axhline(mean_vol, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_vol:.2f}%')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility (%)')
        ax.set_title(f'Rolling {window}-Day Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved rolling volatility plot to {save_path}")
        
        plt.close()
    
    def plot_stress_test_results(self, stress_summary: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot stress test scenario results.
        
        Args:
            stress_summary (pd.DataFrame): Stress test summary with Scenario, Loss columns
            save_path (str): Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by loss
        stress_summary = stress_summary.sort_values('Portfolio Loss', ascending=True)
        
        # Color code based on sign
        colors = ['green' if x < 0 else 'red' for x in stress_summary['Portfolio Loss']]
        
        y_pos = np.arange(len(stress_summary))
        ax.barh(y_pos, stress_summary['Portfolio Loss'], color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stress_summary['Scenario'])
        ax.set_xlabel('Portfolio Loss (Currency Units)')
        ax.set_title('Stress Test Results')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved stress test plot to {save_path}")
        
        plt.close()
    
    def plot_component_var(self, component_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot Component VaR contributions.
        
        Args:
            component_df (pd.DataFrame): Component VaR results
            save_path (str): Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Absolute contributions
        ax1.bar(component_df['Asset'], component_df['Component VaR'], 
               color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Asset')
        ax1.set_ylabel('Component VaR (Currency Units)')
        ax1.set_title('Absolute VaR Contribution by Asset')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Percentage contributions
        colors = plt.cm.Set3(np.linspace(0, 1, len(component_df)))
        ax2.pie(component_df['Contribution (%)'], labels=component_df['Asset'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Percentage VaR Contribution by Asset')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved component VaR plot to {save_path}")
        
        plt.close()
    
    def plot_correlation_matrix(self, returns: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            returns (pd.DataFrame): Asset returns
            save_path (str): Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = returns.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Asset Return Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to {save_path}")
        
        plt.close()
    
    def create_summary_dashboard(self, returns: pd.Series, var_results: Dict, 
                                backtest_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Create comprehensive summary dashboard.
        
        Args:
            returns (pd.Series): Portfolio returns
            var_results (Dict): VaR calculation results
            backtest_df (pd.DataFrame): Backtesting results
            save_path (str): Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Return distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, label='95% VaR')
        ax1.axvline(var_99, color='red', linestyle='--', linewidth=2, label='99% VaR')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Density')
        ax1.set_title('Return Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling volatility
        ax2 = fig.add_subplot(gs[0, 1])
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
        ax2.plot(rolling_vol.index, rolling_vol, color='steelblue', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Annualized Volatility (%)')
        ax2.set_title('30-Day Rolling Volatility')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Backtesting
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(backtest_df.index, backtest_df['Actual_PnL'], label='Actual P&L', 
                color='steelblue', linewidth=1, alpha=0.7)
        ax3.plot(backtest_df.index, backtest_df['VaR'], label='VaR', 
                color='red', linestyle='--', linewidth=2)
        exceptions = backtest_df[backtest_df['Exception']]
        if len(exceptions) > 0:
            ax3.scatter(exceptions.index, exceptions['Actual_PnL'], 
                       color='red', s=50, marker='x', linewidths=2, label='Exceptions')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('P&L')
        ax3.set_title('Backtesting Results')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_text = f"""
        Portfolio Statistics:
        - Mean Daily Return: {returns.mean()*100:.4f}%
        - Daily Volatility: {returns.std()*100:.4f}%
        - Annualized Return: {returns.mean()*252*100:.2f}%
        - Annualized Volatility: {returns.std()*np.sqrt(252)*100:.2f}%
        - Sharpe Ratio: {(returns.mean()*252)/(returns.std()*np.sqrt(252)):.2f}
        
        VaR Estimates:
        - 95% VaR: {var_results.get('var_95', 'N/A')}
        - 99% VaR: {var_results.get('var_99', 'N/A')}
        
        Backtesting (99% VaR):
        - Exceptions: {len(exceptions)}
        - Exception Rate: {(len(exceptions)/len(backtest_df))*100:.2f}%
        - Traffic Light: {var_results.get('traffic_light', 'N/A')}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
                verticalalignment='center')
        
        fig.suptitle('VaR Risk Management Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary dashboard to {save_path}")
        
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    returns = pd.Series(
        np.random.randn(500) * 0.015,
        index=dates
    )
    
    # Initialize dashboard
    dashboard = RiskDashboard('reports/charts')
    
    # Generate plots
    dashboard.plot_return_distribution(returns, 'return_distribution.png')
    dashboard.plot_rolling_volatility(returns, window=30, save_path='rolling_volatility.png')
    
    logger.info("Dashboard examples generated successfully")
