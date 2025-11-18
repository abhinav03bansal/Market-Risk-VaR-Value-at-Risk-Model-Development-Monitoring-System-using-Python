"""
Interactive VaR Analysis Notebook
==================================

This notebook provides interactive analysis of the VaR model.
Copy this structure into a Jupyter notebook (.ipynb) for execution.

"""

# Cell 1: Setup and Imports
# ===========================
import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import DataLoader
from src.var_model import VaRModel
from src.monte_carlo import MonteCarloVaR
from src.backtesting import Backtester
from src.stress_test import StressTester
from src.dashboard import RiskDashboard

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 4)

plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline

print("✓ Imports successful")

# Cell 2: Load Data
# ==================
print("Loading data...")

loader = DataLoader('../data/sample_prices.csv')
prices, returns = loader.load_and_preprocess()

print(f"Data loaded: {len(returns)} observations, {returns.shape[1]} assets")
print(f"Date range: {returns.index[0]} to {returns.index[-1]}")

# Display summary statistics
loader.get_summary_statistics()


# Cell 3: Visualize Returns
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Individual asset returns
for i, asset in enumerate(returns.columns[:4]):
    ax = axes[i//2, i%2]
    returns[asset].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
    ax.set_title(f'{asset} Return Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1)
plt.title('Asset Return Correlation Matrix')
plt.show()


# Cell 4: Portfolio Configuration
# =================================
# Define portfolio
PORTFOLIO_VALUE = 10_000_000  # $10 million
WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal-weighted

print(f"Portfolio Value: ${PORTFOLIO_VALUE:,}")
print(f"Asset Weights: {WEIGHTS}")

# Calculate portfolio returns
portfolio_returns = (returns * WEIGHTS).sum(axis=1)

# Portfolio statistics
print(f"\nPortfolio Statistics:")
print(f"Mean Daily Return: {portfolio_returns.mean()*100:.4f}%")
print(f"Daily Volatility: {portfolio_returns.std()*100:.4f}%")
print(f"Annualized Return: {portfolio_returns.mean()*252*100:.2f}%")
print(f"Annualized Volatility: {portfolio_returns.std()*np.sqrt(252)*100:.2f}%")
print(f"Sharpe Ratio: {(portfolio_returns.mean()*252)/(portfolio_returns.std()*np.sqrt(252)):.2f}")


# Cell 5: Calculate Historical VaR
# ==================================
print("Calculating Historical Simulation VaR...")

var_model = VaRModel(returns, WEIGHTS, PORTFOLIO_VALUE)

# Multiple confidence levels
var_table = var_model.calculate_multiple_vars([0.90, 0.95, 0.99, 0.999])
print("\nHistorical Simulation VaR:")
display(var_table)

# Visualize
plt.figure(figsize=(12, 6))
plt.hist(portfolio_returns, bins=100, density=True, alpha=0.7, 
         color='steelblue', edgecolor='black')

# Add VaR lines
for conf in [0.95, 0.99]:
    var_return = np.percentile(portfolio_returns, (1-conf)*100)
    plt.axvline(var_return, color='red' if conf==0.99 else 'orange', 
                linestyle='--', linewidth=2, 
                label=f'{conf*100:.0f}% VaR ({var_return*100:.2f}%)')

plt.xlabel('Portfolio Return')
plt.ylabel('Density')
plt.title('Portfolio Return Distribution with VaR Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Cell 6: Component VaR Analysis
# ================================
print("Calculating Component VaR...")

component_var = var_model.component_var(0.99)
display(component_var)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Bar chart
ax1.bar(component_var['Asset'], component_var['Component VaR'], 
       color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Asset')
ax1.set_ylabel('Component VaR ($)')
ax1.set_title('Absolute VaR Contribution by Asset')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# Pie chart
ax2.pie(component_var['Contribution (%)'], labels=component_var['Asset'], 
       autopct='%1.1f%%', startangle=90)
ax2.set_title('Percentage VaR Contribution')

plt.tight_layout()
plt.show()


# Cell 7: Monte Carlo VaR
# =========================
print("Running Monte Carlo Simulation...")

mc_var = MonteCarloVaR(returns, WEIGHTS, PORTFOLIO_VALUE)

# Calculate VaR
mc_table = mc_var.calculate_multiple_vars([0.95, 0.99], n_simulations=10000)
print("\nMonte Carlo VaR:")
display(mc_table)

# Sensitivity to number of simulations
sensitivity = mc_var.sensitivity_analysis(confidence=0.99, 
                                         n_simulations_list=[1000, 5000, 10000, 25000, 50000])
print("\nSensitivity to Simulation Count:")
display(sensitivity)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sensitivity['Simulations'], sensitivity['VaR (Currency)'], 
         marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Simulations')
plt.ylabel('VaR (Currency)')
plt.title('Monte Carlo VaR Convergence')
plt.grid(True, alpha=0.3)
plt.show()


# Cell 8: Backtesting
# =====================
print("Running backtesting...")

backtester = Backtester(returns, WEIGHTS, PORTFOLIO_VALUE)

# Run backtest
backtest_results = backtester.run_backtest(window=250, confidence=0.99)

summary = backtest_results['summary']
print(f"\nBacktesting Results:")
print(f"Observations: {summary['n_observations']}")
print(f"Exceptions: {summary['n_exceptions']}")
print(f"Exception Rate: {summary['exception_rate']*100:.2f}%")
print(f"Traffic Light: {summary['traffic_light']}")

# Visualize
backtest_df = backtest_results['results_df']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# P&L vs VaR
ax1.plot(backtest_df.index, backtest_df['Actual_PnL'], 
        label='Actual P&L', color='steelblue', linewidth=1, alpha=0.7)
ax1.plot(backtest_df.index, backtest_df['VaR'], 
        label='VaR', color='red', linestyle='--', linewidth=2)

exceptions = backtest_df[backtest_df['Exception']]
if len(exceptions) > 0:
    ax1.scatter(exceptions.index, exceptions['Actual_PnL'], 
               color='red', s=100, marker='x', linewidths=3, 
               label=f'Exceptions ({len(exceptions)})', zorder=5)

ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax1.set_ylabel('P&L ($)')
ax1.set_title('Backtesting: Actual P&L vs VaR')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Exception timeline
exception_series = backtest_df['Exception'].astype(int)
ax2.fill_between(backtest_df.index, 0, exception_series, 
                where=exception_series>0, color='red', alpha=0.5)
ax2.set_ylabel('Exception')
ax2.set_xlabel('Date')
ax2.set_title('Exception Timeline')
ax2.set_ylim(0, 1.2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Cell 9: Statistical Tests
# ===========================
print("Running statistical tests...")

# Kupiec test
kupiec = backtester.kupiec_test(backtest_results)
print("\nKupiec POF Test:")
print(f"LR Statistic: {kupiec['lr_statistic']:.4f}")
print(f"P-value: {kupiec['p_value']:.4f}")
print(f"Result: {kupiec['interpretation']}")

# Christoffersen test
chris = backtester.christoffersen_test(backtest_results)
print("\nChristoffersen Independence Test:")
print(f"LR Statistic: {chris['lr_independence']:.4f}")
print(f"P-value: {chris['p_value']:.4f}")
print(f"Result: {chris['interpretation']}")

# Exception analysis
if summary['n_exceptions'] > 0:
    exception_df = backtester.get_exception_analysis(backtest_results)
    print("\nException Details:")
    display(exception_df)


# Cell 10: Stress Testing
# =========================
print("Running stress tests...")

stress_tester = StressTester(returns, WEIGHTS, PORTFOLIO_VALUE)

# Standard scenarios
standard_scenarios = stress_tester.run_standard_scenarios()
standard_summary = stress_tester.create_summary_table(standard_scenarios)

print("\nStandard Stress Scenarios:")
display(standard_summary)

# Visualize
plt.figure(figsize=(12, 8))
standard_summary_sorted = standard_summary.sort_values('Portfolio Loss')
colors = ['green' if x < 0 else 'red' for x in standard_summary_sorted['Portfolio Loss']]

plt.barh(range(len(standard_summary_sorted)), standard_summary_sorted['Portfolio Loss'], 
        color=colors, alpha=0.7, edgecolor='black')
plt.yticks(range(len(standard_summary_sorted)), standard_summary_sorted['Scenario'])
plt.xlabel('Portfolio Loss ($)')
plt.title('Stress Test Results')
plt.axvline(0, color='black', linestyle='-', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()


# Cell 11: Worst Historical Scenarios
# =====================================
print("Identifying worst historical scenarios...")

worst_scenarios = stress_tester.run_historical_scenarios(10)
print("\nTop 10 Worst Historical Days:")
display(worst_scenarios)

# Visualize on timeline
plt.figure(figsize=(15, 6))
plt.plot(portfolio_returns.index, portfolio_returns * PORTFOLIO_VALUE * -1, 
        color='steelblue', alpha=0.5, linewidth=1)

for _, row in worst_scenarios.iterrows():
    plt.scatter(row['Date'], row['Loss'], color='red', s=100, marker='x', linewidths=3)

plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Date')
plt.ylabel('P&L ($)')
plt.title('Portfolio P&L with Worst Historical Losses Highlighted')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Cell 12: Summary Dashboard
# ============================
print("Creating summary dashboard...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Return distribution
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(portfolio_returns, bins=100, density=True, alpha=0.7, 
        color='steelblue', edgecolor='black')
var_95 = np.percentile(portfolio_returns, 5)
var_99 = np.percentile(portfolio_returns, 1)
ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, label='95% VaR')
ax1.axvline(var_99, color='red', linestyle='--', linewidth=2, label='99% VaR')
ax1.set_xlabel('Return')
ax1.set_ylabel('Density')
ax1.set_title('Portfolio Return Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. VaR comparison
ax2 = fig.add_subplot(gs[0, 2])
var_data = var_model.calculate_multiple_vars([0.90, 0.95, 0.99])
ax2.bar(range(len(var_data)), var_data['VaR (Currency)'], color='steelblue', alpha=0.7)
ax2.set_xticks(range(len(var_data)))
ax2.set_xticklabels(var_data['Confidence'])
ax2.set_ylabel('VaR ($)')
ax2.set_title('VaR by Confidence Level')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Backtesting
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(backtest_df.index, backtest_df['Actual_PnL'], 
        label='Actual P&L', color='steelblue', linewidth=1, alpha=0.7)
ax3.plot(backtest_df.index, backtest_df['VaR'], 
        label='VaR', color='red', linestyle='--', linewidth=2)
exceptions = backtest_df[backtest_df['Exception']]
if len(exceptions) > 0:
    ax3.scatter(exceptions.index, exceptions['Actual_PnL'], 
               color='red', s=50, marker='x', linewidths=2, label='Exceptions')
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Date')
ax3.set_ylabel('P&L ($)')
ax3.set_title(f'Backtesting Results - {summary["traffic_light"]} Zone')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Rolling volatility
ax4 = fig.add_subplot(gs[2, :2])
rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252) * 100
ax4.plot(rolling_vol.index, rolling_vol, color='steelblue', linewidth=2)
ax4.fill_between(rolling_vol.index, 0, rolling_vol, alpha=0.3, color='steelblue')
ax4.set_xlabel('Date')
ax4.set_ylabel('Annualized Volatility (%)')
ax4.set_title('30-Day Rolling Volatility')
ax4.grid(True, alpha=0.3)

# 5. Traffic light indicator
ax5 = fig.add_subplot(gs[2, 2])
traffic_light_color = {'GREEN': 'green', 'YELLOW': 'yellow', 'RED': 'red'}
color = traffic_light_color.get(summary['traffic_light'], 'gray')
circle = plt.Circle((0.5, 0.5), 0.3, color=color, alpha=0.7)
ax5.add_patch(circle)
ax5.text(0.5, 0.5, summary['traffic_light'], 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')
ax5.set_title('Model Status')

fig.suptitle('VaR Risk Management Dashboard', fontsize=18, fontweight='bold', y=0.995)
plt.show()

print("\n✓ Analysis complete!")


# Cell 13: Export Results
# =========================
print("Exporting results...")

# Save key metrics to CSV
results_summary = pd.DataFrame({
    'Metric': [
        'Portfolio Value',
        'VaR 95%',
        'VaR 99%',
        'Backtest Exceptions',
        'Exception Rate',
        'Traffic Light'
    ],
    'Value': [
        f"${PORTFOLIO_VALUE:,}",
        f"${var_model.historical_var(0.95)['var_absolute']:,.2f}",
        f"${var_model.historical_var(0.99)['var_absolute']:,.2f}",
        summary['n_exceptions'],
        f"{summary['exception_rate']*100:.2f}%",
        summary['traffic_light']
    ]
})

results_summary.to_csv('var_analysis_summary.csv', index=False)
print("✓ Results saved to: var_analysis_summary.csv")

print("\nNotebook execution complete!")
