"""
Main Execution Script
=====================
Complete VaR analysis pipeline execution.

This script runs the complete VaR model development and monitoring workflow:
1. Data loading and preprocessing
2. VaR calculation (Historical Simulation & Monte Carlo)
3. Backtesting and model validation
4. Stress testing
5. Report generation (Excel & PowerPoint)
6. Visualization dashboard

Author: Risk Analytics Team
Date: November 2025
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.data_loader import DataLoader
from src.var_model import VaRModel
from src.monte_carlo import MonteCarloVaR
from src.backtesting import Backtester
from src.stress_test import StressTester
from src.dashboard import RiskDashboard
from src.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('var_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_header(text: str) -> None:
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    
    print_header("MARKET RISK VaR MODEL DEVELOPMENT & MONITORING SYSTEM")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    DATA_FILE = 'data/sample_prices.csv'
    PORTFOLIO_VALUE = 10_000_000  # $10 million
    WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal-weighted portfolio
    CONFIDENCE_LEVELS = [0.95, 0.99]
    BACKTEST_WINDOW = 250
    MC_SIMULATIONS = 10000
    
    try:
        # =================================================================
        # STEP 1: DATA LOADING
        # =================================================================
        print_header("STEP 1: DATA LOADING & PREPROCESSING")
        
        loader = DataLoader(DATA_FILE)
        prices, returns = loader.load_and_preprocess()
        
        print(f"Data loaded successfully:")
        print(f"  - {len(returns)} observations")
        print(f"  - {returns.shape[1]} assets")
        print(f"  - Date range: {returns.index[0]} to {returns.index[-1]}")
        
        # Summary statistics
        print("\nSummary Statistics:")
        summary_stats = loader.get_summary_statistics()
        print(summary_stats.to_string())
        
        print("\nCorrelation Matrix:")
        corr_matrix = loader.get_correlation_matrix()
        print(corr_matrix.to_string())
        
        # =================================================================
        # STEP 2: VaR CALCULATION - HISTORICAL SIMULATION
        # =================================================================
        print_header("STEP 2: HISTORICAL SIMULATION VaR")
        
        var_model = VaRModel(returns, WEIGHTS, PORTFOLIO_VALUE)
        
        print("Calculating VaR at multiple confidence levels...")
        var_table = var_model.calculate_multiple_vars(CONFIDENCE_LEVELS)
        print("\n" + var_table.to_string(index=False))
        
        # Store results
        var_results = {
            'var_95': var_model.historical_var(0.95)['var_absolute'],
            'var_99': var_model.historical_var(0.99)['var_absolute']
        }
        
        # Component VaR
        print("\nComponent VaR Analysis (99% confidence):")
        component_var = var_model.component_var(0.99)
        print(component_var.to_string(index=False))
        
        # Expected Shortfall
        print("\nExpected Shortfall:")
        es_95 = var_model.calculate_expected_shortfall(0.95)
        es_99 = var_model.calculate_expected_shortfall(0.99)
        print(f"  95% ES: ${es_95['es_absolute']:,.2f}")
        print(f"  99% ES: ${es_99['es_absolute']:,.2f}")
        
        # Portfolio statistics
        print("\nPortfolio Statistics:")
        portfolio_stats = var_model.get_portfolio_statistics()
        for key, value in portfolio_stats.items():
            if isinstance(value, float):
                if 'Ratio' in key or 'Return' in key or 'Volatility' in key:
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value}")
        
        # =================================================================
        # STEP 3: VaR CALCULATION - MONTE CARLO
        # =================================================================
        print_header("STEP 3: MONTE CARLO VaR")
        
        mc_var = MonteCarloVaR(returns, WEIGHTS, PORTFOLIO_VALUE)
        
        print(f"Running {MC_SIMULATIONS:,} Monte Carlo simulations...")
        mc_table = mc_var.calculate_multiple_vars(CONFIDENCE_LEVELS, MC_SIMULATIONS)
        print("\n" + mc_table.to_string(index=False))
        
        # Compare distributions
        print("\nDistribution Comparison (Normal vs Student's t):")
        dist_comparison = mc_var.compare_distributions(0.99, MC_SIMULATIONS)
        print(f"  Normal VaR: ${dist_comparison['normal_var']:,.2f}")
        print(f"  t-distribution VaR: ${dist_comparison['t_distribution_var']:,.2f}")
        print(f"  Difference: ${dist_comparison['difference']:,.2f} ({dist_comparison['difference_pct']:.2f}%)")
        
        # =================================================================
        # STEP 4: BACKTESTING
        # =================================================================
        print_header("STEP 4: BACKTESTING & MODEL VALIDATION")
        
        backtester = Backtester(returns, WEIGHTS, PORTFOLIO_VALUE)
        
        print(f"Running backtest with {BACKTEST_WINDOW}-day rolling window...")
        backtest_results = backtester.run_backtest(
            window=BACKTEST_WINDOW,
            confidence=0.99
        )
        
        summary = backtest_results['summary']
        print("\nBacktesting Results (99% VaR):")
        print(f"  Test Period: {summary['start_date']} to {summary['end_date']}")
        print(f"  Observations: {summary['n_observations']}")
        print(f"  Exceptions: {summary['n_exceptions']}")
        print(f"  Exception Rate: {summary['exception_rate']*100:.2f}%")
        print(f"  Expected Rate: {summary['expected_rate']*100:.2f}%")
        print(f"  Traffic Light: {summary['traffic_light']}")
        
        if summary['traffic_light'] == 'GREEN':
            print(f"\n  ✓ Model Status: ADEQUATE")
        elif summary['traffic_light'] == 'YELLOW':
            print(f"\n  ⚠ Model Status: MONITORING REQUIRED")
        else:
            print(f"\n  ✗ Model Status: INADEQUATE - RECALIBRATION NEEDED")
        
        # Statistical tests
        print("\nStatistical Tests:")
        
        kupiec = backtester.kupiec_test(backtest_results)
        print(f"  Kupiec POF Test:")
        print(f"    LR Statistic: {kupiec['lr_statistic']:.4f}")
        print(f"    P-value: {kupiec['p_value']:.4f}")
        print(f"    Result: {kupiec['interpretation']}")
        
        chris = backtester.christoffersen_test(backtest_results)
        print(f"\n  Christoffersen Independence Test:")
        print(f"    LR Statistic: {chris['lr_independence']:.4f}")
        print(f"    P-value: {chris['p_value']:.4f}")
        print(f"    Result: {chris['interpretation']}")
        
        # Exception analysis
        if summary['n_exceptions'] > 0:
            print("\nException Details:")
            exception_df = backtester.get_exception_analysis(backtest_results)
            print(exception_df.to_string(index=False))
        
        # Store traffic light for reporting
        var_results['traffic_light'] = summary['traffic_light']
        
        # =================================================================
        # STEP 5: STRESS TESTING
        # =================================================================
        print_header("STEP 5: STRESS TESTING")
        
        stress_tester = StressTester(returns, WEIGHTS, PORTFOLIO_VALUE)
        
        print("Running standard stress scenarios...")
        standard_scenarios = stress_tester.run_standard_scenarios()
        standard_summary = stress_tester.create_summary_table(standard_scenarios)
        print("\nStandard Scenarios:")
        print(standard_summary.to_string(index=False))
        
        print("\nRunning individual asset shocks (-10%)...")
        individual_shocks = stress_tester.run_individual_shocks(-0.10)
        individual_summary = stress_tester.create_summary_table(individual_shocks)
        print(individual_summary.to_string(index=False))
        
        print("\nCorrelation Breakdown Scenario:")
        corr_breakdown = stress_tester.run_correlation_breakdown()
        print(f"  Current Volatility: {corr_breakdown['current_volatility']*100:.4f}%")
        print(f"  Stressed Volatility: {corr_breakdown['stressed_volatility']*100:.4f}%")
        print(f"  Increase: {corr_breakdown['volatility_increase_pct']:.2f}%")
        
        print("\nWorst Historical Scenarios (Top 5):")
        worst_historical = stress_tester.run_historical_scenarios(5)
        print(worst_historical.to_string())
        
        # Comparison to VaR
        print("\nStress Loss vs VaR Comparison:")
        var_comparison = stress_tester.compare_to_var(standard_scenarios, var_results['var_99'])
        print(var_comparison.to_string(index=False))
        
        # Store stress results
        stress_results = {
            'standard_scenarios': standard_scenarios,
            'individual_shocks': individual_shocks,
            'correlation_breakdown': corr_breakdown,
            'worst_historical': worst_historical
        }
        
        # =================================================================
        # STEP 6: VISUALIZATION DASHBOARD
        # =================================================================
        print_header("STEP 6: GENERATING VISUALIZATIONS")
        
        dashboard = RiskDashboard('reports/charts')
        
        # Portfolio returns for plotting
        portfolio_returns = (returns * WEIGHTS).sum(axis=1)
        
        print("Creating visualizations...")
        
        # Return distribution
        dashboard.plot_return_distribution(portfolio_returns, 'return_distribution.png')
        print("  ✓ Return distribution plot")
        
        # VaR comparison
        dashboard.plot_var_comparison(var_table, 'var_comparison.png')
        print("  ✓ VaR comparison plot")
        
        # Backtesting results
        dashboard.plot_backtest_results(backtest_results['results_df'], 'backtest_results.png')
        print("  ✓ Backtesting plot")
        
        # Traffic light
        dashboard.plot_traffic_light(
            summary['n_exceptions'],
            0.99,
            summary['n_observations'],
            'traffic_light.png'
        )
        print("  ✓ Traffic-light classification plot")
        
        # Rolling volatility
        dashboard.plot_rolling_volatility(portfolio_returns, 30, 'rolling_volatility.png')
        print("  ✓ Rolling volatility plot")
        
        # Stress test results
        dashboard.plot_stress_test_results(standard_summary, 'stress_test_results.png')
        print("  ✓ Stress test plot")
        
        # Component VaR
        dashboard.plot_component_var(component_var, 'component_var.png')
        print("  ✓ Component VaR plot")
        
        # Correlation matrix
        dashboard.plot_correlation_matrix(returns, 'correlation_matrix.png')
        print("  ✓ Correlation matrix plot")
        
        # Summary dashboard
        dashboard.create_summary_dashboard(
            portfolio_returns,
            var_results,
            backtest_results['results_df'],
            'summary_dashboard.png'
        )
        print("  ✓ Summary dashboard")
        
        print(f"\nAll charts saved to: reports/charts/")
        
        # =================================================================
        # STEP 7: REPORT GENERATION
        # =================================================================
        print_header("STEP 7: GENERATING REPORTS")
        
        reporter = ReportGenerator(returns, WEIGHTS, PORTFOLIO_VALUE)
        
        # Excel report
        print("Generating Excel report...")
        excel_path = 'reports/VaR_Risk_Report.xlsx'
        reporter.generate_excel_report(
            excel_path,
            var_results,
            backtest_results,
            stress_results
        )
        print(f"  ✓ Excel report: {excel_path}")
        
        # PowerPoint report
        print("\nGenerating PowerPoint presentation...")
        ppt_path = 'reports/VaR_Executive_Presentation.pptx'
        reporter.generate_ppt_report(
            ppt_path,
            var_results,
            backtest_results,
            stress_results,
            'reports/charts'
        )
        print(f"  ✓ PowerPoint report: {ppt_path}")
        
        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        print_header("ANALYSIS COMPLETE")
        
        print("Summary:")
        print(f"  Portfolio Value: ${PORTFOLIO_VALUE:,.0f}")
        print(f"  Number of Assets: {len(WEIGHTS)}")
        print(f"  Analysis Period: {len(returns)} days")
        print(f"\n  95% VaR (1-day): ${var_results['var_95']:,.2f}")
        print(f"  99% VaR (1-day): ${var_results['var_99']:,.2f}")
        print(f"\n  Backtest Status: {summary['traffic_light']}")
        print(f"  Exception Rate: {summary['exception_rate']*100:.2f}%")
        
        if summary['traffic_light'] == 'GREEN':
            print(f"\n  ✓ MODEL IS ADEQUATE - Risk management framework performing well")
        elif summary['traffic_light'] == 'YELLOW':
            print(f"\n  ⚠ MONITORING REQUIRED - Increase oversight frequency")
        else:
            print(f"\n  ✗ MODEL RECALIBRATION NEEDED - Review methodology and parameters")
        
        print(f"\nAll reports generated successfully:")
        print(f"  - Excel Report: {excel_path}")
        print(f"  - PowerPoint: {ppt_path}")
        print(f"  - Charts: reports/charts/")
        print(f"  - Log File: var_analysis.log")
        
        print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(f"\n✗ ERROR: {str(e)}")
        print("Check var_analysis.log for details")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
