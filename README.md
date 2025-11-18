# Market-Risk-VaR-Value-at-Risk-Model-Development-Monitoring-System-using-Python

# Market Risk VaR Model Development & Monitoring System

## 📊 Project Overview

This is a production-quality Market Risk Value-at-Risk (VaR) system implementing industry-standard methodologies used in major investment banks (HSBC, Barclays, JPMC, Standard Chartered). The system provides comprehensive risk measurement, backtesting, and monitoring capabilities for multi-asset portfolios.

### Key Features
- **Historical Simulation VaR** (95% & 99% confidence levels)
- **Monte Carlo Simulation VaR** with parametric assumptions
- **Comprehensive Backtesting** with Basel traffic-light classification
- **Stress Testing** with multiple shock scenarios
- **Automated Reporting** (Excel & PowerPoint outputs)
- **Real-time Monitoring Dashboard** with visualization

## 🏗️ System Architecture

```
risk_var_project/
│
├── data/
│   └── sample_prices.csv              # Historical price data (5 assets, 3 years)
│
├── src/
│   ├── data_loader.py                 # Data ingestion and preprocessing
│   ├── var_model.py                   # Historical Simulation VaR engine
│   ├── monte_carlo.py                 # Monte Carlo VaR engine
│   ├── backtesting.py                 # Backtesting & model validation
│   ├── stress_test.py                 # Stress testing framework
│   ├── dashboard.py                   # Visualization and monitoring
│   └── report_generator.py            # Automated Excel & PPT reports
│
├── notebooks/
│   └── analysis.ipynb                 # Interactive analysis notebook
│
├── reports/                           # Generated reports (Excel, PPT)
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── main.py                           # Main execution script
```

## 📐 VaR Methodology

### What is Value-at-Risk (VaR)?
VaR measures the maximum potential loss in portfolio value over a specified time horizon at a given confidence level.

**Example**: A 1-day 99% VaR of $1 million means:
- There is a 99% probability that losses will not exceed $1 million over the next day
- Or equivalently: losses will exceed $1 million only 1% of the time (1 in 100 days)

### 1. Historical Simulation VaR
**Methodology**:
- Uses actual historical returns to simulate portfolio losses
- No distributional assumptions required (non-parametric)
- Process:
  1. Calculate historical returns for each asset
  2. Apply portfolio weights to compute portfolio returns
  3. Sort returns from worst to best
  4. VaR = percentile corresponding to confidence level (5th percentile for 95% VaR)

**Advantages**: Simple, captures fat tails, actual market behavior
**Disadvantages**: Limited by historical data, assumes future like past

### 2. Monte Carlo Simulation VaR
**Methodology**:
- Generates thousands of random return scenarios based on statistical properties
- Assumes returns follow normal distribution (can be extended)
- Process:
  1. Calculate mean and covariance matrix from historical data
  2. Generate 10,000+ random return scenarios
  3. Calculate portfolio returns for each scenario
  4. VaR = percentile of simulated loss distribution

**Advantages**: Can model complex portfolios, flexible scenario generation
**Disadvantages**: Computationally intensive, depends on distributional assumptions

## 🔍 Backtesting Framework

### Purpose
Validate VaR model accuracy by comparing predicted VaR against actual realized losses.

### Process
1. Calculate daily VaR estimates
2. Compare against actual daily P&L
3. Count "exceptions" (days where losses exceed VaR)
4. Classify model performance using Basel traffic-light approach

### Basel Traffic-Light Classification

The Basel Committee defines zones based on number of exceptions in 250 trading days:

| Zone   | Exceptions at 99% VaR | Exceptions at 95% VaR | Interpretation                |
|--------|----------------------|----------------------|-------------------------------|
| GREEN  | 0-4                  | 0-13                 | Model is accurate             |
| YELLOW | 5-9                  | 14-19                | Model needs monitoring        |
| RED    | 10+                  | 20+                  | Model is inadequate, recalibrate |

**Statistical Basis**:
- At 99% confidence, we expect ~2.5 exceptions per year (250 days × 1% = 2.5)
- At 95% confidence, we expect ~12.5 exceptions per year (250 days × 5% = 12.5)
- Green zone allows for statistical variation
- Red zone suggests systematic underestimation of risk

### Key Metrics
- **Exception Rate**: % of days with losses exceeding VaR
- **Coverage Ratio**: Actual exceptions / Expected exceptions
- **Average Excess**: Mean loss on exception days beyond VaR
- **Maximum Excess**: Worst breach of VaR

## ⚡ Stress Testing

### Scenarios Implemented
1. **Market Crash**: -10% shock to all assets
2. **Moderate Decline**: -5% shock to all assets
3. **Market Rally**: +5% shock to all assets
4. **Strong Rally**: +10% shock to all assets
5. **Individual Asset Shocks**: Isolated shocks to each asset

### Outputs
- Stressed portfolio value
- Stressed P&L
- Comparison to VaR estimates
- Risk concentration analysis

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd risk_var_project

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
statsmodels>=0.13.0
python-pptx>=0.6.21
openpyxl>=3.0.10
```

## 💻 Usage

### Quick Start
```python
# Run complete VaR analysis
python main.py
```

### Custom Analysis
```python
from src.data_loader import DataLoader
from src.var_model import VaRModel
from src.backtesting import Backtester

# Load data
loader = DataLoader('data/sample_prices.csv')
prices, returns = loader.load_and_preprocess()

# Define portfolio weights
weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal-weighted

# Calculate VaR
var_model = VaRModel(returns, weights)
var_95 = var_model.historical_var(confidence=0.95)
var_99 = var_model.historical_var(confidence=0.99)

# Backtest
backtester = Backtester(returns, weights)
results = backtester.run_backtest(window=250, confidence=0.99)
```

### Generate Reports
```python
from src.report_generator import ReportGenerator

# Initialize report generator
reporter = ReportGenerator(
    returns=returns,
    weights=weights,
    var_results=var_results,
    backtest_results=backtest_results
)

# Generate Excel report
reporter.generate_excel_report('reports/var_report.xlsx')

# Generate PowerPoint deck
reporter.generate_ppt_report('reports/var_presentation.pptx')
```

## 📊 Output Examples

### Console Output
```
=== VaR Analysis Results ===
Portfolio Statistics:
- Daily Mean Return: 0.05%
- Daily Volatility: 1.23%
- Sharpe Ratio: 1.45

Historical VaR (1-day):
- 95% Confidence: $156,789
- 99% Confidence: $234,567

Monte Carlo VaR (1-day, 10,000 simulations):
- 95% Confidence: $152,341
- 99% Confidence: $228,976

Backtesting Results (Last 250 days):
- Exceptions at 99%: 3
- Exception Rate: 1.2%
- Traffic Light: GREEN ✓
- Model Status: ADEQUATE
```

### Generated Files
1. **var_report.xlsx**: Comprehensive Excel workbook with:
   - VaR summary tables
   - Backtesting results
   - Exception details
   - Stress test results
   - Portfolio analytics

2. **var_presentation.pptx**: Executive PowerPoint deck with:
   - Title slide
   - VaR methodology overview
   - Current VaR estimates with charts
   - Backtesting analysis with exception timeline
   - Traffic-light classification
   - Stress test results
   - Recommendations slide

3. **Visualization Charts**:
   - P&L vs VaR overlay plot
   - Exception timeline
   - Rolling volatility
   - Return distribution histogram
   - Stress test impact analysis

## 🎯 Use Cases

### Risk Management
- Daily VaR reporting for trading desks
- Portfolio risk limit monitoring
- Regulatory capital calculation (Basel III)

### Model Validation
- Independent model verification
- Regulatory backtesting requirements
- Model performance tracking

### Strategic Planning
- Risk-adjusted performance measurement
- Capital allocation decisions
- Stress testing for extreme scenarios

## 📈 Model Assumptions & Limitations

### Assumptions
1. **Historical VaR**: Past returns representative of future risk
2. **Monte Carlo**: Returns follow normal distribution (can be relaxed)
3. **Holding Period**: 1-day VaR (can be scaled)
4. **Liquidation**: Positions can be closed without market impact
5. **Correlations**: Historical correlations remain stable

### Limitations
1. **Tail Risk**: VaR doesn't measure severity beyond threshold (use Expected Shortfall)
2. **Model Risk**: All models are approximations of reality
3. **Data Quality**: Results depend on accurate, clean historical data
4. **Static Portfolio**: Assumes no rebalancing during holding period
5. **Normal Distribution**: MC method may underestimate tail risk

## 🔧 Customization & Extensions

### Add New Assets
Update portfolio weights and ensure corresponding price data exists:
```python
weights = [0.15, 0.15, 0.15, 0.20, 0.20, 0.15]  # 6 assets
```

### Change Confidence Levels
```python
var_999 = var_model.historical_var(confidence=0.999)  # 99.9% VaR
```

### Extend Stress Scenarios
```python
stress_tester.add_custom_scenario("Flash Crash", shocks=[-0.15, -0.15, -0.15, -0.15, -0.15])
```

### Alternative Distributions
Modify `monte_carlo.py` to use Student's t-distribution or other distributions for fat tails.

## 📚 References

1. **Basel Committee on Banking Supervision** - "Supervisory Framework for the Use of Backtesting in Conjunction with the Internal Models Approach"
2. **J.P. Morgan** - "RiskMetrics Technical Document" (1996)
3. **Jorion, Philippe** - "Value at Risk: The New Benchmark for Managing Financial Risk" (3rd Edition)
4. **Dowd, Kevin** - "Measuring Market Risk" (2nd Edition)

## 👨‍💼 Author & Professional Context

This project demonstrates:
- **Market Risk Analytics**: Industry-standard VaR methodologies
- **Model Validation**: Comprehensive backtesting frameworks
- **Regulatory Compliance**: Basel Committee standards
- **Production Code**: Clean, maintainable, enterprise-grade Python
- **Business Communication**: Automated reporting for stakeholders

**Target Roles**: Market Risk Analyst, Quantitative Risk Manager, Traded Risk Analytics, Model Validation

## 📄 License

This project is for educational and professional development purposes.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Expected Shortfall (CVaR) implementation
- Parametric VaR with GARCH models
- Multi-period VaR scaling
- Factor-based VaR decomposition
- Real-time data integration

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✓
