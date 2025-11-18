# Market Risk VaR System - Installation & Usage Guide

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Project Structure](#project-structure)
4. [Quick Start Guide](#quick-start-guide)
5. [Detailed Usage](#detailed-usage)
6. [Customization Options](#customization-options)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

---

## 🖥️ System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB for code and reports
- **Operating System**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 18.04+)

### Required Software
- Python 3.8+
- pip (Python package manager)
- Git (optional, for version control)
- Jupyter Notebook (optional, for interactive analysis)

---

## 🚀 Installation Steps

### Step 1: Clone or Download the Project

**Option A: Using Git**
```bash
git clone 
cd risk_var_project
```

**Option B: Manual Download**
1. Download the project ZIP file
2. Extract to your desired location
3. Open terminal/command prompt in the extracted folder

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**If you encounter errors, install packages individually:**
```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels python-pptx openpyxl
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, matplotlib; print('✓ Installation successful')"
```

### Step 5: Create Required Directories

```bash
# Windows
mkdir data reports reports\charts

# macOS/Linux
mkdir -p data reports/charts
```

---

## 📁 Project Structure

```
risk_var_project/
│
├── data/
│   └── sample_prices.csv              # Historical price data (auto-generated)
│
├── src/                               # Source code modules
│   ├── data_loader.py                 # Data ingestion & preprocessing
│   ├── var_model.py                   # Historical Simulation VaR
│   ├── monte_carlo.py                 # Monte Carlo VaR
│   ├── backtesting.py                 # Model validation & Basel tests
│   ├── stress_test.py                 # Stress testing framework
│   ├── dashboard.py                   # Visualization engine
│   └── report_generator.py            # Excel & PPT report generation
│
├── notebooks/
│   └── analysis.ipynb                 # Interactive Jupyter notebook
│
├── reports/                           # Generated reports (auto-created)
│   ├── charts/                        # Visualization outputs
│   ├── VaR_Risk_Report.xlsx          # Excel report
│   └── VaR_Executive_Presentation.pptx # PowerPoint deck
│
├── main.py                            # Main execution script
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── INSTALLATION_AND_USAGE_GUIDE.md   # This file
└── var_analysis.log                  # Execution logs (auto-created)
```

---

## ⚡ Quick Start Guide

### Run Complete Analysis (Automated)

```bash
python main.py
```

**What happens:**
1. Loads or generates sample data (5 assets, 3 years)
2. Calculates Historical Simulation VaR (95% & 99%)
3. Runs Monte Carlo simulations (10,000 iterations)
4. Performs backtesting with Basel traffic-light classification
5. Executes stress testing scenarios
6. Generates all visualizations
7. Creates Excel report and PowerPoint presentation

**Expected Output:**
- Console displays progress and results
- Excel report: `reports/VaR_Risk_Report.xlsx`
- PowerPoint: `reports/VaR_Executive_Presentation.pptx`
- Charts: `reports/charts/*.png`
- Log file: `var_analysis.log`

**Execution Time:** ~2-5 minutes (depending on hardware)

---

## 📊 Detailed Usage

### Using Your Own Data

#### Data Format Requirements

Your CSV file should have:
- **First column**: Dates (YYYY-MM-DD format)
- **Subsequent columns**: Daily prices for each asset
- **Header row**: Asset names

**Example: `my_portfolio.csv`**
```csv
Date,Stock_A,Stock_B,Bond_X,Commodity_Y
2022-01-03,105.50,203.25,98.75,1250.00
2022-01-04,106.25,205.10,98.80,1245.50
2022-01-05,104.80,204.50,98.85,1248.00
...
```

#### Update Portfolio Configuration

Edit `main.py` (lines 30-35):

```python
# Configuration
DATA_FILE = 'data/my_portfolio.csv'  # Your data file
PORTFOLIO_VALUE = 5_000_000          # Your portfolio value
WEIGHTS = [0.30, 0.25, 0.25, 0.20]  # Your asset weights (must sum to 1.0)
```

Then run:
```bash
python main.py
```

### Interactive Analysis with Jupyter Notebook

1. **Start Jupyter:**
```bash
jupyter notebook
```

2. **Open:** `notebooks/analysis.ipynb`

3. **Run cells sequentially** (Shift + Enter)

**Benefits:**
- Step-by-step exploration
- Modify parameters on-the-fly
- Create custom visualizations
- Export specific analyses

### Module-by-Module Usage

#### Load Data Only
```python
from src.data_loader import DataLoader

loader = DataLoader('data/sample_prices.csv')
prices, returns = loader.load_and_preprocess()

# View summary statistics
print(loader.get_summary_statistics())
print(loader.get_correlation_matrix())
```

#### Calculate VaR Only
```python
from src.var_model import VaRModel

weights = [0.2, 0.2, 0.2, 0.2, 0.2]
var_model = VaRModel(returns, weights, portfolio_value=10_000_000)

# Calculate VaR
var_95 = var_model.historical_var(confidence=0.95)
var_99 = var_model.historical_var(confidence=0.99)

print(f"95% VaR: ${var_95['var_absolute']:,.2f}")
print(f"99% VaR: ${var_99['var_absolute']:,.2f}")

# Component VaR
component_var = var_model.component_var(0.99)
print(component_var)
```

#### Monte Carlo Simulation
```python
from src.monte_carlo import MonteCarloVaR

mc_var = MonteCarloVaR(returns, weights, portfolio_value=10_000_000)

# Run simulations
mc_result = mc_var.calculate_var(confidence=0.99, n_simulations=10000)
print(f"Monte Carlo VaR: ${mc_result['var_absolute']:,.2f}")

# Compare distributions
comparison = mc_var.compare_distributions(0.99, 10000)
print(comparison)
```

#### Backtesting
```python
from src.backtesting import Backtester

backtester = Backtester(returns, weights, portfolio_value=10_000_000)

# Run backtest
results = backtester.run_backtest(window=250, confidence=0.99)

summary = results['summary']
print(f"Exceptions: {summary['n_exceptions']}")
print(f"Traffic Light: {summary['traffic_light']}")

# Statistical tests
kupiec = backtester.kupiec_test(results)
print(f"Kupiec Test: {kupiec['interpretation']}")
```

#### Stress Testing
```python
from src.stress_test import StressTester

stress_tester = StressTester(returns, weights, portfolio_value=10_000_000)

# Run scenarios
standard_scenarios = stress_tester.run_standard_scenarios()
individual_shocks = stress_tester.run_individual_shocks(-0.10)

# Get summary
summary = stress_tester.create_summary_table(standard_scenarios)
print(summary)
```

---

## 🎨 Customization Options

### 1. Change Confidence Levels

In `main.py`:
```python
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99, 0.999]  # Add 99.9% VaR
```

### 2. Adjust Backtesting Window

```python
BACKTEST_WINDOW = 500  # Use 500-day rolling window
```

### 3. Increase Monte Carlo Simulations

```python
MC_SIMULATIONS = 50000  # More accurate but slower
```

### 4. Custom Stress Scenarios

In `src/stress_test.py`, add to `run_standard_scenarios()`:

```python
scenarios = {
    'Market Crash (-10%)': [-0.10] * len(self.weights),
    'Custom Crisis': [-0.20, -0.15, -0.10, -0.05, 0.00],  # Custom shocks
    'Sector Specific': [-0.25, -0.25, 0.00, 0.00, 0.00]   # First 2 assets only
}
```

### 5. Modify Chart Styles

In `src/dashboard.py`:

```python
# Change color scheme
sns.set_palette("husl")

# Modify figure sizes
plt.rcParams['figure.figsize'] = (14, 8)

# Change fonts
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
```

### 6. Export Additional Formats

Add to `src/report_generator.py`:

```python
# Export to CSV
results_df.to_csv('reports/var_results.csv', index=False)

# Export to JSON
import json
with open('reports/var_results.json', 'w') as f:
    json.dump(var_results, f, indent=2)
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Module Import Errors
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Issue 2: Permission Denied (Windows)
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
- Close Excel files in `reports/` folder
- Run as administrator
- Or delete old report files first

#### Issue 3: Memory Error with Large Datasets
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce Monte Carlo simulations
MC_SIMULATIONS = 5000  # Instead of 50000

# Or use smaller data window
returns = returns.tail(252)  # Last year only
```

#### Issue 4: Matplotlib Display Issues
```
RuntimeError: Cannot show plot
```

**Solution:**
```python
# Add at top of script
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

#### Issue 5: Date Parsing Errors
```
ParserError: Unable to parse date
```

**Solution:**
Ensure your CSV has proper date format:
```python
# In data_loader.py, specify format
self.prices = pd.read_csv(
    self.filepath, 
    index_col=0, 
    parse_dates=True,
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
```

### Getting Help

1. **Check log file:** `var_analysis.log`
2. **Enable debug logging:**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```
3. **Test individual modules:**
   ```bash
   python src/data_loader.py
   python src/var_model.py
   ```

---

## 🚀 Advanced Features

### 1. Rolling Window Analysis

Analyze how VaR evolves over time:

```python
windows = [125, 250, 500]
comparison = backtester.compare_multiple_windows(windows, confidence=0.99)
print(comparison)
```

### 2. Expected Shortfall (CVaR)

Measure tail risk beyond VaR:

```python
es_result = var_model.calculate_expected_shortfall(0.99)
print(f"Expected Shortfall: ${es_result['es_absolute']:,.2f}")
print(f"ES/VaR Ratio: {es_result['es_var_ratio']:.2f}")
```

### 3. Distribution Comparison

Test different distributional assumptions:

```python
comparison = mc_var.compare_distributions(0.99, 10000)
print(f"Normal VaR: ${comparison['normal_var']:,.2f}")
print(f"t-distribution VaR: ${comparison['t_distribution_var']:,.2f}")
```

### 4. Correlation Stress Testing

Test portfolio under correlation breakdown:

```python
corr_stress = stress_tester.run_correlation_breakdown()
print(f"Volatility increase: {corr_stress['volatility_increase_pct']:.2f}%")
```

### 5. Automated Scheduling

**Windows (Task Scheduler):**
```bash
schtasks /create /tn "Daily VaR" /tr "C:\path\to\python.exe C:\path\to\main.py" /sc daily /st 08:00
```

**Linux/macOS (cron):**
```bash
# Edit crontab
crontab -e

# Add line (runs daily at 8 AM)
0 8 * * * cd /path/to/project && /path/to/python main.py >> var_cron.log 2>&1
```

### 6. Email Alerts

Add to `main.py`:

```python
import smtplib
from email.mime.text import MIMEText

def send_alert(summary):
    if summary['traffic_light'] in ['YELLOW', 'RED']:
        msg = MIMEText(f"VaR Model Alert: {summary['traffic_light']} zone")
        msg['Subject'] = 'Risk Alert'
        msg['From'] = 'risk@company.com'
        msg['To'] = 'manager@company.com'
        
        with smtplib.SMTP('smtp.company.com', 587) as server:
            server.send_message(msg)
```

### 7. Database Integration

Store results in database:

```python
import sqlite3

def save_to_database(var_results, backtest_results):
    conn = sqlite3.connect('risk_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO var_history (date, var_95, var_99, exceptions, traffic_light)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.now(),
        var_results['var_95'],
        var_results['var_99'],
        backtest_results['summary']['n_exceptions'],
        backtest_results['summary']['traffic_light']
    ))
    
    conn.commit()
    conn.close()
```

---

## 📈 Performance Optimization

### For Large Portfolios (50+ assets)

```python
# 1. Use parallel processing
from multiprocessing import Pool

# 2. Reduce data frequency
returns = returns.resample('W').last()  # Weekly instead of daily

# 3. Optimize Monte Carlo
MC_SIMULATIONS = 5000  # Reduce simulations
```

### For Real-time Analysis

```python
# Cache frequently used calculations
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_var_calculation(returns_hash, confidence):
    # VaR calculation here
    pass
```

---

## 🎓 Learning Resources

### Understanding VaR
- Basel Committee documentation
- J.P. Morgan RiskMetrics
- Philippe Jorion's "Value at Risk" textbook

### Python Resources
- pandas documentation: https://pandas.pydata.org
- NumPy documentation: https://numpy.org
- Matplotlib tutorials: https://matplotlib.org

### Financial Risk Management
- CFA Level 1 Market Risk section
- GARP FRM exam materials
- Investopedia VaR articles

---

## 📞 Support

For issues specific to this implementation:
1. Check the log file: `var_analysis.log`
2. Review this guide's Troubleshooting section
3. Test individual modules
4. Verify data format and quality

For general VaR methodology questions:
- Consult Basel Committee guidelines
- Review academic literature on VaR
- Seek guidance from risk management professionals

---

## ✅ Validation Checklist

Before using results in production:

- [ ] Verified data quality (no missing values, outliers)
- [ ] Confirmed weights sum to 1.0
- [ ] Reviewed backtesting results (Green zone preferred)
- [ ] Examined exception dates for patterns
- [ ] Validated stress test scenarios are realistic
- [ ] Compared Historical vs Monte Carlo results
- [ ] Documented all assumptions and limitations
- [ ] Obtained approval from risk management team

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Maintained by:** Risk Analytics Team
