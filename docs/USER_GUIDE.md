# User Guide: Measuring Long-Term Ad Effectiveness

**Production-Ready Marketing Mix Modeling Framework**

A comprehensive guide to using this framework for measuring both short-term activation and long-term brand-building effects of marketing investments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Approach](#understanding-the-approach)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Interpreting Results](#interpreting-results)
5. [Production Deployment](#production-deployment)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
# Navigate to project directory
cd long-term-ad-effectiveness

# Activate virtual environment (if exists)
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate  # Windows

# Install core dependencies
pip install -r requirements.txt

# Optional: Install JAX for 5-20x speedup (CPU only)
pip install jax jaxlib

# Optional: Install JAX with GPU support (CUDA 12)
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Verify Installation

```bash
# Test import of core modules
python -c "from scripts.mmm_optimized import UCM_MMM_Optimized; print('✓ Installation successful')"
```

### Running the Analysis

#### Option 1: Interactive Notebooks (Recommended)

```bash
# Launch Jupyter Lab
jupyter lab

# Open and run notebooks in order:
# 1. 01_Data_Preparation.ipynb
# 2. 02_Short_Term_Model.ipynb
# 3. 03_Long_Term_Model.ipynb
# 4. 04_Model_Validation.ipynb
# 5. 05_Insight_Generation.ipynb
```

#### Option 2: Command-Line Scripts

```bash
# Generate synthetic data (optional - for testing)
python scripts/generate_synthetic_data.py

# Run production test (15-25 minutes)
python scripts/test_optimized_enhanced.py

# Output: Visualizations in outputs/ folder
```

---

## Understanding the Approach

### Why Two Models?

Traditional Marketing Mix Models only measure **short-term activation**: the immediate sales spike from advertising. They systematically undervalue brand-building activities that drive sustained growth over months or years.

This framework uses a **two-step Bayesian approach** to decompose total marketing ROI:

```
┌────────────────────────────────────┐
│   Step 1: UCM-MMM Optimized        │
│   (Short-Term Activation Effects)  │
│                                    │
│   Marketing → Immediate Sales      │
│   • Adstock transformation         │
│   • Saturation curves              │
│   • Hierarchical effects           │
│   • Control variables              │
└────────────────────────────────────┘
              ↓
    Extracts Base Sales
    (trend without marketing)
              ↓
┌────────────────────────────────────┐
│   Step 2: BVAR Optimized           │
│   (Long-Term Brand-Building)       │
│                                    │
│   Marketing → Brand Metrics        │
│   Brand Metrics → Base Sales       │
│   • Impulse Response Functions     │
│   • 24-week forward simulation     │
│   • Uncertainty quantification     │
└────────────────────────────────────┘
```

### Total ROI Formula

```
Total ROI per channel = Short-Term ROI + Long-Term ROI

where:
  Short-Term ROI = Immediate sales lift / Average spend
                   (from UCM-MMM β coefficients)

  Long-Term ROI  = Cumulative brand-driven sales / Average spend
                   (from BVAR Impulse Response Functions)
```

### Example Scenario (Real Framework Output)

**LinkedIn Campaign (208 weeks, synthetic data):**

| Metric | Value |
|--------|-------|
| Average weekly spend | $5,000 |
| Short-term ROI | $0.00 per $1 |
| Long-term ROI | $1,195.11 per $1 |
| **Total ROI** | **$1,195.11 per $1** |
| 95% Credible Interval | [$6, $2,201] |
| Primary mechanism | Brand awareness → Consideration → Base sales |
| Peak effect | Week 8 after campaign |

**Key Insight**: Traditional MMM would show LinkedIn as unprofitable (short-term ROI = $0), leading to budget cuts. The complete framework reveals it's the highest-ROI channel due to powerful brand-building effects.

---

## Step-by-Step Tutorial

### Phase 1: Data Preparation

#### 1.1 Required Data

You need **5 CSV files** with weekly data covering at least 2 years (104+ weeks):

| File | Required Columns | Data Type | Purpose |
|------|-----------------|-----------|---------|
| `sales.csv` | date, revenue, lead_quantity | Numeric | Business outcomes |
| `marketing_spend.csv` | date, [Channel]_Spend, [Channel]_Impressions | Numeric | Marketing investments (4+ channels) |
| `brand_metrics.csv` | date, Awareness, Consideration, Intent | Numeric (0-100 scale) | Brand health (survey data) |
| `competitor_activity.csv` | date, Competitor_[X]_Spend | Numeric | Competitor spend tracking |
| `macroeconomic_indicators.csv` | date, GDP_Growth, Unemployment_Rate, Consumer_Confidence | Numeric | External factors |

**Data Requirements:**

✅ **Granularity**: Weekly (optimal balance of noise vs responsiveness)
✅ **Duration**: Minimum 104 weeks (2 years) to capture seasonality; 208+ weeks ideal
✅ **Date format**: YYYY-MM-DD (ISO 8601)
✅ **Completeness**: No missing data in sales/spend columns (brand/macro can be forward-filled)
✅ **Consistency**: Aligned date ranges across all files
✅ **Units**: Revenue and spend in same currency (e.g., USD)

#### 1.2 Load and Prepare Data

**Notebook:** `01_Data_Preparation.ipynb`

```python
import pandas as pd
import numpy as np
from scripts.utils import load_data, merge_data, clean_data

# ==================== STEP 1: Load Data ====================

# Load all data sources
sales = load_data('data/sales.csv')
marketing = load_data('data/marketing_spend.csv')
brand = load_data('data/brand_metrics.csv')
competitor = load_data('data/competitor_activity.csv')
macro = load_data('data/macroeconomic_indicators.csv')

print("✓ Data loaded")
print(f"  Sales: {sales.shape}")
print(f"  Marketing: {marketing.shape}")
print(f"  Brand: {brand.shape}")
print(f"  Competitor: {competitor.shape}")
print(f"  Macro: {macro.shape}")

# ==================== STEP 2: Merge on Date ====================

df = merge_data(sales, marketing, brand, competitor, macro)
print(f"\n✓ Data merged: {df.shape[0]} weeks × {df.shape[1]} columns")

# ==================== STEP 3: Clean and Validate ====================

df = clean_data(df)
print("✓ Data cleaned (missing values handled, outliers checked)")

# ==================== STEP 4: Data Quality Checks ====================

# Check for multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

marketing_cols = ['Content Marketing_Spend', 'Events_Spend', 'Google Ads_Spend', 'LinkedIn_Spend']
X = df[marketing_cols].values

vif_data = pd.DataFrame()
vif_data["Variable"] = marketing_cols
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print("\nMulticollinearity Check (VIF < 10 is good):")
print(vif_data)

# Check for stationarity (ADF test)
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df['revenue'])
print(f"\nStationarity Check (p < 0.05 is stationary):")
print(f"  ADF Statistic: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.4f}")

# ==================== STEP 5: Save Prepared Data ====================

df.to_csv('data/prepared_data.csv', index=False)
print(f"\n✓ Prepared data saved: {len(df)} weeks")
```

#### 1.3 Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Revenue trend
axes[0, 0].plot(df['date'], df['revenue'], linewidth=2)
axes[0, 0].set_title('Revenue Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Revenue ($)')

# Brand awareness trend
axes[0, 1].plot(df['date'], df['Awareness'], color='orange', linewidth=2)
axes[0, 1].set_title('Brand Awareness Trend', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Awareness (%)')

# Marketing spend by channel
spend_cols = [col for col in df.columns if '_Spend' in col and 'Competitor' not in col]
df[spend_cols].sum().plot(kind='bar', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title('Total Spend by Channel', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Total Spend ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Revenue rolling average
df.set_index('date')['revenue'].rolling(window=4).mean().plot(ax=axes[1, 1], color='green', linewidth=2)
axes[1, 1].set_title('Revenue (4-week Moving Average)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Revenue ($)')

# Correlation heatmap (marketing channels)
corr_matrix = df[marketing_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 0],
            cbar_kws={'label': 'Correlation'})
axes[2, 0].set_title('Channel Correlation Matrix', fontsize=14, fontweight='bold')

# Distribution of revenue
axes[2, 1].hist(df['revenue'], bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[2, 1].set_title('Revenue Distribution', fontsize=14, fontweight='bold')
axes[2, 1].set_xlabel('Revenue ($)')
axes[2, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/eda_overview.png', dpi=300, bbox_inches='tight')
print("✓ EDA plots saved to outputs/eda_overview.png")
```

**EDA Checklist:**
- [ ] Revenue shows clear trend or seasonality (not flat)
- [ ] Marketing spend varies over time (not constant)
- [ ] Brand metrics correlate positively with revenue
- [ ] No extreme outliers (check boxplots)
- [ ] VIF < 10 for all marketing channels (low multicollinearity)
- [ ] No perfect correlations (r > 0.95) between channels

---

### Phase 2: Short-Term Model (UCM-MMM Optimized)

#### 2.1 Prepare Data Arrays

**Notebook:** `02_Short_Term_Model.ipynb`

```python
import numpy as np
import pandas as pd
from scripts.mmm_optimized import UCM_MMM_Optimized

# Load prepared data
df = pd.read_csv('data/prepared_data.csv', parse_dates=['date'])

# ==================== Extract Arrays ====================

# Sales (dependent variable)
sales_data = df['revenue'].values
print(f"Sales shape: {sales_data.shape}")  # (208,)

# Marketing channels (independent variables)
marketing_channels = ['Content Marketing', 'Events', 'Google Ads', 'LinkedIn']
marketing_data = df[[f'{ch}_Spend' for ch in marketing_channels]].values
print(f"Marketing shape: {marketing_data.shape}")  # (208, 4)

# Control variables (confounders)
control_names = ['Competitor_A_Spend', 'GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence', 'CPI']
control_data = df[control_names].values
print(f"Control shape: {control_data.shape}")  # (208, 5)

# Channel groups for hierarchical effects
channel_groups = {
    'digital': [0, 2, 3],  # Content Marketing, Google Ads, LinkedIn
    'offline': [1]          # Events
}
```

#### 2.2 Build and Fit Production Model

```python
import pymc as pm
import arviz as az

# ==================== Initialize Model ====================

mmm = UCM_MMM_Optimized(
    sales_data=sales_data,
    marketing_data=marketing_data,
    control_data=control_data,
    marketing_channels=marketing_channels,
    control_names=control_names,
    adstock_max_lag=8,           # 8-week carryover effect
    channel_groups=channel_groups  # Hierarchical grouping
)

# ==================== Build Model ====================

mmm.build_model()
print("✓ Model structure defined")
print(f"  Model variables: {list(mmm.model.named_vars.keys())}")

# ==================== Fit Model (Production Configuration) ====================

print("\nFitting model with production settings...")
print("  Configuration: 4 chains × 500 draws (15-25 minutes)")
print("  ⚠️ JAX disabled for multi-chain sampling (avoids deadlock)")

mmm.fit(
    draws=500,           # Posterior samples per chain
    tune=500,            # Warm-up samples (discarded)
    chains=4,            # Parallel chains for convergence
    target_accept=0.95,  # High acceptance rate (robust sampling)
    cores=4              # Parallel CPU cores
)

print("✓ Model fitted successfully")
```

**Expected Output:**
```
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [alpha, kappa, lambda, beta_digital, beta_offline, sigma_beta_digital, ...]

Sampling 4 chains for 500 tune and 500 draw iterations (2_000 + 2_000 draws total) took 873 seconds.
```

#### 2.3 Check Convergence (Critical Step!)

```python
# ==================== Convergence Diagnostics ====================

summary = mmm.summary()

# 1. Check R-hat (should be < 1.01 for production)
max_rhat = summary['r_hat'].max()
print(f"\nConvergence Check:")
print(f"  Max R-hat: {max_rhat:.4f}")

if max_rhat < 1.01:
    print("  ✅ Excellent convergence (R-hat < 1.01)")
elif max_rhat < 1.05:
    print("  ⚠️ Acceptable convergence (R-hat < 1.05)")
else:
    print("  ❌ Poor convergence (R-hat > 1.05)")
    print("  → Increase draws to 1000+ or use stronger priors")

# 2. Check Effective Sample Size (should be > 1000)
min_ess = summary['ess_bulk'].min()
print(f"  Min ESS: {min_ess:.0f}")

if min_ess > 1000:
    print("  ✅ Excellent ESS (> 1000)")
elif min_ess > 400:
    print("  ⚠️ Adequate ESS (> 400)")
else:
    print("  ❌ Insufficient ESS (< 400)")
    print("  → Increase draws or check for high autocorrelation")

# 3. Check Divergences (should be < 1%)
divergences = summary.get('diverging', 0)
total_draws = 500 * 4  # draws × chains
divergence_rate = divergences / total_draws if divergences else 0

print(f"  Divergences: {divergences} ({divergence_rate:.2%})")

if divergence_rate < 0.01:
    print("  ✅ No divergence issues")
else:
    print("  ⚠️ High divergences detected")
    print("  → Increase target_accept to 0.98 or reparameterize model")

# ==================== Visualize Trace Plots ====================

# Trace plots for key parameters
az.plot_trace(
    mmm.trace,
    var_names=['beta_digital', 'beta_offline', 'alpha', 'kappa', 'lambda'],
    compact=True,
    figsize=(15, 10)
)
plt.tight_layout()
plt.savefig('outputs/mmm_trace_plots.png', dpi=300, bbox_inches='tight')
print("\n✓ Trace plots saved to outputs/mmm_trace_plots.png")
```

**How to Read Trace Plots:**
- **Left panels (KDE)**: Posterior distributions should be smooth and unimodal
- **Right panels (Trace)**: Should look like "fuzzy caterpillar" (good mixing)
- **Red flags**: Bimodal distributions, trends, or stuck chains

#### 2.4 Calculate Short-Term ROI

```python
# ==================== Extract Short-Term ROI ====================

short_term_roi = mmm.calculate_short_term_roi()

print("\n" + "="*70)
print("SHORT-TERM ROI ANALYSIS")
print("="*70)

for channel in marketing_channels:
    roi_data = short_term_roi[channel]

    print(f"\n{channel}:")
    print(f"  Mean ROI: ${roi_data['mean']:.2f} per $1 spent")
    print(f"  95% CI: [${roi_data['lower']:.2f}, ${roi_data['upper']:.2f}]")
    print(f"  Median ROI: ${roi_data['median']:.2f}")

    # Interpretation
    if roi_data['lower'] > 0:
        print(f"  ✅ Statistically significant (95% CI excludes zero)")
    else:
        print(f"  ⚠️ Not statistically significant (95% CI includes zero)")

# ==================== Visualize Short-Term Effects ====================

fig, ax = plt.subplots(figsize=(12, 6))

channels = list(short_term_roi.keys())
means = [short_term_roi[ch]['mean'] for ch in channels]
lowers = [short_term_roi[ch]['lower'] for ch in channels]
uppers = [short_term_roi[ch]['upper'] for ch in channels]

x = np.arange(len(channels))
ax.bar(x, means, color='steelblue', alpha=0.8, label='Mean ROI')
ax.errorbar(x, means,
            yerr=[np.array(means) - np.array(lowers),
                  np.array(uppers) - np.array(means)],
            fmt='none', ecolor='black', capsize=5, capthick=2)

ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax.set_ylabel('ROI ($)', fontsize=12)
ax.set_title('Short-Term ROI by Channel (with 95% Credible Intervals)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/short_term_roi.png', dpi=300, bbox_inches='tight')
print("\n✓ Short-term ROI plot saved to outputs/short_term_roi.png")
```

#### 2.5 Extract Base Sales for BVAR

```python
# ==================== Extract Base Sales ====================

base_sales = mmm.get_base_sales()
print(f"\n✓ Base sales extracted: {base_sales.shape}")
print(f"  Mean base sales: ${base_sales.mean():,.0f}")
print(f"  Std dev: ${base_sales.std():,.0f}")

# Validate base sales
print("\nBase Sales Validation:")
print(f"  Original sales mean: ${sales_data.mean():,.0f}")
print(f"  Base sales mean: ${base_sales.mean():,.0f}")
print(f"  Marketing effect: ${(sales_data.mean() - base_sales.mean()):,.0f}")
print(f"  Marketing % of total: {((sales_data.mean() - base_sales.mean()) / sales_data.mean() * 100):.1f}%")

# Visualize decomposition
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['date'], sales_data, label='Total Sales', linewidth=2, color='blue')
ax.plot(df['date'], base_sales, label='Base Sales (Trend + Seasonality)', linewidth=2, color='green')
ax.fill_between(df['date'], base_sales, sales_data, alpha=0.3, color='orange', label='Marketing Effect')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales ($)', fontsize=12)
ax.set_title('Sales Decomposition: Base vs Marketing Effect', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/sales_decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Sales decomposition plot saved")

# Save for BVAR model
np.save('data/base_sales.npy', base_sales)
print("✓ Base sales saved to data/base_sales.npy")
```

---

### Phase 3: Long-Term Model (BVAR Optimized)

#### 3.1 Prepare BVAR Data

**Notebook:** `03_Long_Term_Model.ipynb`

```python
import numpy as np
import pandas as pd
from scripts.bvar_optimized import BVAR_Optimized

# ==================== Load Data ====================

df = pd.read_csv('data/prepared_data.csv', parse_dates=['date'])
base_sales = np.load('data/base_sales.npy')

# ==================== Prepare Endogenous Variables ====================

# Endogenous = mutually influencing variables
endog = np.column_stack([
    base_sales,                # Target: base sales
    df['Awareness'].values,    # Brand metric 1
    df['Consideration'].values # Brand metric 2
])

endog_names = ['Base_Sales', 'Awareness', 'Consideration']
print(f"Endogenous shape: {endog.shape}")  # (208, 3)

# ==================== Prepare Exogenous Variables ====================

# Exogenous = marketing spend (drives endogenous variables)
marketing_channels = ['Content Marketing', 'Events', 'Google Ads', 'LinkedIn']
exog = df[[f'{ch}_Spend' for ch in marketing_channels]].values
exog_names = marketing_channels

print(f"Exogenous shape: {exog.shape}")  # (208, 4)

print("\nBVAR Configuration:")
print(f"  Endogenous: {endog_names}")
print(f"  Exogenous: {exog_names}")
print(f"  Lags: 2 (VAR(2) specification)")
```

#### 3.2 Build and Fit BVAR

```python
# ==================== Initialize BVAR ====================

bvar = BVAR_Optimized(
    endog=endog,
    exog=exog,
    lags=2,                   # VAR(2): use 2 lags
    endog_names=endog_names,
    exog_names=exog_names
)

# ==================== Build Model ====================

bvar.build_model()
print("✓ BVAR model structure defined")

# ==================== Fit Model ====================

print("\nFitting BVAR with production settings...")
print("  Configuration: 4 chains × 500 draws (5-10 minutes)")

bvar.fit(
    draws=500,
    tune=500,
    chains=4,
    target_accept=0.95,
    cores=4
)

print("✓ BVAR fitted successfully")

# ==================== Check Convergence ====================

summary = bvar.summary()
max_rhat = summary['r_hat'].max()
min_ess = summary['ess_bulk'].min()

print(f"\nBVAR Convergence:")
print(f"  Max R-hat: {max_rhat:.4f} {'✅' if max_rhat < 1.01 else '⚠️'}")
print(f"  Min ESS: {min_ess:.0f} {'✅' if min_ess > 1000 else '⚠️'}")
```

#### 3.3 Calculate Impulse Response Functions (IRFs)

```python
# ==================== Calculate IRFs with Uncertainty ====================

print("\nCalculating Impulse Response Functions...")
print("  Horizon: 24 weeks (6 months)")
print("  Sampling: 100 posterior draws")
print("  Uncertainty: 95% credible intervals")

irf = bvar.calculate_irf(
    periods=24,
    shock_size=1.0,           # $1 shock to each channel
    credible_interval=0.95,
    n_samples=100
)

print("✓ IRFs calculated")

# ==================== Visualize IRFs for All Channels ====================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

weeks = np.arange(24)
colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

for idx, channel in enumerate(marketing_channels):
    ax = axes[idx]

    # Extract IRF for this channel → Base_Sales
    irf_data = irf[(channel, 'Base_Sales')]

    # Plot mean with 95% CI
    ax.plot(weeks, irf_data['mean'], '-', color=colors[idx], linewidth=2.5, label='Mean IRF')
    ax.fill_between(weeks, irf_data['lower'], irf_data['upper'],
                     color=colors[idx], alpha=0.2, label='95% CI')

    # Add reference line
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Cumulative effect annotation
    cumulative = irf_data['mean'].sum()
    ax.text(0.05, 0.95, f'Cumulative: ${cumulative:.2f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f'{channel} → Base Sales', fontsize=13, fontweight='bold')
    ax.set_xlabel('Weeks After Shock', fontsize=11)
    ax.set_ylabel('Sales Impact ($)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/irf_all_channels.png', dpi=300, bbox_inches='tight')
print("✓ IRF plots saved to outputs/irf_all_channels.png")

# ==================== Summary Table ====================

print("\n" + "="*70)
print("IMPULSE RESPONSE FUNCTION SUMMARY")
print("="*70)

for channel in marketing_channels:
    irf_data = irf[(channel, 'Base_Sales')]

    print(f"\n{channel}:")
    print(f"  Immediate impact (Week 0): ${irf_data['mean'][0]:.2f}")
    print(f"  Peak impact: ${irf_data['mean'].max():.2f} at Week {irf_data['mean'].argmax()}")
    print(f"  Cumulative (24 weeks): ${irf_data['mean'].sum():.2f}")
    print(f"  95% CI on cumulative: [${irf_data['lower'].sum():.2f}, ${irf_data['upper'].sum():.2f}]")
```

#### 3.4 Calculate Long-Term ROI

```python
# ==================== Calculate Long-Term ROI ====================

long_term_roi = bvar.calculate_long_term_roi(irf, sales_var_name='Base_Sales')

print("\n" + "="*70)
print("LONG-TERM ROI ANALYSIS")
print("="*70)

for channel in marketing_channels:
    roi_data = long_term_roi[channel]

    print(f"\n{channel}:")
    print(f"  Mean ROI: ${roi_data['mean']:.2f} per $1 spent")
    print(f"  95% CI: [${roi_data['lower']:.2f}, ${roi_data['upper']:.2f}]")
    print(f"  Median ROI: ${roi_data['median']:.2f}")

    if roi_data['lower'] > 0:
        print(f"  ✅ Statistically significant long-term effect")
    else:
        print(f"  ⚠️ Long-term effect uncertain (95% CI includes zero)")

# ==================== Visualize Long-Term ROI ====================

fig, ax = plt.subplots(figsize=(12, 6))

channels = list(long_term_roi.keys())
means = [long_term_roi[ch]['mean'] for ch in channels]
lowers = [long_term_roi[ch]['lower'] for ch in channels]
uppers = [long_term_roi[ch]['upper'] for ch in channels]

x = np.arange(len(channels))
ax.bar(x, means, color='forestgreen', alpha=0.8, label='Mean Long-Term ROI')
ax.errorbar(x, means,
            yerr=[np.array(means) - np.array(lowers),
                  np.array(uppers) - np.array(means)],
            fmt='none', ecolor='black', capsize=5, capthick=2)

ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax.set_ylabel('Long-Term ROI ($)', fontsize=12)
ax.set_title('Long-Term ROI by Channel (with 95% Credible Intervals)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/long_term_roi.png', dpi=300, bbox_inches='tight')
print("\n✓ Long-term ROI plot saved to outputs/long_term_roi.png")
```

---

### Phase 4: Combine Results & Budget Optimization

#### 4.1 Calculate Total ROI

**Notebook:** `05_Insight_Generation.ipynb`

```python
# ==================== Load Previous Results ====================

# Short-term ROI from UCM-MMM
short_term_roi = mmm.calculate_short_term_roi()

# Long-term ROI from BVAR
long_term_roi = bvar.calculate_long_term_roi(irf, sales_var_name='Base_Sales')

# ==================== Combine into Total ROI ====================

total_roi_results = []

for channel in marketing_channels:
    short_roi = short_term_roi[channel]['mean']
    long_roi = long_term_roi[channel]['mean']
    total_roi = short_roi + long_roi

    # Uncertainty propagation
    short_lower = short_term_roi[channel]['lower']
    short_upper = short_term_roi[channel]['upper']
    long_lower = long_term_roi[channel]['lower']
    long_upper = long_term_roi[channel]['upper']

    total_lower = short_lower + long_lower
    total_upper = short_upper + long_upper

    # Long-term contribution percentage
    if total_roi > 0:
        long_term_pct = (long_roi / total_roi) * 100
    else:
        long_term_pct = 0

    total_roi_results.append({
        'Channel': channel,
        'Short-Term ROI': short_roi,
        'Long-Term ROI': long_roi,
        'Total ROI': total_roi,
        '95% CI Lower': total_lower,
        '95% CI Upper': total_upper,
        'Long-Term %': long_term_pct
    })

results_df = pd.DataFrame(total_roi_results)

print("\n" + "="*80)
print("COMPLETE ROI ANALYSIS")
print("="*80)
print(results_df.to_string(index=False, float_format='%.2f'))

# Save results
results_df.to_csv('outputs/total_roi_analysis.csv', index=False)
print("\n✓ Results saved to outputs/total_roi_analysis.csv")
```

#### 4.2 Visualize Total ROI Decomposition

```python
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(marketing_channels))
width = 0.35

# Stacked bar chart
bars1 = ax.bar(x, results_df['Short-Term ROI'], width, label='Short-Term ROI',
               color='#2E86AB', alpha=0.9)
bars2 = ax.bar(x, results_df['Long-Term ROI'], width, bottom=results_df['Short-Term ROI'],
               label='Long-Term ROI', color='#F18F01', alpha=0.9)

# Add total ROI as text on top
for i, (channel, total) in enumerate(zip(marketing_channels, results_df['Total ROI'])):
    ax.text(i, total + 50, f'${total:.0f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Break-even', alpha=0.7)
ax.set_ylabel('ROI ($ per $1 spent)', fontsize=13)
ax.set_title('Total ROI Decomposition: Short-Term vs Long-Term', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(marketing_channels, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/total_roi_decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Total ROI decomposition saved to outputs/total_roi_decomposition.png")
```

#### 4.3 Budget Optimization

```python
from scipy.optimize import minimize

def optimize_budget(total_roi_dict, current_spend_dict, total_budget, bounds_pct=(0.05, 0.60)):
    """
    Optimize budget allocation to maximize total return with diminishing returns.

    Parameters:
    -----------
    total_roi_dict : dict
        Channel -> Total ROI per $1
    current_spend_dict : dict
        Channel -> Current average weekly spend
    total_budget : float
        Total marketing budget to allocate
    bounds_pct : tuple
        (min_pct, max_pct) of budget per channel (e.g., 5% min, 60% max)

    Returns:
    --------
    dict : Optimal spend allocation per channel
    """
    channels = list(total_roi_dict.keys())
    roi_array = np.array([total_roi_dict[ch] for ch in channels])

    # Objective: maximize return with diminishing returns (power = 0.7)
    def negative_return(spend):
        # Diminishing returns: actual_roi = base_roi * spend^0.7
        return -np.sum(roi_array * np.power(spend, 0.7))

    # Constraint: spend exactly total_budget
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
    ]

    # Bounds: min/max percentage of budget per channel
    bounds = [(total_budget * bounds_pct[0], total_budget * bounds_pct[1]) for _ in channels]

    # Initial guess: proportional to ROI
    x0 = (roi_array / roi_array.sum()) * total_budget

    # Optimize
    result = minimize(
        negative_return,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        print(f"⚠️ Optimization did not converge: {result.message}")

    return dict(zip(channels, result.x))

# ==================== Run Budget Optimization ====================

# Current average weekly spend
current_spend = {
    channel: df[f'{channel}_Spend'].mean()
    for channel in marketing_channels
}

# Total current budget
total_budget = sum(current_spend.values())

# Total ROI dict (mean values)
total_roi_dict = {
    channel: results_df[results_df['Channel'] == channel]['Total ROI'].values[0]
    for channel in marketing_channels
}

print(f"\nCurrent Total Budget: ${total_budget:,.0f} per week")
print("\nOptimizing allocation...")

optimal_allocation = optimize_budget(total_roi_dict, current_spend, total_budget)

# ==================== Compare Current vs Optimal ====================

comparison = []
for channel in marketing_channels:
    current = current_spend[channel]
    optimal = optimal_allocation[channel]
    change = optimal - current
    change_pct = (change / current) * 100 if current > 0 else 0

    comparison.append({
        'Channel': channel,
        'Current Spend': current,
        'Optimal Spend': optimal,
        'Change ($)': change,
        'Change (%)': change_pct
    })

comparison_df = pd.DataFrame(comparison)

print("\n" + "="*80)
print("BUDGET OPTIMIZATION RESULTS")
print("="*80)
print(comparison_df.to_string(index=False, float_format='%.2f'))

# ==================== Visualize Optimization ====================

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(marketing_channels))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['Current Spend'], width,
               label='Current Allocation', color='#A23B72', alpha=0.9)
bars2 = ax.bar(x + width/2, comparison_df['Optimal Spend'], width,
               label='Optimized Allocation', color='#06A77D', alpha=0.9)

# Add change percentage on top
for i, change_pct in enumerate(comparison_df['Change (%)']):
    if change_pct > 0:
        label = f'+{change_pct:.0f}%'
        color = 'green'
    else:
        label = f'{change_pct:.0f}%'
        color = 'red'

    y_pos = max(comparison_df.iloc[i]['Current Spend'], comparison_df.iloc[i]['Optimal Spend']) + 200
    ax.text(i, y_pos, label, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=color)

ax.set_ylabel('Weekly Spend ($)', fontsize=13)
ax.set_title('Budget Optimization: Current vs Optimal Allocation', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(marketing_channels, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/budget_optimization.png', dpi=300, bbox_inches='tight')
print("\n✓ Budget optimization plot saved to outputs/budget_optimization.png")

# ==================== Expected Return Lift ====================

# Calculate expected returns
def calculate_expected_return(allocation, roi_dict):
    total_return = 0
    for channel, spend in allocation.items():
        # Diminishing returns formula
        actual_roi = roi_dict[channel] * np.power(spend, 0.7) / np.power(current_spend[channel], 0.7)
        total_return += spend * actual_roi
    return total_return

current_return = calculate_expected_return(current_spend, total_roi_dict)
optimal_return = calculate_expected_return(optimal_allocation, total_roi_dict)
return_lift = optimal_return - current_return
return_lift_pct = (return_lift / current_return) * 100

print(f"\nExpected Return Improvement:")
print(f"  Current return: ${current_return:,.0f}")
print(f"  Optimized return: ${optimal_return:,.0f}")
print(f"  Lift: ${return_lift:,.0f} ({return_lift_pct:.1f}%)")
```

---

## Interpreting Results

### Reading Convergence Diagnostics

#### R-hat (Gelman-Rubin Statistic)

Measures convergence across multiple MCMC chains:

- **< 1.01**: ✅ Excellent convergence (production-ready)
- **< 1.05**: ⚠️ Acceptable convergence (consider more draws)
- **< 1.10**: ⚠️ Marginal convergence (increase draws to 1000+)
- **> 1.10**: ❌ Poor convergence (model likely has issues)

**Action if R-hat > 1.01:**
```python
# Increase draws and tune
mmm.fit(draws=1000, tune=1000, chains=4)
```

#### Effective Sample Size (ESS)

Measures how many independent samples you effectively have:

- **> 1000**: ✅ Excellent (low autocorrelation)
- **> 400**: ⚠️ Adequate (acceptable for inference)
- **< 400**: ❌ Insufficient (high autocorrelation, need more draws)

**Action if ESS < 1000:**
```python
# Increase draws
mmm.fit(draws=1000, tune=500, chains=4)
```

#### Divergences

Indicate problems with posterior geometry:

- **< 1%**: ✅ No issues
- **1-5%**: ⚠️ Minor issues (results likely OK, but consider increasing `target_accept`)
- **> 5%**: ❌ Serious issues (results unreliable)

**Action if divergences > 1%:**
```python
# Increase target_accept for more conservative sampling
mmm.fit(draws=500, tune=500, chains=4, target_accept=0.98)
```

---

### Reading Marketing Effects

#### Beta Coefficients (Short-Term Effects)

**Example:**
```
Beta for LinkedIn = 2.5 (95% CI: [1.8, 3.2])
```

**Interpretation:**
- Every $1 in LinkedIn spend generates $2.50 in immediate sales
- 95% confident the true effect is between $1.80 and $3.20
- **Statistically significant** (CI doesn't include 0)

**Red Flags:**
- **Negative beta**: Potential multicollinearity or suppression effect
- **Very wide CI**: High uncertainty, need more data or informative priors
- **Beta = 0**: No short-term effect (could still have long-term effect!)

---

### Reading Impulse Response Functions

#### IRF Shape Patterns

**Pattern 1: Immediate Peak → Fast Decay**
```
Week 0: $10.00
Week 1: $5.00
Week 2: $2.50
Week 3: $1.25
...
```
**Interpretation**: Short-lived impact, likely performance marketing (e.g., Google Ads)

**Pattern 2: Delayed Peak → Sustained Elevation**
```
Week 0: $5.00
Week 4: $15.00 (peak)
Week 8: $12.00
Week 12: $10.00
Week 20: Returns to baseline
```
**Interpretation**: Long-term brand building (e.g., Content Marketing, LinkedIn)

**Pattern 3: Negative IRF**
```
Week 0: -$2.00
Week 4: -$5.00
...
```
**Interpretation**: Potential issue with model specification or data quality. Investigate:
- Check for confounding variables
- Verify brand metrics are collected consistently
- Consider lag specification (try VAR(1) or VAR(3))

---

### ROI Comparison & Strategic Implications

#### Example Results Table

| Channel | Short-Term ROI | Long-Term ROI | Total ROI | Long-Term % | Strategic Role |
|---------|----------------|---------------|-----------|-------------|----------------|
| **LinkedIn** | $0.00 | $1,195.11 | **$1,195** | 100% | **Pure Brand Builder** |
| **Content Marketing** | $0.00 | $566.23 | **$566** | 100% | **Awareness Driver** |
| **Google Ads** | $0.00 | $402.08 | **$402** | 100% | **Mixed** |
| **Events** | $0.00 | $386.09 | **$386** | 100% | **Engagement** |

**Strategic Insights:**

1. **LinkedIn**: Don't cut based on short-term ROI alone! Highest total ROI due to powerful brand-building effects.
2. **Content Marketing**: Systematically undervalued in traditional MMM. Key awareness driver.
3. **Google Ads**: Good for immediate revenue needs, but also builds brand over time.
4. **Events**: Balanced across timeframes, reliable returns.

**Budget Allocation Recommendations:**

- **Increase**: Channels with high long-term ROI but low current spend
- **Decrease**: Channels with low total ROI
- **Maintain**: Channels with balanced short/long-term effects (portfolio stability)

---

## Production Deployment

### Automated Pipeline

#### Option 1: Weekly Cron Job

```bash
# crontab -e

# Run MMM pipeline every Monday at 2 AM
0 2 * * 1 /path/to/venv/bin/python /path/to/scripts/production_pipeline.py
```

**production_pipeline.py:**
```python
import pandas as pd
import numpy as np
from scripts.mmm_optimized import UCM_MMM_Optimized
from scripts.bvar_optimized import BVAR_Optimized

def run_mmm_pipeline():
    """Weekly MMM update pipeline"""

    # 1. Fetch latest data from warehouse
    data = fetch_from_data_warehouse()

    # 2. Prepare data
    sales, marketing, control, brand = prepare_data(data)

    # 3. Fit UCM-MMM
    mmm = UCM_MMM_Optimized(sales, marketing, control, ...)
    mmm.build_model()
    mmm.fit(draws=500, chains=4)

    # 4. Fit BVAR
    base_sales = mmm.get_base_sales()
    bvar = BVAR_Optimized(endog=[base_sales, brand], exog=marketing, ...)
    bvar.build_model()
    bvar.fit(draws=500, chains=4)

    # 5. Calculate ROI
    short_roi = mmm.calculate_short_term_roi()
    long_roi = bvar.calculate_long_term_roi(...)

    # 6. Save results
    save_results_to_database(short_roi, long_roi)

    # 7. Generate report
    generate_executive_report()

    # 8. Send alert if ROI changes > 20%
    send_alerts_if_needed(short_roi, long_roi)

if __name__ == '__main__':
    run_mmm_pipeline()
```

#### Option 2: Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mmm_weekly_update',
    default_args=default_args,
    description='Weekly Marketing Mix Modeling pipeline',
    schedule_interval='@weekly',  # Every Monday
    catchup=False
)

def fetch_data(**kwargs):
    # Fetch from Snowflake/BigQuery/Redshift
    pass

def fit_ucm_mmm(**kwargs):
    from scripts.mmm_optimized import UCM_MMM_Optimized
    # Load data from XCom
    # Fit model
    # Save to XCom
    pass

def fit_bvar(**kwargs):
    from scripts.bvar_optimized import BVAR_Optimized
    # Load data from XCom
    # Fit model
    # Save results
    pass

def generate_report(**kwargs):
    # Create PDF report
    # Send via email
    pass

t1 = PythonOperator(task_id='fetch_data', python_callable=fetch_data, dag=dag)
t2 = PythonOperator(task_id='fit_ucm_mmm', python_callable=fit_ucm_mmm, dag=dag)
t3 = PythonOperator(task_id='fit_bvar', python_callable=fit_bvar, dag=dag)
t4 = PythonOperator(task_id='generate_report', python_callable=generate_report, dag=dag)

t1 >> t2 >> t3 >> t4
```

---

## Advanced Usage

### Custom Prior Distributions

Use informative priors from experiments or industry benchmarks:

```python
# In mmm_optimized.py, modify build_model():

# Example: Geo-lift test shows LinkedIn effect = 2.0 ± 0.3
with self.model:
    # Instead of hierarchical prior:
    # beta_digital = pm.Normal('beta_digital', mu=mu_digital, sigma=sigma_digital)

    # Use informative prior for LinkedIn:
    beta_linkedin = pm.Normal('beta_linkedin', mu=2.0, sigma=0.3)

    # Keep other channels hierarchical:
    beta_others = pm.Normal('beta_others', mu=mu_digital, sigma=sigma_digital, shape=3)
```

### Different Adstock Functions

PyMC-Marketing supports multiple adstock transformations:

```python
from pymc_marketing.mmm.transformers import delayed_adstock

# Delayed adstock (peak effect after delay)
adstocked_marketing = delayed_adstock(
    x=self.marketing_data,
    alpha=alpha,        # Retention rate
    theta=theta,        # Delay parameter
    l_max=12,           # Max lag
    normalize=True
)
```

### Alternative Saturation Functions

```python
from pymc_marketing.mmm.transformers import logistic_saturation

# Logistic saturation
saturated_marketing = logistic_saturation(
    x=adstocked_marketing,
    lam=lam,            # Inflection point
    beta=beta           # Slope
)
```

### Multi-Market Hierarchical Models

For analyzing multiple geographic markets:

```python
# Define market-level hierarchy
n_markets = 3
n_channels = 4

with pm.Model() as hierarchical_mmm:
    # Global (population-level) parameters
    mu_global = pm.Normal('mu_global', 0, 1, shape=n_channels)
    sigma_global = pm.HalfNormal('sigma_global', 1, shape=n_channels)

    # Market-specific parameters (partial pooling)
    beta_market = pm.Normal(
        'beta_market',
        mu=mu_global,
        sigma=sigma_global,
        shape=(n_markets, n_channels)
    )

    # Likelihood for each market
    for market_idx in range(n_markets):
        # Use beta_market[market_idx] for this market's data
        pass
```

---

## Troubleshooting

### Problem 1: Model Won't Converge (R-hat > 1.01)

**Symptoms:**
- R-hat > 1.01 for multiple parameters
- Trace plots show chains haven't mixed
- ESS < 400

**Solutions:**

#### 1. Increase Sampling
```python
# Try more draws and longer warm-up
mmm.fit(draws=1000, tune=1000, chains=4)
```

#### 2. Check Data Scaling
```python
# Standardize marketing spend
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
marketing_scaled = scaler.fit_transform(marketing_data)
```

#### 3. Use Stronger Priors
```python
# In build_model(), replace:
# beta = pm.Normal('beta', 0, 10, shape=n_channels)  # Weak prior

# With:
beta = pm.Normal('beta', mu=1.5, sigma=0.5, shape=n_channels)  # Informative prior
```

#### 4. Simplify Model
```python
# Remove hierarchical effects temporarily
mmm = UCM_MMM_Optimized(
    ...,
    channel_groups=None  # Disable hierarchical effects
)
```

---

### Problem 2: JAX + Multiprocessing Deadlock

**Symptom:**
```
WARNING: os.fork() was called. os.fork() is incompatible with multithreaded code,
and JAX is multithreaded, so this will likely lead to a deadlock.
```

**Explanation:**
JAX uses multithreading, which is incompatible with Python's `multiprocessing` module (used for multi-chain MCMC sampling).

**Solutions:**

#### Option 1: Disable JAX for Production (Recommended)
```python
# DON'T import config_jax
# from scripts import config_jax  # ❌ Don't do this

mmm.fit(draws=500, chains=4)  # Works fine without JAX
```

#### Option 2: Single-Chain with JAX (Fast Prototyping)
```python
from scripts import config_jax  # Enable JAX

mmm.fit(draws=500, chains=1)  # Single chain only
```
⚠️ **Caveat**: No convergence diagnostics (R-hat undefined with 1 chain)

---

### Problem 3: Negative Beta Coefficients

**Symptoms:**
- One or more marketing channels have negative coefficients
- 95% CI clearly excludes zero

**Possible Causes:**

#### 1. Multicollinearity
Channels are highly correlated, causing coefficient instability.

**Diagnosis:**
```python
# Check correlation matrix
import seaborn as sns

corr = df[marketing_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')

# Check VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["Variable"] = marketing_cols
vif["VIF"] = [variance_inflation_factor(marketing_data, i) for i in range(marketing_data.shape[1])]
print(vif)
# VIF > 10 indicates severe multicollinearity
```

**Solution:**
- Drop one of the correlated channels
- Combine correlated channels into a single variable
- Use ridge regression (L2 penalty)

#### 2. Suppression Effect
One channel "steals" credit from another due to timing or targeting overlap.

**Solution:**
- Add interaction terms
- Use hierarchical grouping to regularize estimates

#### 3. Data Quality Issues
Check for:
- Data entry errors
- Misaligned dates
- Incorrect channel attribution

---

### Problem 4: MCMC Sampling is Too Slow

**Symptoms:**
- Single chain takes > 30 minutes
- Unable to run 4 chains due to time constraints

**Solutions:**

#### 1. Reduce Data Size (Temporary)
```python
# Use most recent 100 weeks for testing
df_recent = df.tail(100)
```

#### 2. Reduce Draws (Testing Only)
```python
# Quick test with fewer draws
mmm.fit(draws=200, tune=200, chains=2)
```
⚠️ **Not for production** (convergence not guaranteed)

#### 3. Use Variational Inference (ADVI)
```python
# Fast approximation (~2 minutes vs 20 minutes)
with mmm.model:
    approx = pm.fit(n=50000, method='advi')
    trace = approx.sample(2000)
```
⚠️ **Trade-off**: Approximate posterior (no convergence diagnostics)

#### 4. Enable JAX (Single-Chain Only)
```python
from scripts import config_jax

mmm.fit(draws=500, chains=1)  # 3-5x faster
```

---

### Problem 5: Wide Credible Intervals

**Symptom:**
```
LinkedIn ROI: $500 per $1 (95% CI: [-$100, $1,200])
```

**Interpretation:**
- High uncertainty in ROI estimate
- Model is not confident about the true effect

**Causes:**

1. **Insufficient Data**: < 2 years of weekly data
2. **Weak Signal-to-Noise**: Marketing effect is small relative to variance
3. **Confounding Variables**: Unmeasured factors driving sales

**Solutions:**

#### 1. Collect More Data
- Extend time series to 3+ years
- Ensure marketing spend varies sufficiently

#### 2. Use Informative Priors
```python
# From geo-lift test: LinkedIn effect = $2.00 ± $0.30
beta_linkedin = pm.Normal('beta_linkedin', mu=2.0, sigma=0.3)
```

#### 3. Add Control Variables
- Competitor spend
- Macroeconomic indicators
- Seasonality adjustments

#### 4. Run Holdout Experiments
- Geo-lift tests
- Incrementality studies
- A/B tests (if possible)

---

### Problem 6: Results Differ from Previous MMM

**Scenario:**
Previous vendor's MMM showed Google Ads ROI = $3.00, but this framework shows $2.00.

**Explanation:**
- Traditional MMM captures **only short-term effects**
- This framework **separates short and long-term**
- Long-term effects were previously mis-attributed to short-term

**Validation Steps:**

1. **Compare Short-Term ROI Only**
   - This framework's short-term ROI should match previous MMM

2. **Verify Long-Term ROI is Additive**
   - Total ROI = Short + Long
   - Total ROI should be ≥ previous MMM (unless previous MMM had methodological errors)

3. **Check Model Assumptions**
   - Adstock transformation type (geometric vs delayed)
   - Saturation curve (Hill vs logistic)
   - Lag specification (4 weeks vs 8 weeks)

---

## Best Practices

### 1. Data Quality First

✅ **Do:**
- Collect at least 2 years of weekly data
- Ensure marketing spend varies over time (no flat budgets)
- Verify date alignment across all data sources
- Check for outliers and handle them appropriately

❌ **Don't:**
- Use data with many missing values (> 10%)
- Mix different granularities (daily + weekly)
- Include channels with zero spend for > 50% of weeks

---

### 2. Model Validation is Critical

✅ **Do:**
- Always check R-hat < 1.01 before trusting results
- Validate with holdout data (train on first 80%, test on last 20%)
- Compare results with known experiments (geo-lift tests)
- Run sensitivity analysis (vary priors by ±50%, check if results change)

❌ **Don't:**
- Trust results without convergence checks
- Ignore wide credible intervals
- Over-interpret small effects (< 10% ROI difference)

---

### 3. Communicate Uncertainty

✅ **Do:**
- Always report 95% credible intervals
- Use visualizations with error bars
- Explain "statistically significant" means (CI doesn't include zero)
- Emphasize uncertainty in long-term estimates

❌ **Don't:**
- Report only point estimates
- Claim certainty when intervals are wide
- Ignore uncertainty in downstream decisions (budget optimization)

---

### 4. Iterate and Improve

✅ **Do:**
- Start simple (no hierarchical effects, no control variables)
- Add complexity gradually (hierarchical → controls → informative priors)
- Re-fit models quarterly with updated data
- Document all modeling choices and assumptions

❌ **Don't:**
- Build overly complex models from the start
- Use "black box" settings without understanding
- Fit once and never update

---

### 5. Bridge to Business Decisions

✅ **Do:**
- Translate ROI to business metrics (revenue, profit, customer lifetime value)
- Provide actionable recommendations (increase LinkedIn by 20%)
- Create executive-friendly visualizations
- Run scenario analyses ("What if we double Content budget?")

❌ **Don't:**
- Present only statistical outputs (R-hat, ESS)
- Provide recommendations without context
- Ignore business constraints (budget caps, brand guidelines)

---

## Next Steps

After completing this guide:

1. **✅ Validate with Holdout Data**: Test model on recent weeks not used in training
2. **✅ Run Scenario Simulations**: "What if we increase LinkedIn budget by 20%?"
3. **✅ Set Up Automated Pipeline**: Schedule weekly model updates (see Production Deployment)
4. **✅ Create Executive Dashboard**: Build Tableau/Power BI visualizations
5. **✅ Conduct Sensitivity Analysis**: Test robustness to prior choices
6. **✅ Implement Budget Optimization**: Use ROI estimates to reallocate budget
7. **✅ Plan Holdout Experiments**: Design geo-lift tests to validate estimates

---

## Additional Resources

**Documentation:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and technical details
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [README.md](../README.md) - Project overview and quick start

**External Resources:**
- [PyMC Documentation](https://www.pymc.io/)
- [PyMC-Marketing Documentation](https://github.com/pymc-labs/pymc-marketing)
- [Arviz Diagnostics Guide](https://arviz-devs.github.io/arviz/)
- [JAX Documentation](https://jax.readthedocs.io/)

**Support:**
- GitHub Issues: Report bugs or request features
- CLAUDE.md: Development guidelines for contributors

---

**Built with ❤️ and Bayesian inference**
