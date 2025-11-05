# User Guide: Measuring Long-Term Ad Effectiveness

A practical guide to using this framework for Marketing Mix Modeling with short-term and long-term ROI decomposition.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Approach](#understanding-the-approach)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Interpreting Results](#interpreting-results)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Navigate to project directory
cd long-term-ad-effectiveness

# Activate virtual environment (if exists)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

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

---

## Understanding the Approach

### Why Two Models?

Traditional Marketing Mix Models only measure **short-term activation**: the immediate sales spike from advertising. They miss the **long-term brand-building** effects that drive sustained growth.

This framework uses a **two-step approach**:

```
┌─────────────────────────┐
│   Step 1: UCM-MMM       │
│   Short-Term Effects    │
│                         │
│   Marketing → Sales     │
│   (immediate response)  │
└─────────────────────────┘
           ↓
   Extracts Base Sales
           ↓
┌─────────────────────────┐
│   Step 2: BVAR          │
│   Long-Term Effects     │
│                         │
│   Marketing → Brand     │
│   Brand → Base Sales    │
│   (sustained lift)      │
└─────────────────────────┘
```

### Total ROI Formula

```
Total ROI = Short-Term ROI + Long-Term ROI

where:
  Short-Term ROI = Immediate sales lift / Spend
  Long-Term ROI  = Sustained base sales lift / Spend
```

### Example Scenario

**LinkedIn Campaign:**
- Spend: $100,000
- Short-term lift: $80,000 (immediate response)
- Long-term lift: $120,000 (brand building over 12 months)
- **Total return: $200,000 → 2.0x ROI**

Without the long-term component, LinkedIn would appear to have 0.8x ROI and be cut from the budget—a costly mistake!

---

## Step-by-Step Tutorial

### Phase 1: Data Preparation

#### 1.1 Required Data

You need 5 CSV files with weekly data:

| File | Required Columns | Purpose |
|------|-----------------|---------|
| `sales.csv` | Date, revenue, lead_quantity | Business outcomes |
| `marketing_spend.csv` | Date, Channel, Spend | Marketing investments |
| `brand_metrics.csv` | Date, Awareness, Consideration | Brand health |
| `competitor_activity.csv` | Date, Competitor_A_Spend, ... | Market context |
| `macroeconomic_indicators.csv` | Date, GDP_Growth, Unemployment_Rate, ... | External factors |

**Data Requirements:**
- **Granularity:** Weekly (balance of noise vs. responsiveness)
- **Duration:** Minimum 2 years (104 weeks) to capture seasonality
- **Date format:** YYYY-MM-DD
- **Numeric values:** No missing data in sales/spend

#### 1.2 Load and Prepare Data

**Notebook:** `01_Data_Preparation.ipynb`

```python
from scripts.utils import load_data, merge_data, clean_data

# Load all data sources
sales = load_data('data/sales.csv')
marketing = load_data('data/marketing_spend.csv')
brand = load_data('data/brand_metrics.csv')
competitor = load_data('data/competitor_activity.csv')
macro = load_data('data/macroeconomic_indicators.csv')

# Merge on Date
df = merge_data(sales, marketing, brand, competitor, macro)

# Handle missing values
df = clean_data(df)

# Save prepared dataset
df.to_csv('data/prepared_data.csv', index=False)

print(f"✓ Prepared {len(df)} weeks of data")
print(f"✓ Columns: {df.shape[1]}")
```

#### 1.3 Exploratory Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Check for trends
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df.plot(x='Date', y='revenue', ax=axes[0,0], title='Revenue Trend')
df.plot(x='Date', y='Awareness', ax=axes[0,1], title='Brand Awareness')
df.groupby('Channel')['Spend'].sum().plot(kind='bar', ax=axes[1,0], title='Spend by Channel')
df[['Date', 'revenue']].set_index('Date').rolling(4).mean().plot(ax=axes[1,1], title='Revenue (4-week MA)')

plt.tight_layout()
plt.savefig('reports/eda_overview.png')
```

**Checklist:**
- [ ] Revenue shows seasonality or trend
- [ ] Marketing spend varies over time
- [ ] Brand metrics correlate with revenue
- [ ] No extreme outliers

---

### Phase 2: Short-Term Model (UCM-MMM)

#### 2.1 Prepare Data Arrays

**Notebook:** `02_Short_Term_Model.ipynb`

```python
import numpy as np
import pandas as pd
from scripts.mmm import UCM_MMM

# Load prepared data
df = pd.read_csv('data/prepared_data.csv', parse_dates=['Date'])

# Extract sales (dependent variable)
sales = df['revenue'].values

# Extract marketing spend by channel (independent variables)
channels = ['LinkedIn_Spend', 'Google_Spend', 'Content_Spend', 'Events_Spend']
marketing = df[channels].values

print(f"Sales shape: {sales.shape}")       # (208,)
print(f"Marketing shape: {marketing.shape}") # (208, 4)
```

#### 2.2 Build and Fit Model

```python
# Initialize model
mmm = UCM_MMM(sales, marketing)

# Build Bayesian model
mmm.build_model()
print("✓ Model structure defined")

# Fit using MCMC
print("Fitting model (this may take 5-10 minutes)...")
mmm.fit(draws=3000, tune=1500)
print("✓ Model fitted")
```

#### 2.3 Check Convergence

```python
import pymc as pm

summary = mmm.summary()

# Check R-hat (should be < 1.1)
rhat_check = summary['r_hat'] < 1.1
if rhat_check.all():
    print("✓ All parameters converged (R-hat < 1.1)")
else:
    problematic = summary[summary['r_hat'] >= 1.1]
    print("⚠ Convergence issues:")
    print(problematic[['mean', 'r_hat']])
    print("→ Consider increasing draws/tune")

# Check effective sample size (should be > 400)
ess_check = summary['ess_bulk'] > 400
if ess_check.all():
    print("✓ Sufficient effective sample size")
else:
    print("⚠ Low ESS for some parameters")
```

#### 2.4 Extract Short-Term ROI

```python
# Get marketing coefficients (beta)
beta_summary = summary.loc[summary.index.str.startswith('beta')]

print("\nShort-Term Marketing Effects:")
print("=" * 50)
for i, channel in enumerate(channels):
    effect = beta_summary.iloc[i]['mean']
    lower = beta_summary.iloc[i]['hdi_3%']
    upper = beta_summary.iloc[i]['hdi_97%']

    print(f"\n{channel}:")
    print(f"  Effect size: {effect:.2f}")
    print(f"  94% HDI: [{lower:.2f}, {upper:.2f}]")

    # Calculate short-term ROI
    avg_spend = df[channel].mean()
    short_roi = effect / avg_spend
    print(f"  Short-term ROI: {short_roi:.2f}x")
```

#### 2.5 Visualize Short-Term Effects

```python
import arviz as az

# Trace plots for convergence
az.plot_trace(mmm.trace, var_names=['beta', 'alpha'])
plt.savefig('reports/mmm_trace_plots.png')

# Posterior distributions
az.plot_posterior(mmm.trace, var_names=['beta'], hdi_prob=0.94)
plt.savefig('reports/mmm_posteriors.png')

# Effect sizes with uncertainty
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(
    channels,
    beta_summary['mean'],
    yerr=[beta_summary['mean'] - beta_summary['hdi_3%'],
          beta_summary['hdi_97%'] - beta_summary['mean']],
    fmt='o', capsize=5
)
ax.axhline(0, color='red', linestyle='--')
ax.set_ylabel('Marketing Effect Size')
ax.set_title('Short-Term Channel Effects (with 94% HDI)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/short_term_effects.png')
```

#### 2.6 Extract Base Sales

```python
# Extract base sales for BVAR model
# TODO: Once implemented in mmm.py
base_sales = mmm.extract_base_sales()

# For now, approximate as:
# base_sales = sales - marketing_effect_mean
marketing_effect = np.dot(marketing, beta_summary['mean'].values)
base_sales_approx = sales - marketing_effect

# Save for next phase
np.save('data/base_sales.npy', base_sales_approx)
print(f"✓ Base sales extracted: {base_sales_approx.shape}")
```

---

### Phase 3: Long-Term Model (BVAR)

#### 3.1 Prepare BVAR Data

**Notebook:** `03_Long_Term_Model.ipynb`

```python
from scripts.bvar import BVAR

# Load base sales from UCM-MMM
base_sales = np.load('data/base_sales.npy')

# Prepare endogenous variables (mutually influencing)
endog = np.column_stack([
    base_sales,
    df['Awareness'].values,
    df['Consideration'].values
])

# Exogenous variables (marketing spend)
exog = marketing

print(f"Endogenous shape: {endog.shape}")  # (208, 3)
print(f"Exogenous shape: {exog.shape}")    # (208, 4)
```

#### 3.2 Build and Fit BVAR

```python
# Initialize with 4-week lags
bvar = BVAR(endog, exog, lags=4)

# Build model
bvar.build_model()
print("✓ BVAR model defined")

# Fit (may take 10-15 minutes)
print("Fitting BVAR model...")
bvar.fit(draws=3000, tune=1500)
print("✓ BVAR fitted")
```

#### 3.3 Calculate Impulse Response Functions

```python
# Calculate IRF over 52 weeks (1 year)
irf = bvar.calculate_irf(periods=52)

# Extract specific relationships
linkedin_to_sales = irf[('LinkedIn', 'Base Sales')]
linkedin_to_awareness = irf[('LinkedIn', 'Awareness')]
awareness_to_sales = irf[('Awareness', 'Base Sales')]

print("\nImpulse Response Summary:")
print("=" * 50)
print(f"LinkedIn → Base Sales cumulative: {linkedin_to_sales.sum():.2f}")
print(f"LinkedIn → Awareness cumulative: {linkedin_to_awareness.sum():.4f}")
print(f"Awareness → Base Sales cumulative: {awareness_to_sales.sum():.2f}")
```

#### 3.4 Visualize IRFs

```python
# Plot IRF for LinkedIn → Base Sales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(linkedin_to_sales)
axes[0, 0].set_title('LinkedIn → Base Sales')
axes[0, 0].set_xlabel('Weeks')
axes[0, 0].axhline(0, color='red', linestyle='--')

axes[0, 1].plot(linkedin_to_awareness)
axes[0, 1].set_title('LinkedIn → Brand Awareness')
axes[0, 1].set_xlabel('Weeks')

axes[1, 0].plot(awareness_to_sales)
axes[1, 0].set_title('Awareness → Base Sales')
axes[1, 0].set_xlabel('Weeks')

# Cumulative effect
cumulative = np.cumsum(linkedin_to_sales)
axes[1, 1].plot(cumulative)
axes[1, 1].set_title('Cumulative Long-Term Effect')
axes[1, 1].set_xlabel('Weeks')

plt.tight_layout()
plt.savefig('reports/impulse_response_functions.png')
```

#### 3.5 Calculate Long-Term ROI

```python
# Calculate long-term ROI for each channel
long_term_roi = bvar.calculate_long_term_roi(irf)

print("\nLong-Term ROI by Channel:")
print("=" * 50)
for channel, metrics in long_term_roi.items():
    print(f"\n{channel}:")
    print(f"  Total lift: ${metrics['total_lift']:,.0f}")
    print(f"  Long-term ROI: {metrics['roi']:.2f}x")
    print(f"  Peak effect week: {metrics['peak_effect_week']}")
```

---

### Phase 4: Combine Results

#### 4.1 Total ROI Calculation

```python
# Combine short-term and long-term ROI
results = []

for i, channel in enumerate(channels):
    # Short-term ROI from UCM-MMM
    short_effect = beta_summary.iloc[i]['mean']
    avg_spend = df[channels[i]].mean()
    short_roi = short_effect / avg_spend

    # Long-term ROI from BVAR
    long_roi = long_term_roi[channel]['roi']

    # Total ROI
    total_roi = short_roi + long_roi

    results.append({
        'Channel': channel,
        'Avg Weekly Spend': avg_spend,
        'Short-Term ROI': short_roi,
        'Long-Term ROI': long_roi,
        'Total ROI': total_roi,
        'Long-Term %': (long_roi / total_roi * 100)
    })

results_df = pd.DataFrame(results)
print("\nComplete ROI Analysis:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('reports/roi_analysis.csv', index=False)
```

#### 4.2 Visualize Total ROI

```python
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(channels))
width = 0.35

ax.bar(x - width/2, results_df['Short-Term ROI'], width, label='Short-Term ROI')
ax.bar(x + width/2, results_df['Long-Term ROI'], width, label='Long-Term ROI')

ax.axhline(1.0, color='red', linestyle='--', label='Break-even')
ax.set_ylabel('ROI (x)')
ax.set_title('Marketing ROI Decomposition')
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('reports/roi_decomposition.png')
```

---

## Interpreting Results

### Reading Model Outputs

#### 1. Convergence Diagnostics

**R-hat (Gelman-Rubin statistic):**
- `< 1.01`: Excellent convergence
- `< 1.05`: Good convergence
- `< 1.1`: Acceptable
- `> 1.1`: Poor convergence, need more samples

**Effective Sample Size (ESS):**
- `> 1000`: Excellent
- `> 400`: Adequate
- `< 100`: Insufficient, need more draws

#### 2. Marketing Effects

**Beta coefficients:**
- **Positive & significant:** Channel drives sales
- **Near zero:** Weak or no effect
- **Negative:** Potential data issues or suppression effect

**Example interpretation:**
```
Beta for LinkedIn = 2.5 (HDI: [1.8, 3.2])
```
- Every $1 in LinkedIn spend generates $2.50 in immediate sales
- 94% confident the true effect is between $1.80 and $3.20
- Statistically significant (HDI doesn't include 0)

#### 3. Impulse Response Functions

**Shape patterns:**
- **Immediate peak:** Quick brand awareness lift
- **Delayed peak:** Brand consideration takes time
- **Sustained elevation:** Long-lasting brand equity effect
- **Quick decay:** Short-lived impact

**Example interpretation:**
```
LinkedIn IRF to Base Sales:
- Peaks at week 4
- Returns to baseline by week 20
- Cumulative effect: $50,000
```
- Marketing impact fully materializes after 4 weeks
- Effects persist for 5 months
- Total long-term lift is $50k from a one-time $10k investment

#### 4. ROI Comparison

| Channel | Short ROI | Long ROI | Total ROI | Interpretation |
|---------|-----------|----------|-----------|----------------|
| LinkedIn | 0.8x | 1.5x | 2.3x | **Brand builder:** Most value in long-term |
| Google Ads | 2.5x | 0.3x | 2.8x | **Activator:** Immediate response strong |
| Content | 0.2x | 2.0x | 2.2x | **Brand builder:** Pure awareness play |
| Events | 1.5x | 0.8x | 2.3x | **Balanced:** Mix of immediate and sustained |

**Strategic Implications:**
- Don't cut LinkedIn based on short-term ROI alone
- Google Ads good for immediate revenue needs
- Content Marketing undervalued in traditional MMM
- Events provide reliable returns across timeframes

---

## Advanced Usage

### Custom Adstock Functions

Implement different adstock transformations:

```python
# In scripts/mmm.py, modify _adstock method:

def _adstock(self, x, alpha):
    """Geometric adstock with decay"""
    n_periods, n_channels = x.shape
    adstocked = np.zeros_like(x)

    for c in range(n_channels):
        adstocked[0, c] = x[0, c]
        for t in range(1, n_periods):
            adstocked[t, c] = x[t, c] + alpha[c] * adstocked[t-1, c]

    return adstocked
```

### Saturation Curves

Add Hill saturation for diminishing returns:

```python
def hill_saturation(spend, k, s):
    """
    Hill saturation curve
    k: half-saturation point
    s: shape parameter
    """
    return spend**s / (k**s + spend**s)

# In build_model():
k = pm.Gamma('k', alpha=2, beta=1, shape=n_channels)
s = pm.Gamma('s', alpha=2, beta=1, shape=n_channels)
saturated_spend = hill_saturation(adstocked_marketing, k, s)
```

### Hierarchical Models

For many similar channels (e.g., multiple social platforms):

```python
# Group channels
social_channels = ['LinkedIn', 'Facebook', 'Twitter']

# Hierarchical prior
mu_beta = pm.Normal('mu_beta', 0, 1)
sigma_beta = pm.HalfNormal('sigma_beta', 1)
beta_social = pm.Normal('beta_social', mu_beta, sigma_beta, shape=len(social_channels))
```

### Informative Priors from Experiments

Use geo-lift test results as priors:

```python
# From experiment: LinkedIn effect = 2.0 ± 0.3
linkedin_prior_mean = 2.0
linkedin_prior_sd = 0.3

beta_linkedin = pm.Normal(
    'beta_linkedin',
    mu=linkedin_prior_mean,
    sigma=linkedin_prior_sd
)
```

---

## Troubleshooting

### Problem: Model won't converge (R-hat > 1.1)

**Solutions:**
1. Increase sampling:
   ```python
   mmm.fit(draws=10000, tune=5000)
   ```

2. Check data scaling:
   ```python
   # Normalize spend to similar scales
   marketing_scaled = (marketing - marketing.mean(axis=0)) / marketing.std(axis=0)
   ```

3. Use stronger priors:
   ```python
   # More informative prior
   beta = pm.Normal('beta', mu=1.5, sigma=0.5, shape=n_channels)
   ```

### Problem: Negative beta coefficients

**Causes:**
- Multicollinearity between channels
- Suppression effects
- Data quality issues

**Solutions:**
1. Check correlation matrix:
   ```python
   corr = df[channels].corr()
   print(corr)
   # High correlation (>0.8) indicates multicollinearity
   ```

2. Drop highly correlated channels or combine them

3. Use VIF to diagnose:
   ```python
   from statsmodels.stats.outliers_influence import variance_inflation_factor

   vif = [variance_inflation_factor(marketing, i) for i in range(marketing.shape[1])]
   print(dict(zip(channels, vif)))
   # VIF > 10 indicates severe multicollinearity
   ```

### Problem: MCMC sampling is too slow

**Solutions:**
1. Use fewer draws for testing:
   ```python
   mmm.fit(draws=500, tune=500)  # Quick test
   ```

2. Enable parallel sampling (modify source code):
   ```python
   pm.sample(draws=draws, tune=tune, cores=4)
   ```

3. Use ADVI for initial values:
   ```python
   with mmm.model:
       mean_field = pm.fit(method='advi', n=10000)
       trace = pm.sample(draws=2000, start=mean_field.sample())
   ```

### Problem: IRF results seem unrealistic

**Checks:**
1. Verify base sales extraction:
   ```python
   plt.plot(sales, label='Total Sales')
   plt.plot(base_sales, label='Base Sales')
   plt.legend()
   # Base sales should be smoother, lower than total
   ```

2. Check BVAR coefficient signs:
   ```python
   summary = pm.summary(bvar.trace)
   print(summary[summary.index.str.startswith('B')])
   # Marketing → Awareness should be positive
   # Awareness → Sales should be positive
   ```

3. Validate against business intuition:
   - Do IRFs make sense directionally?
   - Are magnitudes reasonable?
   - Does timing align with known customer journey?

### Problem: Results differ from previous MMM

**Explanations:**
- Traditional MMM captures only short-term effects
- This framework separates short and long-term
- Long-term effects were previously attributed incorrectly

**Validation:**
- Compare short-term ROI only (should be similar)
- Long-term ROI is additional value discovered
- Total ROI should be higher than previous estimates

---

## Next Steps

After completing the tutorial:

1. **Validate with holdout data** - Test model on recent weeks not used in training

2. **Run scenario simulations** - "What if we increase LinkedIn budget by 20%?"

3. **Set up automated pipeline** - Schedule weekly model updates

4. **Create executive dashboard** - Build Tableau/Power BI visualizations

5. **Conduct sensitivity analysis** - Test robustness to prior choices

6. **Implement optimization** - Use marginal ROI to reallocate budget

**Resources:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design details
- [API_REFERENCE.md](API_REFERENCE.md) - Complete function documentation
- [CLAUDE.md](../CLAUDE.md) - Development guidelines

**Support:**
- GitHub Issues: Report bugs or request features
- Documentation: Check docs folder for technical details
