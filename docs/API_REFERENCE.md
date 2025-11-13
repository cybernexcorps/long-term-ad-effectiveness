# API Reference

Complete reference for all modules, classes, and functions in the long-term ad effectiveness framework.

## Table of Contents

- [scripts.mmm](#scriptsmmm) - Short-term marketing effects
- [scripts.bvar](#scriptsbvar) - Long-term brand effects
- [scripts.utils](#scriptsutils) - Data utilities

---

## scripts.mmm

### Module Overview

Implements the Unobserved Components Model for Marketing Mix Modeling (UCM-MMM) to capture short-term activation effects of marketing.

**Import:**
```python
from scripts.mmm import UCM_MMM
```

---

### Class: `UCM_MMM`

Bayesian Marketing Mix Model for short-term marketing effects.

#### Constructor

```python
UCM_MMM(sales_data, marketing_data)
```

**Parameters:**
- `sales_data` (np.ndarray): 1D array of sales/revenue values
  - Shape: `(n_periods,)`
  - Example: `[1000000, 1050000, 1100000, ...]`

- `marketing_data` (np.ndarray): 2D array of marketing spend by channel
  - Shape: `(n_periods, n_channels)`
  - Example: `[[10000, 15000, 5000], [12000, 16000, 6000], ...]`

**Returns:** UCM_MMM instance

**Example:**
```python
import numpy as np
from scripts.mmm import UCM_MMM

# Load your data
sales = np.array([...])  # 100 weeks of sales
marketing = np.array([...])  # 100 weeks x 4 channels

# Initialize model
mmm = UCM_MMM(sales, marketing)
```

---

#### Method: `build_model()`

Build the Bayesian model structure using PyMC.

```python
mmm.build_model()
```

**Parameters:** None

**Returns:** None (stores model in `self.model`)

**Model Components:**
- **Priors:**
  - `alpha`: Beta(2, 2) for adstock decay parameters
  - `beta`: HalfNormal(1) for marketing effect coefficients
  - `baseline`: Normal(sales_mean, sales_std) for base sales
  - `sigma`: HalfNormal(1) for observation noise

- **Transformations:**
  - Adstock: Models carryover effects
  - Saturation: Models diminishing returns (TODO)

**Example:**
```python
mmm = UCM_MMM(sales, marketing)
mmm.build_model()
```

---

#### Method: `fit()`

Fit the model using MCMC sampling.

```python
mmm.fit(draws=1000, tune=1000)
```

**Parameters:**
- `draws` (int, optional): Number of posterior samples to draw
  - Default: 1000
  - Recommended: 2000-5000 for production

- `tune` (int, optional): Number of tuning/warmup samples
  - Default: 1000
  - Higher values for complex models

**Returns:** None (stores trace in `self.trace`)

**Raises:**
- `RuntimeError`: If `build_model()` hasn't been called first

**Example:**
```python
# Quick test
mmm.fit(draws=500, tune=500)

# Production run
mmm.fit(draws=5000, tune=2000)
```

**Performance:**
- Typical runtime: 2-10 minutes depending on data size
- Use `cores=4` in source code for parallel sampling (modify mmm.py)

---

#### Method: `summary()`

Generate summary statistics for model parameters.

```python
summary_df = mmm.summary()
```

**Parameters:** None

**Returns:** pandas.DataFrame with columns:
- `mean`: Posterior mean
- `sd`: Posterior standard deviation
- `hdi_3%`: Lower bound of 94% HDI
- `hdi_97%`: Upper bound of 94% HDI
- `mcse_mean`: Monte Carlo standard error
- `mcse_sd`: MCSE for standard deviation
- `ess_bulk`: Effective sample size (bulk)
- `ess_tail`: Effective sample size (tail)
- `r_hat`: Gelman-Rubin convergence diagnostic

**Example:**
```python
summary = mmm.summary()

# Check convergence
print(summary[['mean', 'r_hat']])

# Extract marketing coefficients
beta_estimates = summary.loc['beta', 'mean']
print(f"Channel effects: {beta_estimates}")
```

**Interpretation:**
- `r_hat < 1.1`: Good convergence
- `r_hat > 1.1`: Need more samples or reparameterization
- `ess_bulk > 400`: Sufficient effective sample size

---

#### Method: `extract_base_sales()`

Extract baseline sales (sales minus marketing effects).

```python
base_sales = mmm.extract_base_sales()
```

**Parameters:** None

**Returns:** np.ndarray
- 1D array of base sales time series
- Shape: `(n_periods,)`
- Used as input for BVAR model

**Status:** TODO - Not yet implemented

**Planned Implementation:**
```python
# Will extract from posterior:
# base_sales = sales - marketing_effect (averaged over MCMC samples)
```

---

#### Private Method: `_adstock()`

Apply adstock transformation to marketing spend.

```python
adstocked = mmm._adstock(x, alpha)
```

**Parameters:**
- `x` (np.ndarray): Marketing spend matrix `(n_periods, n_channels)`
- `alpha` (np.ndarray): Decay parameters `(n_channels,)`, range [0, 1]

**Returns:** np.ndarray
- Adstocked marketing spend with same shape as input

**Status:** TODO - Currently returns input unchanged

**Planned Formula:**
```
Adstock[t] = Spend[t] + alpha × Adstock[t-1]
```

---

## scripts.bvar

### Module Overview

Implements Bayesian Vector Autoregression (BVAR) to model long-term brand-building effects of marketing.

**Import:**
```python
from scripts.bvar import BVAR
```

---

### Class: `BVAR`

Bayesian VAR model for long-term marketing effects through brand equity.

#### Constructor

```python
BVAR(endog, exog, lags)
```

**Parameters:**
- `endog` (np.ndarray): Endogenous variables (interdependent time series)
  - Shape: `(n_periods, n_variables)`
  - Typically: [base_sales, awareness, consideration]
  - Example: `[[1M, 0.45, 0.30], [1.05M, 0.47, 0.31], ...]`

- `exog` (np.ndarray): Exogenous variables (marketing spend)
  - Shape: `(n_periods, n_channels)`
  - Example: `[[10000, 15000, 5000], ...]`

- `lags` (int): Number of autoregressive lags
  - Typical: 2-4 weeks
  - Higher lags capture longer-term dynamics but increase complexity

**Returns:** BVAR instance

**Example:**
```python
import numpy as np
from scripts.bvar import BVAR

# Prepare data
base_sales = mmm.extract_base_sales()
awareness = df['Awareness'].values
consideration = df['Consideration'].values

endog = np.column_stack([base_sales, awareness, consideration])
exog = marketing_spend  # From original data

# Initialize model
bvar = BVAR(endog, exog, lags=4)
```

---

#### Method: `build_model()`

Build the Bayesian VAR model structure.

```python
bvar.build_model()
```

**Parameters:** None

**Returns:** None (stores model in `self.model`)

**Model Equation:**
```
Y[t] = A₁·Y[t-1] + A₂·Y[t-2] + ... + Aₚ·Y[t-p] + B·X[t] + ε[t]
```

**Priors:**
- `A`: Normal(0, 1) - VAR coefficients
- `B`: Normal(0, 1) - Exogenous effects
- `Sigma`: LKJ(2) - Covariance matrix

**Status:** Partially implemented (likelihood TODO)

---

#### Method: `fit()`

Fit the BVAR model using MCMC.

```python
bvar.fit(draws=1000, tune=1000)
```

**Parameters:**
- `draws` (int, optional): Posterior samples
  - Default: 1000
  - Recommended: 2000+ for BVAR (covariance estimation)

- `tune` (int, optional): Warmup samples
  - Default: 1000

**Returns:** None (stores trace in `self.trace`)

**Example:**
```python
bvar.build_model()
bvar.fit(draws=3000, tune=1500)

# Check convergence
summary = pm.summary(bvar.trace)
print(summary[summary['r_hat'] > 1.1])  # Flag problematic parameters
```

---

#### Method: `calculate_irf()`

Calculate Impulse Response Functions.

```python
irf = bvar.calculate_irf(periods=52)
```

**Parameters:**
- `periods` (int, optional): Number of periods to trace impulse
  - Default: 52 (one year)
  - Use 104 for 2-year effects

**Returns:** dict
- Keys: `('shock_variable', 'response_variable')` tuples
- Values: np.ndarray of shape `(periods,)` with IRF values

**Example:**
```python
irf = bvar.calculate_irf(periods=52)

# Effect of LinkedIn spend on base sales
linkedin_to_sales = irf[('linkedin', 'base_sales')]
print(f"Cumulative 1-year effect: {linkedin_to_sales.sum():.2f}")

# Peak effect timing
peak_week = linkedin_to_sales.argmax()
print(f"Peak effect at week: {peak_week}")
```

**Status:** TODO - Not yet implemented

---

#### Method: `plot_irf()`

Visualize Impulse Response Functions.

```python
fig = bvar.plot_irf(irf, variable_names=['Sales', 'Awareness', 'Consideration'])
```

**Parameters:**
- `irf` (dict): Output from `calculate_irf()`
- `variable_names` (list, optional): Labels for variables
  - Default: ['Var1', 'Var2', ...]

**Returns:** matplotlib.figure.Figure

**Example:**
```python
irf = bvar.calculate_irf(52)
fig = bvar.plot_irf(
    irf,
    variable_names=['Base Sales', 'Brand Awareness', 'Consideration']
)
fig.savefig('reports/impulse_response_analysis.png', dpi=300)
plt.show()
```

**Status:** TODO - Not yet implemented

---

#### Method: `calculate_long_term_roi()`

Calculate long-term ROI from IRF.

```python
roi = bvar.calculate_long_term_roi(irf, base_sales_idx=0, cost_per_unit=1.0)
```

**Parameters:**
- `irf` (dict): Impulse response functions
- `base_sales_idx` (int, optional): Index of base sales in endog
  - Default: 0
- `cost_per_unit` (float, optional): Cost per unit of spend
  - Default: 1.0

**Returns:** dict
- Keys: channel names
- Values: dict with:
  - `'total_lift'`: Cumulative sales increase
  - `'roi'`: Return on investment ratio
  - `'peak_effect_week'`: Week of maximum impact

**Example:**
```python
irf = bvar.calculate_irf(52)
roi = bvar.calculate_long_term_roi(irf)

for channel, metrics in roi.items():
    print(f"{channel}:")
    print(f"  Long-term ROI: {metrics['roi']:.2f}x")
    print(f"  Total lift: ${metrics['total_lift']:,.0f}")
    print(f"  Peak at week: {metrics['peak_effect_week']}")
```

**Status:** TODO - Not yet implemented

---

#### Method: `forecast()`

Generate forecasts for endogenous variables.

```python
forecasts = bvar.forecast(horizon=12)
```

**Parameters:**
- `horizon` (int, optional): Periods to forecast
  - Default: 12 weeks

**Returns:** np.ndarray
- Shape: `(horizon, n_variables)`
- Forecasted values with uncertainty

**Status:** TODO - Not yet implemented

---

## scripts.utils

### Module Overview

Utility functions for data loading, merging, and cleaning.

**Import:**
```python
from scripts.utils import load_data, merge_data, clean_data
```

---

### Function: `load_data()`

Load CSV data with automatic date parsing.

```python
df = load_data(path)
```

**Parameters:**
- `path` (str): Path to CSV file

**Returns:** pandas.DataFrame
- Loaded dataframe with 'Date' column as datetime64

**Example:**
```python
sales = load_data('data/sales.csv')
marketing = load_data('data/marketing_spend.csv')
brand = load_data('data/brand_metrics.csv')
```

---

### Function: `merge_data()`

Merge multiple data sources on Date column.

```python
merged_df = merge_data(sales, marketing, brand, competitor, macro)
```

**Parameters:**
- `sales` (pd.DataFrame): Sales metrics with 'Date' column
- `marketing` (pd.DataFrame): Marketing spend with 'Date'
- `brand` (pd.DataFrame): Brand metrics with 'Date'
- `competitor` (pd.DataFrame): Competitor data with 'Date'
- `macro` (pd.DataFrame): Macroeconomic indicators with 'Date'

**Returns:** pandas.DataFrame
- Merged dataset with all features aligned by date
- Missing values as NaN (need cleaning)

**Join Strategy:**
- Uses left join with sales as base
- Preserves all sales periods
- Other data sources may have gaps

**Example:**
```python
# Load all data sources
sales = load_data('data/sales.csv')
marketing = load_data('data/marketing_spend.csv')
brand = load_data('data/brand_metrics.csv')
competitor = load_data('data/competitor_activity.csv')
macro = load_data('data/macroeconomic_indicators.csv')

# Merge
merged = merge_data(sales, marketing, brand, competitor, macro)
print(f"Shape: {merged.shape}")
print(f"Missing values: {merged.isna().sum().sum()}")
```

---

### Function: `clean_data()`

Clean merged dataset by handling missing values.

```python
clean_df = clean_data(df)
```

**Parameters:**
- `df` (pd.DataFrame): Merged dataset from `merge_data()`

**Returns:** pandas.DataFrame
- Cleaned dataset with no missing values
- Ready for modeling

**Cleaning Strategy:**

1. **Brand Metrics** (forward-fill):
   - Awareness, Consideration, Purchase_Intent
   - Rationale: Surveys collected monthly/quarterly

2. **Macroeconomic Indicators** (forward-fill):
   - GDP_Growth, Unemployment_Rate, Consumer_Confidence
   - Rationale: Economic data updated less frequently

3. **Remaining Values** (fill with 0):
   - Primarily marketing spend
   - Rationale: Missing spend = no spend that week

**Example:**
```python
merged = merge_data(sales, marketing, brand, competitor, macro)
clean = clean_data(merged)

# Verify no missing values
assert clean.isna().sum().sum() == 0

# Save prepared data
clean.to_csv('data/prepared_data.csv', index=False)
```

---

## Complete Workflow Example

End-to-end example using all APIs:

```python
import numpy as np
import pandas as pd
from scripts.utils import load_data, merge_data, clean_data
from scripts.mmm import UCM_MMM
from scripts.bvar import BVAR

# 1. Data Preparation
sales = load_data('data/sales.csv')
marketing = load_data('data/marketing_spend.csv')
brand = load_data('data/brand_metrics.csv')
competitor = load_data('data/competitor_activity.csv')
macro = load_data('data/macroeconomic_indicators.csv')

df = merge_data(sales, marketing, brand, competitor, macro)
df = clean_data(df)

# 2. Short-Term Model (UCM-MMM)
sales_array = df['revenue'].values
marketing_array = df[['LinkedIn_Spend', 'Google_Spend', 'Content_Spend']].values

mmm = UCM_MMM(sales_array, marketing_array)
mmm.build_model()
mmm.fit(draws=3000, tune=1500)

summary = mmm.summary()
print("Short-term effects:")
print(summary.loc['beta'])

base_sales = mmm.extract_base_sales()

# 3. Long-Term Model (BVAR)
endog = np.column_stack([
    base_sales,
    df['Awareness'].values,
    df['Consideration'].values
])

bvar = BVAR(endog, marketing_array, lags=4)
bvar.build_model()
bvar.fit(draws=3000, tune=1500)

# 4. Calculate Long-Term ROI
irf = bvar.calculate_irf(periods=52)
long_term_roi = bvar.calculate_long_term_roi(irf)

# 5. Total ROI
print("\nTotal ROI (Short + Long term):")
for channel in ['LinkedIn', 'Google', 'Content']:
    short_roi = summary.loc[f'beta_{channel}', 'mean']
    long_roi = long_term_roi[channel]['roi']
    total_roi = short_roi + long_roi
    print(f"{channel}: {total_roi:.2f}x")

# 6. Visualizations
fig = bvar.plot_irf(irf, variable_names=['Base Sales', 'Awareness', 'Consideration'])
fig.savefig('reports/long_term_effects.png')
```

---

## Error Handling

### Common Errors

**1. Shape Mismatch**
```python
# Error
mmm = UCM_MMM(sales, marketing)  # Different lengths
# Fix
assert len(sales) == len(marketing), "Data must have same length"
```

**2. Missing Model Build**
```python
# Error
mmm.fit()  # Called before build_model()
# Fix
mmm.build_model()
mmm.fit()
```

**3. Convergence Issues**
```python
# Check diagnostics
summary = mmm.summary()
if (summary['r_hat'] > 1.1).any():
    # Increase sampling
    mmm.fit(draws=10000, tune=5000)
```

**4. Data Type Errors**
```python
# Error: pandas Series instead of numpy array
mmm = UCM_MMM(df['sales'], df[channels])
# Fix
mmm = UCM_MMM(df['sales'].values, df[channels].values)
```

---

## Performance Tuning

### Sampling Performance

```python
# Fast (testing)
mmm.fit(draws=500, tune=500)  # ~1-2 minutes

# Standard (development)
mmm.fit(draws=2000, tune=1000)  # ~5-8 minutes

# High quality (production)
mmm.fit(draws=5000, tune=2000)  # ~15-30 minutes
```

### Memory Usage

- UCM-MMM: ~100 MB for 200 weeks, 5 channels
- BVAR: ~200 MB for 200 weeks, 3 endogenous, 5 exogenous
- Increase available RAM for larger datasets

### Parallel Sampling

Modify source code to enable parallel sampling:

```python
# In mmm.py and bvar.py, change:
pm.sample(draws=draws, tune=tune, cores=1)
# To:
pm.sample(draws=draws, tune=tune, cores=4)
```

**Warning:** Parallel sampling may affect reproducibility.
