# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Marketing Mix Modeling (MMM)** framework for measuring the long-term effectiveness of advertising using a sophisticated two-step Bayesian approach:
- **UCM-MMM** (Unobserved Components Model): Captures short-term activation effects
- **BVAR** (Bayesian Vector Autoregression): Models long-term brand-building effects

The analysis separates immediate sales impact from sustained brand equity growth, providing a complete view of marketing ROI.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (if exists)
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Launch Jupyter Lab for interactive analysis
jupyter lab

# Generate synthetic data (if needed)
python scripts/generate_synthetic_data.py
```

### Notebook Execution Order
Notebooks must be run sequentially:
1. `01_Data_Preparation.ipynb` - Data loading, cleaning, and merging
2. `02_Short_Term_Model.ipynb` - UCM-MMM for immediate marketing effects
3. `03_Long_Term_Model.ipynb` - BVAR for brand-building effects
4. `04_Model_Validation.ipynb` - Diagnostics and validation
5. `05_Insight_Generation.ipynb` - Strategic recommendations

## Core Architecture

### Two-Step Modeling Framework

**Step 1: Short-Term Model (UCM-MMM)** - [scripts/mmm.py](scripts/mmm.py)
- Decomposes sales into: baseline trend, seasonality, and marketing effects
- Implements **adstock transformation** for advertising carryover effects
- Models **saturation/diminishing returns** using non-linear response curves
- Outputs: Short-term ROI per channel and an evolving base sales time series

**Step 2: Long-Term Model (BVAR)** - [scripts/bvar.py](scripts/bvar.py)
- Takes base sales from UCM-MMM as input
- Models dynamic relationships between marketing spend → brand metrics → base sales
- Uses **Impulse Response Functions (IRFs)** to trace long-term effects over 12+ months
- Outputs: Long-term ROI from sustained brand equity lift

**Total ROI = Short-Term ROI (UCM-MMM) + Long-Term ROI (BVAR)**

### Data Flow

```
Raw Data (data/*.csv)
    ↓
Data Preparation (utils.py: load_data, merge_data, clean_data)
    ↓
UCM-MMM Model (mmm.py: adstock, saturation, decomposition)
    ↓ (extracts base sales)
BVAR Model (bvar.py: VAR coefficients, IRF calculation)
    ↓
Combined Insights (holistic ROI, budget optimization)
```

### Key Data Files
- `marketing_spend.csv`: Channel spend and impressions (weekly granularity)
- `sales.csv`: Revenue and customer acquisition metrics
- `brand_metrics.csv`: Awareness, consideration, purchase intent (survey data)
- `competitor_activity.csv`: Competitor marketing spend
- `macroeconomic_indicators.csv`: GDP, unemployment, consumer confidence
- `prepared_data.csv`: Merged and cleaned dataset (output of data preparation)

## Important Technical Context

### PyMC Bayesian Models
Both core models use **PyMC** for Bayesian inference:
- Models are defined with priors, deterministic transformations, and likelihoods
- Fitting uses MCMC sampling (`pm.sample()`) with configurable draws/tune parameters
- Default: `cores=1` for reproducibility

### Model Validation Requirements
When working with Bayesian models, always check:
- **R-hat < 1.1**: MCMC convergence diagnostic
- **ESS (Effective Sample Size)**: Sufficient independent samples
- **Trace plots**: Visual convergence check
- **Posterior predictive checks**: Model fit quality
- **Prior sensitivity analysis**: Robustness of conclusions

### Adstock and Saturation
- **Adstock**: Models how advertising effects decay over time (carryover)
- **Saturation**: Captures diminishing returns as spend increases (typically Hill function)
- These transformations are critical for accurate ROI estimation

### Data Granularity
- Standard granularity: **weekly** (balances noise reduction with responsiveness)
- Minimum historical period: **2 years** (captures seasonality and long-term trends)

## Code Style Notes
- Data processing uses pandas with forward-fill (`ffill`) for interpolating missing brand/macro data
- Models are class-based with `build_model()`, `fit()`, and output methods
- Visualization uses matplotlib, seaborn, and plotly
