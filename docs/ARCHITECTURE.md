# Architecture Documentation

**Production-Ready Marketing Mix Modeling Framework**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow Pipeline](#data-flow-pipeline)
3. [Component Architecture](#component-architecture)
4. [Model Mathematics](#model-mathematics)
5. [Technology Stack](#technology-stack)
6. [Performance Architecture](#performance-architecture)
7. [Model Validation Workflow](#model-validation-workflow)
8. [Deployment Architecture](#deployment-architecture)
9. [Scalability Considerations](#scalability-considerations)
10. [Known Limitations](#known-limitations)

---

## System Overview

This framework implements a **two-step Bayesian approach** to Marketing Mix Modeling that separates short-term activation effects from long-term brand-building effects, providing a complete view of marketing ROI.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        Sales[Sales Data<br/>revenue, lead_quantity]
        Marketing[Marketing Spend<br/>4 channels × 208 weeks]
        Brand[Brand Metrics<br/>Awareness, Consideration]
        Competitor[Competitor Activity<br/>Spend tracking]
        Macro[Macroeconomic Indicators<br/>GDP, Unemployment, Confidence]
    end

    subgraph "Data Preparation"
        Load[Load & Parse<br/>utils.load_data]
        Merge[Merge on Date<br/>utils.merge_data]
        Clean[Clean & Interpolate<br/>utils.clean_data]
        Quality[Quality Checks<br/>VIF, ADF tests]
    end

    subgraph "Step 1: Short-Term Effects (Production)"
        MMM[UCM_MMM_Optimized<br/>mmm_optimized.py]
        Adstock[PyMC-Marketing<br/>geometric_adstock<br/>10-50x faster]
        Saturation[Hill Saturation<br/>Diminishing returns]
        Hierarchical[Hierarchical Priors<br/>Digital vs Offline]
        Control[Control Variables<br/>Competitor + Macro]
        ShortROI[Short-Term ROI<br/>With 95% CI]
        BaseSales[Extract Base Sales<br/>Trend without marketing]
    end

    subgraph "Step 2: Long-Term Effects (Production)"
        BVAR[BVAR_Optimized<br/>bvar_optimized.py]
        IRF[Impulse Response Functions<br/>24-week horizon<br/>With uncertainty]
        LongROI[Long-Term ROI<br/>With 95% CI]
    end

    subgraph "Optimization & Validation"
        PPC[Posterior Predictive Checks<br/>MAPE, R², Coverage]
        BudgetOpt[Budget Optimization<br/>scipy.optimize<br/>Diminishing returns]
        TotalROI[Total ROI<br/>Short + Long term<br/>Full uncertainty]
    end

    subgraph "Acceleration (Optional)"
        JAX[JAX Backend<br/>config_jax.py<br/>5-20x speedup<br/>⚠️ Single-chain only]
    end

    Sales --> Load
    Marketing --> Load
    Brand --> Load
    Competitor --> Load
    Macro --> Load

    Load --> Merge
    Merge --> Clean
    Clean --> Quality

    Quality --> MMM
    Quality --> Control
    MMM --> Adstock
    Adstock --> Saturation
    Saturation --> Hierarchical
    Control --> MMM
    Hierarchical --> ShortROI
    MMM --> BaseSales

    BaseSales --> BVAR
    Quality --> BVAR
    BVAR --> IRF
    IRF --> LongROI

    ShortROI --> TotalROI
    LongROI --> TotalROI
    TotalROI --> PPC
    TotalROI --> BudgetOpt

    JAX -.->|Optional| MMM
    JAX -.->|Optional| BVAR
```

### Design Principles

1. **Production-Ready**: Optimized for performance (10-50x faster with PyMC-Marketing)
2. **Uncertainty Quantification**: Full Bayesian inference with 95% credible intervals
3. **Business-Informed**: Priors based on actual spend data and domain knowledge
4. **Modular**: Separate concerns (data prep → short-term → long-term → validation)
5. **Validated**: Comprehensive diagnostics (R-hat < 1.01, posterior predictive checks)
6. **Scalable**: Handles 208+ weeks, 4+ channels, 5+ control variables

---

## Data Flow Pipeline

### Phase 1: Data Preparation

```mermaid
sequenceDiagram
    participant User
    participant Notebook as 01_Data_Preparation
    participant Utils as scripts/utils.py
    participant CSV as data/*.csv

    User->>Notebook: Run notebook
    Notebook->>Utils: load_data(path)
    Utils->>CSV: Read 5 CSV files
    CSV-->>Utils: Raw dataframes
    Utils-->>Notebook: Parsed dataframes

    Notebook->>Utils: merge_data(sales, marketing, brand, competitor, macro)
    Utils->>Utils: Outer join on 'date'
    Utils->>Utils: Forward-fill missing values
    Utils-->>Notebook: Merged dataframe (208 rows)

    Notebook->>Utils: clean_data(df)
    Utils->>Utils: Handle missing values
    Utils->>Utils: Normalize features
    Utils-->>Notebook: Clean dataframe

    Notebook->>Notebook: Data quality checks
    Note over Notebook: VIF < 10 (multicollinearity)<br/>ADF test (stationarity)<br/>Summary statistics

    Notebook->>CSV: Save prepared_data.csv
```

**Key Transformations**:
- Date parsing with `pd.to_datetime()`
- Forward-fill (`ffill`) for brand metrics and macro indicators (interpolate survey data)
- Feature normalization for MCMC stability
- Quality checks: VIF (variance inflation factor), ADF (Augmented Dickey-Fuller test)

---

### Phase 2: Short-Term Model (UCM-MMM Optimized)

```mermaid
sequenceDiagram
    participant User
    participant Notebook as 02_Short_Term_Model
    participant MMM as scripts/mmm_optimized.py
    participant PyMCMkt as pymc_marketing.mmm.transformers
    participant PyMC
    participant JAX as config_jax.py (optional)

    User->>Notebook: Run notebook
    Note over User: Configuration:<br/>4 chains × 500 draws<br/>target_accept=0.95

    Notebook->>MMM: UCM_MMM_Optimized(sales, marketing, control, channels, groups)
    MMM-->>Notebook: Model instance

    opt JAX Acceleration
        Notebook->>JAX: import config_jax
        JAX-->>Notebook: JAX backend enabled<br/>⚠️ Disable for multi-chain
    end

    Notebook->>MMM: build_model()
    MMM->>MMM: Business-informed priors<br/>(median spend, hierarchical)
    MMM->>PyMCMkt: geometric_adstock(x, alpha, l_max=8)
    PyMCMkt-->>MMM: Adstocked spend (vectorized)
    MMM->>MMM: Hill saturation (x^κ / (λ^κ + x^κ))
    MMM->>MMM: Add control variables
    MMM->>MMM: Fourier seasonality (2 pairs)
    MMM->>PyMC: pm.Model()
    PyMC-->>MMM: Compiled model

    Notebook->>MMM: fit(draws=500, tune=500, chains=4)
    MMM->>PyMC: pm.sample(NUTS, target_accept=0.95)
    Note over PyMC: 4 chains in parallel<br/>~10-15 minutes
    PyMC-->>MMM: MCMC trace (InferenceData)

    Notebook->>MMM: summary()
    MMM->>MMM: Convergence diagnostics (R-hat, ESS)
    MMM-->>Notebook: Parameter estimates

    Notebook->>MMM: calculate_short_term_roi()
    MMM->>MMM: Extract posterior samples
    MMM->>MMM: Calculate ROI per chain/draw
    MMM-->>Notebook: ROI with 95% CI

    Notebook->>MMM: get_base_sales()
    MMM->>MMM: Extract trend component
    MMM-->>Notebook: Base sales time series
```

**Key Features**:
- **PyMC-Marketing Integration**: `geometric_adstock()` replaces custom loops (10-50x speedup)
- **Hierarchical Priors**: Digital vs offline channel grouping for partial pooling
- **Business-Informed Priors**: Lambda scaled by median spend (avoid division by zero)
- **Control Variables**: Competitor spend, GDP growth, unemployment, confidence
- **Convergence**: R-hat < 1.01 with 4 chains × 500 draws

---

### Phase 3: Long-Term Model (BVAR Optimized)

```mermaid
sequenceDiagram
    participant User
    participant Notebook as 03_Long_Term_Model
    participant BVAR as scripts/bvar_optimized.py
    participant PyMC
    participant Scipy as scipy.optimize

    User->>Notebook: Run notebook
    Note over Notebook: Uses base_sales from MMM

    Notebook->>BVAR: BVAR_Optimized(endog, exog, lags=2, names)
    Note over Notebook: endog = [Base_Sales, Awareness, Consideration]<br/>exog = [Content, Events, Google, LinkedIn]
    BVAR-->>Notebook: Model instance

    Notebook->>BVAR: build_model()
    BVAR->>BVAR: VAR(2) specification<br/>Lag-specific priors (recent stronger)
    BVAR->>BVAR: Exogenous coefficients (B matrix)
    BVAR->>BVAR: LKJ prior for covariance
    BVAR->>PyMC: pm.Model()
    PyMC-->>BVAR: Compiled model

    Notebook->>BVAR: fit(draws=500, tune=500, chains=4)
    BVAR->>PyMC: pm.sample(NUTS, target_accept=0.95)
    Note over PyMC: 4 chains in parallel<br/>~5-10 minutes
    PyMC-->>BVAR: MCMC trace

    Notebook->>BVAR: calculate_irf(periods=24, credible_interval=0.95)
    BVAR->>BVAR: Sample 100 posterior draws
    loop For each sample
        BVAR->>BVAR: Forward simulate 24 weeks
        BVAR->>BVAR: Apply shock to each marketing channel
    end
    BVAR->>BVAR: Calculate mean and percentiles
    BVAR-->>Notebook: IRF with 95% CI

    Notebook->>BVAR: calculate_long_term_roi(irf, 'Base_Sales')
    BVAR->>BVAR: Sum IRF over 24 weeks
    BVAR->>BVAR: Normalize by average spend
    BVAR-->>Notebook: Long-term ROI with 95% CI

    Notebook->>Scipy: optimize_budget(total_roi, current_spend, budget)
    Scipy->>Scipy: maximize ROI with diminishing returns
    Scipy-->>Notebook: Optimal allocation
```

**Key Features**:
- **VAR(2) Specification**: 2 lags capture short/medium-term dynamics
- **Lag-Specific Priors**: `σ²ₗₐ𝓰₁ = 1.0, σ²ₗₐ𝓰₂ = 2.0` (recent lags have more influence)
- **Uncertainty Quantification**: Sample 100 posterior draws for IRF calculation
- **IRF Horizons**: 24-week forward simulation (6 months)
- **Long-Term ROI**: Sum of cumulative IRF effects on base sales

---

## Component Architecture

### Production Model Classes

```mermaid
classDiagram
    class UCM_MMM_Optimized {
        +sales_data: ndarray
        +marketing_data: ndarray
        +control_data: ndarray
        +marketing_channels: list
        +control_names: list
        +adstock_max_lag: int
        +channel_groups: dict
        +model: pm.Model
        +trace: InferenceData
        +__init__(sales, marketing, control, channels, controls, lag, groups)
        +build_model()
        +fit(draws, tune, chains, target_accept, cores)
        +summary() dict
        +calculate_short_term_roi() dict
        +get_base_sales() ndarray
    }

    class PyMCMarketing {
        <<external>>
        +geometric_adstock(x, alpha, l_max, normalize, axis)
    }

    class BVAR_Optimized {
        +endog: ndarray
        +exog: ndarray
        +lags: int
        +endog_names: list
        +exog_names: list
        +model: pm.Model
        +trace: InferenceData
        +__init__(endog, exog, lags, endog_names, exog_names)
        +build_model()
        +fit(draws, tune, chains, target_accept, cores)
        +summary() dict
        +calculate_irf(periods, shock_size, credible_interval, n_samples) dict
        +calculate_long_term_roi(irf, sales_var_name) dict
    }

    class ConfigJAX {
        <<module>>
        +setup_jax_backend()
        +check_gpu_availability()
        ⚠️ Incompatible with multiprocessing
    }

    class SyntheticDataGenerator {
        +generate_synthetic_data(n_weeks, channels, seed)
        +save_to_csv(data, output_dir)
    }

    UCM_MMM_Optimized --> PyMCMarketing: uses geometric_adstock
    UCM_MMM_Optimized --> ConfigJAX: optional acceleration
    BVAR_Optimized --> ConfigJAX: optional acceleration
    UCM_MMM_Optimized ..> BVAR_Optimized: provides base_sales
```

### Legacy Models (Deprecated)

```mermaid
classDiagram
    class UCM_MMM {
        <<deprecated>>
        +sales_data: ndarray
        +marketing_data: ndarray
        +_adstock(x, alpha) custom loops (slow)
        ⚠️ Use UCM_MMM_Optimized instead
    }

    class BVAR {
        <<deprecated>>
        +endog: ndarray
        +exog: ndarray
        +calculate_irf(periods) without uncertainty
        ⚠️ Use BVAR_Optimized instead
    }
```

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **PyMC-Marketing Integration** | 10-50x speedup from vectorized adstock (vs nested loops) |
| **Hierarchical Effects** | Partial pooling improves estimates for low-spend channels |
| **Control Variables** | Account for competitor activity and macroeconomic factors |
| **Business-Informed Priors** | Lambda scaled by median spend (avoid unrealistic saturation) |
| **VAR(2) Specification** | Balances model complexity vs interpretability |
| **95% Credible Intervals** | Full uncertainty quantification for ROI estimates |
| **4 Chains × 500 Draws** | R-hat < 1.01 convergence with acceptable runtime (~15-25 min) |
| **JAX Backend (Optional)** | 5-20x speedup on CPU, but incompatible with multiprocessing |

---

## Model Mathematics

### Step 1: UCM-MMM Equation (Production)

**Sales Decomposition**:
```
Sales[t] = μ[t] + Marketing_Effect[t] + Control_Effect[t] + Seasonality[t] + ε[t]

where:
  μ[t] = evolving baseline trend (local level model)
  ε[t] ~ Normal(0, σ²)
```

**Marketing Effect (Channel i)**:
```
Marketing_Effect_i[t] = β_i × Saturation(Adstock(Spend_i[t]))
```

**Adstock Transformation (PyMC-Marketing)**:
```
Adstock_i[t] = Σ(l=0 to L_max) α_i^l × Spend_i[t-l] / Σ(l=0 to L_max) α_i^l

where:
  α_i ~ Beta(3, 3)  # Retention rate
  L_max = 8         # Maximum lag (8 weeks)
  Normalization ensures Σ(weights) = 1
```

**Hill Saturation**:
```
Saturated[t] = Adstocked[t]^κ / (λ^κ + Adstocked[t]^κ)

where:
  κ ~ Gamma(1.5, 0.3)  # Shape parameter
  λ ~ Gamma(2.0, 2.0 / median_spend)  # Half-saturation point
```

**Hierarchical Priors**:
```
β_i ~ Normal(μ_group[i], σ_group[i])

where:
  μ_digital ~ Normal(0, 1)
  μ_offline ~ Normal(0, 1)
  σ_digital ~ HalfNormal(1)
  σ_offline ~ HalfNormal(1)
```

**Control Variables**:
```
Control_Effect[t] = Σ(j=1 to J) γ_j × Control_j[t]

where:
  γ_j ~ Normal(0, 1)  # Standardized coefficients
```

**Seasonality**:
```
Seasonality[t] = Σ(k=1 to K) [a_k × sin(2πkt/52) + b_k × cos(2πkt/52)]

where:
  K = 2  # Number of Fourier pairs
  a_k, b_k ~ Normal(0, 1)
```

---

### Step 2: BVAR Equation (Production)

**Vector Autoregression (VAR(2))**:
```
Y[t] = A₁ × Y[t-1] + A₂ × Y[t-2] + B × X[t] + c + ε[t]

where:
  Y[t] = [Base_Sales[t], Awareness[t], Consideration[t]]ᵀ  (n=3 endogenous)
  X[t] = [Spend_Content[t], Spend_Events[t], Spend_Google[t], Spend_LinkedIn[t]]ᵀ  (m=4 exogenous)
  A₁, A₂ ∈ ℝ^(n×n)  # VAR coefficients
  B ∈ ℝ^(n×m)       # Exogenous coefficients
  c ∈ ℝ^n           # Intercept
  ε[t] ~ MVN(0, Σ)  # Error covariance
```

**Priors**:
```
A₁[i,j] ~ Normal(0, σ²_lag1)  where σ²_lag1 = 1.0
A₂[i,j] ~ Normal(0, σ²_lag2)  where σ²_lag2 = 2.0  (lag 2 has wider variance)
B[i,j] ~ Normal(0, 1)
c[i] ~ Normal(0, 10)
Σ ~ LKJ(η=2, σ ~ HalfNormal(1))  # Covariance with regularization
```

**Impulse Response Function**:
```
IRF(h) = ∂Y[t+h] / ∂X[t]

Computation (for horizon h = 0 to 24):
  Y*[0] = B × shock  (initial impact)
  Y*[h] = A₁ × Y*[h-1] + A₂ × Y*[h-2]  (propagation)

Uncertainty:
  Sample N=100 posterior draws
  Calculate IRF for each draw
  Return: mean, 2.5th percentile, 97.5th percentile
```

**Long-Term ROI Calculation**:
```
Long-Term ROI_i = (Σ(h=0 to 24) IRF_Sales,i[h]) / Average(Spend_i)

where:
  IRF_Sales,i[h] = impulse response of Base_Sales to channel i at horizon h
  Units: $ return per $1 spent (over 24 weeks)
```

---

### Total ROI Calculation

```
Total ROI_i = Short-Term ROI_i + Long-Term ROI_i

where:
  Short-Term ROI_i = (β_i × Saturated_Effect_i) / Average(Spend_i)
  Long-Term ROI_i = (Σ IRF_Sales,i) / Average(Spend_i)
```

**Uncertainty Propagation**:
```
For each posterior sample s:
  Total ROI_i[s] = Short-Term ROI_i[s] + Long-Term ROI_i[s]

Report:
  Mean: mean(Total ROI_i)
  95% CI: [percentile(2.5), percentile(97.5)]
```

---

## Technology Stack

### Core Dependencies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.13+ | Primary programming language |
| **Bayesian Inference** | PyMC | 5.18+ | Probabilistic programming, MCMC sampling |
| **MMM Components** | PyMC-Marketing | 0.11+ | geometric_adstock (10-50x speedup) |
| **Acceleration** | JAX | 0.4+ | GPU/CPU acceleration (5-20x speedup) |
| **Data Processing** | pandas | 2.2+ | DataFrames, time series manipulation |
| **Numerical Computation** | numpy | 2.1+ | Arrays, linear algebra |
| **Optimization** | scipy | 1.14+ | Budget optimization (minimize) |
| **Diagnostics** | arviz | 0.20+ | Convergence checks, trace plots |
| **Visualization** | matplotlib | 3.9+ | Static plots |
| **Visualization** | seaborn | 0.13+ | Statistical visualizations |
| **Visualization** | plotly | 5.24+ | Interactive charts |
| **Environment** | JupyterLab | 4.3+ | Interactive notebooks |

### Installation

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: JAX for 5-20x speedup (CPU only)
pip install jax jaxlib

# Optional: JAX with GPU support (CUDA 12)
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Architecture Layers

```
┌─────────────────────────────────────────┐
│         User Interface Layer            │
│  (Jupyter Notebooks, Python Scripts)    │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│       Application Layer                 │
│  UCM_MMM_Optimized, BVAR_Optimized      │
│  (scripts/mmm_optimized.py,             │
│   scripts/bvar_optimized.py)            │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│     Statistical Modeling Layer          │
│  PyMC (Bayesian inference)              │
│  PyMC-Marketing (geometric_adstock)     │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│      Computational Backend              │
│  PyTensor (tensor operations)           │
│  JAX (optional acceleration)            │
│  NumPy/SciPy (numerical computation)    │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Hardware Layer                  │
│  CPU (multicore), GPU (CUDA/ROCm)       │
└─────────────────────────────────────────┘
```

---

## Performance Architecture

### Optimization Strategies

#### 1. PyMC-Marketing Vectorization

**Before (Custom Loops)**:
```python
# Nested loops: O(T × C × L) complexity
for channel_idx in range(n_channels):
    for t in range(n_periods):
        for lag in range(max_lag):
            adstocked[t, channel_idx] += spend[t-lag, channel_idx] * weights[lag]
```
- **Runtime**: ~15 minutes (208 weeks, 4 channels)

**After (PyMC-Marketing)**:
```python
# Vectorized operation: O(T × C)
from pymc_marketing.mmm.transformers import geometric_adstock
adstocked = geometric_adstock(spend, alpha, l_max=8, normalize=True, axis=0)
```
- **Runtime**: ~1 minute (208 weeks, 4 channels)
- **Speedup**: 10-50x depending on data size

#### 2. JAX Backend Acceleration

```python
# Enable JAX backend (config_jax.py)
import pytensor
pytensor.config.mode = 'JAX'
pytensor.config.floatX = 'float64'
```

**Performance Impact**:
| Configuration | Without JAX | With JAX | Speedup |
|--------------|-------------|----------|---------|
| Single-chain (500 draws) | 12 min | 3-5 min | 3-4x |
| Multi-chain (4 × 500 draws) | 15 min | ⚠️ Deadlock | N/A |

**⚠️ Critical Limitation**: JAX is incompatible with `multiprocessing` (used for multi-chain sampling) due to `os.fork()` incompatibility.

**Workaround**:
- Use JAX for single-chain runs (fast prototyping)
- Disable JAX for production multi-chain runs (robust convergence)

#### 3. MCMC Configuration

| Configuration | Draws | Chains | Runtime | Convergence | Use Case |
|--------------|-------|--------|---------|-------------|----------|
| **Quick Test** | 100 | 2 | 2 min | R-hat ~1.05 | Debugging |
| **Standard Test** | 200 | 2 | 5 min | R-hat ~1.04 | Testing |
| **Production** | 500 | 4 | 15-25 min | R-hat < 1.01 | Deployment |
| **JAX Single-Chain** | 500 | 1 | 3-5 min | N/A | Fast estimation |

**Tuning Parameters**:
```python
mmm.fit(
    draws=500,        # Posterior samples per chain
    tune=500,         # Warm-up samples (discarded)
    chains=4,         # Parallel chains for convergence
    target_accept=0.95,  # High acceptance rate (robust sampling)
    cores=4           # Parallel CPU cores
)
```

#### 4. Memory Optimization

**Strategies**:
- Use `float32` instead of `float64` (2x memory reduction, slight precision loss)
- Thin MCMC traces for large models (`pm.sample(..., thin=2)`)
- Disable trace storage for intermediate variables (`pm.Deterministic(..., store=False)`)

**Memory Usage**:
| Configuration | Memory (GB) |
|--------------|-------------|
| Simple test (50 weeks, 100 draws) | ~2 GB |
| Standard test (208 weeks, 200 draws × 2) | ~3 GB |
| Production test (208 weeks, 500 draws × 4) | ~4 GB |

---

## Model Validation Workflow

```mermaid
flowchart TD
    Start[Model Fitted] --> Convergence{Check Convergence}

    Convergence -->|R-hat > 1.01| Refit[Increase draws/tune<br/>OR simplify model<br/>OR stronger priors]
    Convergence -->|R-hat < 1.01| ESS{Check ESS}

    ESS -->|ESS < 1000| Refit
    ESS -->|ESS > 1000| Divergences{Check Divergences}

    Divergences -->|> 1%| Reparameterize[Increase target_accept<br/>OR reparameterize model]
    Divergences -->|< 1%| PPC[Posterior Predictive Check]

    PPC -->|MAPE > 15%| ReviseModel[Revise model specification<br/>OR add variables<br/>OR change priors]
    PPC -->|MAPE < 15%| Coverage{Check 95% CI Coverage}

    Coverage -->|< 90% or > 98%| CalibrateUncertainty[Adjust σ priors<br/>OR check data quality]
    Coverage -->|~95%| PriorSens[Prior Sensitivity Analysis]

    PriorSens -->|Sensitive| InformPriors[Use informative priors<br/>from experiments/industry]
    PriorSens -->|Robust| BusinessLogic{Business Logic Check}

    BusinessLogic -->|Unrealistic ROI| ReviseModel
    BusinessLogic -->|Reasonable| Validated[Model Validated ✓]

    Refit --> Start
    Reparameterize --> Start
    ReviseModel --> Start
    CalibrateUncertainty --> Start
    InformPriors --> Start

    Validated --> Deploy[Deploy to Production]
```

### Validation Checklist

| Check | Threshold | Purpose | Action if Failed |
|-------|-----------|---------|------------------|
| **R-hat** | < 1.01 | MCMC convergence | Increase draws to 1000+, use 4 chains |
| **ESS** | > 1000 | Effective sample size | Increase draws, reduce autocorrelation |
| **Divergences** | < 1% | Posterior geometry | Increase `target_accept` to 0.95+ |
| **MAPE** | < 15% | Prediction accuracy | Add variables, revise model structure |
| **R²** | > 0.80 | Explained variance | Check for missing confounders |
| **95% CI Coverage** | 90-98% | Calibration | Adjust uncertainty priors |
| **Prior Sensitivity** | < 20% change | Robustness | Use informative priors from data |
| **Business Logic** | ROI > 0, < 1000% | Realism | Validate with holdout experiments |

---

## Deployment Architecture

### Development Workflow

```mermaid
graph LR
    subgraph "Local Development"
        Jupyter[Jupyter Notebooks<br/>Interactive analysis]
        Scripts[Python Scripts<br/>mmm_optimized.py<br/>bvar_optimized.py]
        Tests[Test Scripts<br/>test_optimized_enhanced.py]
        Git[Git Repository<br/>Version control]
    end

    subgraph "Production Pipeline (Optional)"
        Schedule[Scheduler<br/>Airflow/Prefect/Cron]
        DataWarehouse[Data Warehouse<br/>Snowflake/BigQuery/Redshift]
        ModelRegistry[Model Registry<br/>MLflow/Weights & Biases]
        Storage[Cloud Storage<br/>S3/GCS/Azure Blob]
    end

    subgraph "Visualization & Reporting"
        BI[BI Dashboard<br/>Tableau/Power BI/Looker]
        Notebooks[Automated Notebooks<br/>Papermill]
        Reports[Reports<br/>PDF/Email/Slack]
    end

    Jupyter --> Scripts
    Scripts --> Tests
    Tests --> Git

    Git --> Schedule
    DataWarehouse --> Schedule
    Schedule --> ModelRegistry
    ModelRegistry --> Storage

    Storage --> BI
    Storage --> Notebooks
    Notebooks --> Reports
    BI --> Reports
```

### Production Deployment Options

#### Option 1: Scheduled Batch Processing

```python
# Example: Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator

def run_mmm_pipeline():
    from scripts.mmm_optimized import UCM_MMM_Optimized
    from scripts.bvar_optimized import BVAR_Optimized

    # Load data from warehouse
    data = fetch_from_warehouse()

    # Fit models
    mmm = UCM_MMM_Optimized(...)
    mmm.fit(draws=500, chains=4)

    bvar = BVAR_Optimized(...)
    bvar.fit(draws=500, chains=4)

    # Save results
    save_to_model_registry(mmm, bvar)

dag = DAG('mmm_weekly_update', schedule_interval='@weekly')
task = PythonOperator(task_id='run_mmm', python_callable=run_mmm_pipeline, dag=dag)
```

#### Option 2: Interactive Dashboard

```python
# Example: Streamlit app
import streamlit as st
from scripts.mmm_optimized import UCM_MMM_Optimized

st.title("Marketing Mix Modeling Dashboard")

# User inputs
budget = st.slider("Total Budget", 10000, 1000000, 100000)
channels = st.multiselect("Channels", ['Content', 'Events', 'Google', 'LinkedIn'])

# Load pre-trained model
mmm = load_model_from_registry()

# Budget optimization
optimal_allocation = optimize_budget(mmm, budget, channels)

# Display results
st.bar_chart(optimal_allocation)
```

#### Option 3: API Service

```python
# Example: FastAPI endpoint
from fastapi import FastAPI
from scripts.mmm_optimized import UCM_MMM_Optimized

app = FastAPI()

@app.post("/predict_roi")
def predict_roi(spend: dict):
    mmm = load_model_from_registry()
    roi = mmm.calculate_short_term_roi()
    return {"roi": roi}

@app.post("/optimize_budget")
def optimize_budget(total_budget: float):
    mmm = load_model_from_registry()
    allocation = optimize_budget(mmm, total_budget)
    return {"allocation": allocation}
```

---

## Scalability Considerations

### Data Volume

**Current Capacity**:
- Weekly granularity: 2-4 years (100-200 weeks)
- Channels: 4-10
- Control variables: 5-10
- Brand metrics: 2-3

**Scaling Strategies**:

| Dimension | Current | Scale To | Strategy |
|-----------|---------|----------|----------|
| **Time Series Length** | 208 weeks | 1000+ weeks | Use mini-batch sampling, variational inference (ADVI) |
| **Number of Channels** | 4 channels | 20+ channels | Hierarchical models with channel clustering |
| **Granularity** | Weekly | Daily | Aggregate to weekly for modeling, disaggregate for reporting |
| **Geographic Markets** | Single market | 10+ markets | Hierarchical multi-market model (partial pooling) |

### Model Complexity

**Computational Bottleneck**: MCMC sampling time scales as:
```
Runtime ≈ O(T × C × D × N)

where:
  T = time periods
  C = number of channels
  D = number of draws
  N = number of chains
```

**Mitigation Strategies**:
1. **Variational Inference (ADVI)**: Fast approximation (~1 minute vs 15 minutes)
   ```python
   with mmm.model:
       approx = pm.fit(n=50000, method='advi')
       trace = approx.sample(2000)
   ```
   - **Pros**: 10-100x faster
   - **Cons**: Approximate posterior, no convergence diagnostics

2. **Mini-Batch Sampling**: For very long time series (1000+ weeks)
   ```python
   # Split data into batches, fit separately, combine results
   ```

3. **Cloud Infrastructure**: Scale horizontally with GPU clusters
   ```bash
   # AWS EC2 p3.2xlarge (1 GPU): ~5x speedup with JAX
   # GCP n1-highmem-8 (8 CPUs): 4 parallel chains
   ```

### Performance Benchmarks

| Configuration | Data Size | Runtime | Hardware | Notes |
|--------------|-----------|---------|----------|-------|
| Quick test | 50 weeks × 4 channels | 2 min | Local CPU (4 cores) | Debugging only |
| Standard test | 208 weeks × 4 channels | 5 min | Local CPU (4 cores) | Testing |
| Production | 208 weeks × 4 channels | 15-25 min | Local CPU (4 cores) | R-hat < 1.01 |
| Large-scale | 500 weeks × 10 channels | 45 min | Cloud CPU (16 cores) | Multi-market |
| GPU-accelerated | 208 weeks × 4 channels | 3-5 min | GPU (JAX, single-chain) | Prototyping |

---

## Known Limitations

### 1. JAX + Multiprocessing Incompatibility ⚠️

**Issue**: JAX backend cannot be used with multiple chains due to `os.fork()` incompatibility.

**Error Message**:
```
WARNING: os.fork() was called. os.fork() is incompatible with multithreaded code,
and JAX is multithreaded, so this will likely lead to a deadlock.
```

**Workaround**:
```python
# Option 1: Single-chain with JAX (fast prototyping)
import config_jax  # Enable JAX
mmm.fit(draws=500, chains=1)  # No multiprocessing

# Option 2: Multi-chain without JAX (production)
# Don't import config_jax
mmm.fit(draws=500, chains=4)  # Robust convergence
```

**Impact**:
- Single-chain: No convergence diagnostics (R-hat undefined)
- Multi-chain: 3-4x slower without JAX

**Reference**: https://github.com/google/jax/issues/1222

---

### 2. Convergence Challenges with Weak Data

**Issue**: High divergences (300+) indicate challenging posterior geometry, common with:
- Short time series (< 100 weeks)
- Low spend channels (sparse signals)
- Highly correlated channels (multicollinearity)

**Solution**:
```python
# Increase MCMC robustness
mmm.fit(
    draws=1000,          # More posterior samples
    tune=1000,           # Longer warm-up
    chains=4,
    target_accept=0.98   # Very conservative sampling (slower)
)

# Use stronger priors
# In build_model():
alpha = pm.Beta("alpha", alpha=5, beta=5)  # Stronger peak at 0.5
lambda_ = pm.Gamma("lambda", alpha=3, beta=3/median_spend)  # Narrower prior
```

---

### 3. Wide Credible Intervals

**Issue**: High uncertainty in ROI estimates (e.g., 95% CI: [-$10, $500]).

**Causes**:
- Insufficient data (< 2 years)
- Weak signal-to-noise ratio
- Confounding variables (unmeasured factors)

**Solution**:
1. Collect more data (longer time series)
2. Use informative priors from prior experiments or industry benchmarks
3. Run hold-out experiments to validate estimates
4. Accept uncertainty as inherent (report ranges, not point estimates)

---

### 4. Synthetic Data Limitations

**Issue**: Results shown in documentation use synthetic data, which may not reflect real-world complexity.

**Considerations**:
- Synthetic data has no measurement error
- Real data has missing values, outliers, structural breaks
- Real-world ROI may be lower due to confounders

**Recommendation**: Replace synthetic data with actual marketing data before production deployment.

---

### 5. Computational Requirements

**Minimum Hardware**:
- CPU: 4 cores (for 4 parallel chains)
- RAM: 8 GB (for 208 weeks × 4 channels)
- Storage: 5 GB (for trace storage)

**Production Recommendations**:
- CPU: 8+ cores (faster parallel sampling)
- RAM: 16+ GB (handle larger models)
- GPU: Optional (CUDA-enabled for JAX acceleration)

---

## File Structure

```
long-term-ad-effectiveness/
├── data/                              # Raw and prepared datasets
│   ├── sales.csv                      # Revenue, lead_quantity (208 weeks)
│   ├── marketing_spend.csv            # 4 channels: Content, Events, Google, LinkedIn
│   ├── brand_metrics.csv              # Awareness, Consideration (survey data)
│   ├── competitor_activity.csv        # Competitor_A_Spend
│   ├── macroeconomic_indicators.csv   # GDP_Growth, Unemployment_Rate, Consumer_Confidence
│   └── prepared_data.csv              # Merged and cleaned (output of 01_Data_Preparation)
│
├── scripts/                           # Core modeling modules
│   ├── mmm_optimized.py               # ⭐ Production UCM-MMM (PyMC-Marketing)
│   ├── bvar_optimized.py              # ⭐ Production BVAR (with uncertainty)
│   ├── config_jax.py                  # JAX backend configuration (optional)
│   ├── mmm.py                         # Legacy UCM-MMM (deprecated)
│   ├── bvar.py                        # Legacy BVAR (deprecated)
│   ├── utils.py                       # Data utilities (load, merge, clean)
│   ├── generate_synthetic_data.py     # Synthetic data generator
│   ├── test_models_simple.py          # Quick test (50 weeks, 100 draws)
│   ├── test_optimized.py              # Standard test (208 weeks, 200 draws × 2)
│   └── test_optimized_enhanced.py     # ⭐ Production test (500 draws × 4, validation)
│
├── notebooks/                         # Interactive analysis workflow
│   ├── 01_Data_Preparation.ipynb      # Load, merge, clean, quality checks
│   ├── 02_Short_Term_Model.ipynb      # UCM-MMM with hierarchical effects
│   ├── 03_Long_Term_Model.ipynb       # BVAR with IRF and long-term ROI
│   ├── 04_Model_Validation.ipynb      # Convergence diagnostics, PPC
│   └── 05_Insight_Generation.ipynb    # Budget optimization, recommendations
│
├── outputs/                           # Generated visualizations
│   ├── irf_plot_optimized.png         # Impulse response functions (24 weeks)
│   ├── long_term_roi_optimized.png    # Long-term ROI by channel
│   ├── total_roi_comparison_optimized.png  # Short vs long-term comparison
│   ├── channel_contribution_timeline.png   # Marketing effects over time
│   ├── posterior_predictive_check.png      # Model fit validation
│   └── budget_optimization.png             # Optimal allocation vs current
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md                # This file (system design)
│   ├── API_REFERENCE.md               # API documentation (classes, methods)
│   ├── USER_GUIDE.md                  # Step-by-step usage guide
│   └── README.md                      # Overview and quick start
│
├── CLAUDE.md                          # Project instructions for Claude Code
├── requirements.txt                   # Python dependencies
└── README.md                          # Main project README
```

---

## Summary

This architecture document describes a **production-ready Marketing Mix Modeling framework** with:

✅ **Performance**: 10-50x speedup with PyMC-Marketing, optional 5-20x with JAX
✅ **Robustness**: 4 chains × 500 draws for R-hat < 1.01 convergence
✅ **Uncertainty**: Full Bayesian inference with 95% credible intervals
✅ **Business-Informed**: Priors based on actual spend data and domain knowledge
✅ **Validated**: Comprehensive diagnostics (convergence, posterior predictive checks, business logic)
✅ **Scalable**: Handles 208+ weeks, 4+ channels, 5+ control variables
✅ **Documented**: Complete API reference, user guide, and architecture docs

**Key Innovation**: Two-step Bayesian approach separating short-term activation (UCM-MMM) from long-term brand-building (BVAR) for holistic ROI measurement.

---

**For More Information**:
- **API Documentation**: See `docs/API_REFERENCE.md`
- **User Guide**: See `docs/USER_GUIDE.md`
- **Quick Start**: See `README.md` in project root

**Built with ❤️ and Bayesian inference**
