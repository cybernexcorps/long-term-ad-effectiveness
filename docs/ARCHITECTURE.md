# Architecture Documentation

## System Overview

This framework implements a two-step Bayesian approach to Marketing Mix Modeling that separates short-term activation effects from long-term brand-building effects.

```mermaid
graph TB
    subgraph "Data Sources"
        Sales[Sales Data<br/>revenue, lead_quantity]
        Marketing[Marketing Spend<br/>LinkedIn, Google, Content, Events]
        Brand[Brand Metrics<br/>Awareness, Consideration]
        Competitor[Competitor Activity<br/>Spend tracking]
        Macro[Macroeconomic Indicators<br/>GDP, Unemployment]
    end

    subgraph "Data Preparation"
        Load[Load & Parse<br/>utils.load_data]
        Merge[Merge on Date<br/>utils.merge_data]
        Clean[Clean & Interpolate<br/>utils.clean_data]
    end

    subgraph "Step 1: Short-Term Effects"
        MMM[UCM-MMM Model<br/>mmm.UCM_MMM]
        Adstock[Adstock Transform<br/>Carryover effects]
        Saturation[Saturation Curve<br/>Diminishing returns]
        ShortROI[Short-Term ROI<br/>By channel]
        BaseSales[Extract Base Sales<br/>Trend without marketing]
    end

    subgraph "Step 2: Long-Term Effects"
        BVAR[BVAR Model<br/>bvar.BVAR]
        IRF[Impulse Response<br/>Functions]
        LongROI[Long-Term ROI<br/>Brand building]
    end

    subgraph "Outputs"
        TotalROI[Total ROI<br/>Short + Long term]
        Optimization[Budget Optimization<br/>mROI maximization]
        Forecasts[Scenario Planning<br/>What-if analysis]
    end

    Sales --> Load
    Marketing --> Load
    Brand --> Load
    Competitor --> Load
    Macro --> Load

    Load --> Merge
    Merge --> Clean

    Clean --> MMM
    MMM --> Adstock
    Adstock --> Saturation
    Saturation --> ShortROI
    MMM --> BaseSales

    BaseSales --> BVAR
    Clean --> BVAR
    BVAR --> IRF
    IRF --> LongROI

    ShortROI --> TotalROI
    LongROI --> TotalROI
    TotalROI --> Optimization
    TotalROI --> Forecasts
```

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
    Utils->>CSV: Read CSV files
    CSV-->>Utils: Raw dataframes
    Utils-->>Notebook: Parsed dataframes

    Notebook->>Utils: merge_data(sales, marketing, brand, competitor, macro)
    Utils-->>Notebook: Merged dataframe

    Notebook->>Utils: clean_data(df)
    Utils-->>Notebook: Clean dataframe

    Notebook->>CSV: Save prepared_data.csv
```

### Phase 2: Short-Term Model (UCM-MMM)

```mermaid
sequenceDiagram
    participant User
    participant Notebook as 02_Short_Term_Model
    participant MMM as scripts/mmm.py
    participant PyMC

    User->>Notebook: Run notebook
    Notebook->>MMM: UCM_MMM(sales_data, marketing_data)
    MMM-->>Notebook: Model instance

    Notebook->>MMM: build_model()
    MMM->>MMM: Define priors (alpha, beta)
    MMM->>MMM: Apply adstock transformation
    MMM->>MMM: Define likelihood
    MMM->>PyMC: pm.Model()
    PyMC-->>MMM: Compiled model

    Notebook->>MMM: fit(draws=2000, tune=1000)
    MMM->>PyMC: pm.sample()
    PyMC-->>MMM: MCMC trace

    Notebook->>MMM: summary()
    MMM-->>Notebook: Parameter estimates

    Notebook->>MMM: extract_base_sales()
    MMM-->>Notebook: Base sales time series
```

### Phase 3: Long-Term Model (BVAR)

```mermaid
sequenceDiagram
    participant User
    participant Notebook as 03_Long_Term_Model
    participant BVAR as scripts/bvar.py
    participant PyMC

    User->>Notebook: Run notebook
    Note over Notebook: Uses base_sales from MMM

    Notebook->>BVAR: BVAR(endog, exog, lags=4)
    BVAR-->>Notebook: Model instance

    Notebook->>BVAR: build_model()
    BVAR->>BVAR: Define VAR coefficients (A)
    BVAR->>BVAR: Define exog coefficients (B)
    BVAR->>BVAR: Define covariance (LKJ prior)
    BVAR->>PyMC: pm.Model()
    PyMC-->>BVAR: Compiled model

    Notebook->>BVAR: fit(draws=2000, tune=1000)
    BVAR->>PyMC: pm.sample()
    PyMC-->>BVAR: MCMC trace

    Notebook->>BVAR: calculate_irf(periods=52)
    BVAR-->>Notebook: Impulse response functions

    Notebook->>BVAR: calculate_long_term_roi(irf)
    BVAR-->>Notebook: Long-term ROI by channel
```

## Component Architecture

### UCM-MMM Model Components

```mermaid
classDiagram
    class UCM_MMM {
        +sales_data: ndarray
        +marketing_data: ndarray
        +model: pm.Model
        +trace: InferenceData
        +__init__(sales_data, marketing_data)
        +build_model()
        +fit(draws, tune)
        +summary()
        +extract_base_sales()
        -_adstock(x, alpha)
    }

    class AdstockTransformation {
        <<function>>
        +geometric_adstock(spend, alpha)
        +delayed_adstock(spend, alpha, theta)
    }

    class SaturationCurve {
        <<function>>
        +hill_saturation(spend, k, s)
        +logistic_saturation(spend, L, k)
    }

    UCM_MMM --> AdstockTransformation: uses
    UCM_MMM --> SaturationCurve: applies
```

### BVAR Model Components

```mermaid
classDiagram
    class BVAR {
        +endog: ndarray
        +exog: ndarray
        +lags: int
        +model: pm.Model
        +trace: InferenceData
        +__init__(endog, exog, lags)
        +build_model()
        +fit(draws, tune)
        +calculate_irf(periods)
        +plot_irf(irf, variable_names)
        +calculate_long_term_roi(irf)
        +forecast(horizon)
    }

    class ImpulseResponse {
        <<function>>
        +compute_irf(A_matrix, periods)
        +cumulative_irf(irf)
    }

    class ROICalculator {
        <<function>>
        +calculate_roi(sales_lift, cost)
        +marginal_roi(response_curve)
    }

    BVAR --> ImpulseResponse: calculates
    BVAR --> ROICalculator: uses
```

## Model Mathematics

### UCM-MMM Equation

The UCM-MMM decomposes sales into components:

```
Sales[t] = Baseline[t] + Marketing_Effect[t] + Seasonality[t] + Error[t]

Marketing_Effect[t] = Σ(β_i × Saturation(Adstock(Spend_i[t])))

where:
  - Adstock[t] = Spend[t] + α × Adstock[t-1]  (geometric decay)
  - Saturation(x) = k × x^s / (λ^s + x^s)      (Hill function)
```

### BVAR Equation

The BVAR models dynamic relationships:

```
Y[t] = A₁×Y[t-1] + A₂×Y[t-2] + ... + Aₚ×Y[t-p] + B×X[t] + ε[t]

where:
  Y[t] = [BaseSales[t], Awareness[t], Consideration[t]]ᵀ
  X[t] = [Spend_LinkedIn[t], Spend_Google[t], ...]ᵀ
  ε[t] ~ MVN(0, Σ)
```

### Impulse Response Function

IRF traces the effect of a shock over time:

```
IRF(h) = ∂Y[t+h] / ∂ε[t]

Long-term effect = Σ(IRF(h)) for h = 0 to ∞
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Core Language** | Python 3.13+ | Primary programming language |
| **Statistical Modeling** | PyMC | Bayesian inference and MCMC sampling |
| **Marketing Models** | PyMC-Marketing | Pre-built MMM components |
| **Data Processing** | pandas, numpy | Data manipulation and numerical computation |
| **Visualization** | matplotlib, seaborn, plotly | Charts and interactive plots |
| **Analysis Environment** | JupyterLab | Interactive notebooks |

## Model Validation Workflow

```mermaid
flowchart TD
    Start[Model Fitted] --> Convergence{Check Convergence}

    Convergence -->|R-hat > 1.1| Refit[Increase draws/tune<br/>Reparameterize model]
    Convergence -->|R-hat < 1.1| ESS{Check ESS}

    ESS -->|ESS < 400| Refit
    ESS -->|ESS > 400| PPC[Posterior Predictive Check]

    PPC -->|Poor fit| ReviseModel[Revise model specification]
    PPC -->|Good fit| PriorSens[Prior Sensitivity Analysis]

    PriorSens -->|Sensitive| InformPriors[Use informative priors<br/>from experiments]
    PriorSens -->|Robust| Validated[Model Validated ✓]

    Refit --> Start
    ReviseModel --> Start
    InformPriors --> Start
```

## Deployment Architecture

```mermaid
graph LR
    subgraph "Development"
        Jupyter[Jupyter Notebooks<br/>Interactive analysis]
        Scripts[Python Scripts<br/>Reusable modules]
    end

    subgraph "Production Pipeline"
        Schedule[Airflow/Prefect<br/>Scheduled execution]
        DataWarehouse[Data Warehouse<br/>Snowflake/BigQuery]
        ModelStore[Model Registry<br/>MLflow]
    end

    subgraph "Visualization"
        BI[BI Dashboard<br/>Tableau/Power BI]
        Reports[Automated Reports<br/>PDF/Email]
    end

    Jupyter --> Scripts
    Scripts --> Schedule
    DataWarehouse --> Schedule
    Schedule --> ModelStore
    ModelStore --> BI
    ModelStore --> Reports
```

## File Structure

```
long-term-ad-effectiveness/
├── data/                          # Raw and prepared datasets
│   ├── sales.csv                  # Revenue and customer metrics
│   ├── marketing_spend.csv        # Channel spend by week
│   ├── brand_metrics.csv          # Awareness, consideration
│   ├── competitor_activity.csv    # Competitor spend
│   ├── macroeconomic_indicators.csv
│   └── prepared_data.csv          # Merged and cleaned
│
├── scripts/                       # Core modeling modules
│   ├── mmm.py                    # UCM-MMM implementation
│   ├── bvar.py                   # BVAR implementation
│   ├── utils.py                  # Data utilities
│   └── generate_synthetic_data.py
│
├── notebooks/                     # Analysis workflow
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Short_Term_Model.ipynb
│   ├── 03_Long_Term_Model.ipynb
│   ├── 04_Model_Validation.ipynb
│   └── 05_Insight_Generation.ipynb
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # This file
│   ├── API_REFERENCE.md          # API documentation
│   └── USER_GUIDE.md             # Usage examples
│
└── reports/                      # Generated outputs
    └── (figures, tables, PDFs)
```

## Scalability Considerations

### Data Volume
- **Current**: Weekly granularity, 2-4 years (100-200 weeks)
- **Scalable to**: Daily granularity, 5+ years (1800+ days)
- **Bottleneck**: MCMC sampling time increases with observations

### Model Complexity
- **Current**: 3-4 marketing channels, 2-3 brand metrics
- **Scalable to**: 10+ channels with hierarchical models
- **Approach**: Use partial pooling for similar channels

### Computational Performance
- **MCMC Sampling**: Enable `cores > 1` for parallel chains
- **Large datasets**: Use mini-batch sampling or variational inference (ADVI)
- **Cloud deployment**: Use GPU-accelerated sampling on cloud VMs

## Security & Privacy

- **Data Handling**: All data stays within the organization's infrastructure
- **No external APIs**: Models run entirely offline
- **PII Protection**: No personally identifiable information in datasets
- **Access Control**: Use git repository permissions for code access
