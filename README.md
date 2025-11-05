# Long-Term Ad Effectiveness Analysis

A **two-step Bayesian framework** for Marketing Mix Modeling (MMM) that measures both short-term activation and long-term brand-building effects of advertising.

## Why This Framework?

Traditional MMM only captures immediate sales response, missing 40-60% of marketing's total value. This framework separates:

- **Short-Term Effects** (UCM-MMM): Immediate sales activation
- **Long-Term Effects** (BVAR): Sustained brand equity lift

**Result:** Complete ROI picture that prevents undervaluing brand marketing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab

# Run notebooks in order (01 → 05)
```

## Documentation

📚 **[Complete Documentation →](docs/README.md)**

- **[User Guide](docs/USER_GUIDE.md)** - Step-by-step tutorial with examples
- **[API Reference](docs/API_REFERENCE.md)** - Complete function documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- **[CLAUDE.md](CLAUDE.md)** - AI assistant development guide

## Project Structure

```
long-term-ad-effectiveness/
├── data/                      # Raw and prepared datasets
│   ├── sales.csv             # Revenue and customer metrics
│   ├── marketing_spend.csv   # Channel spend by week
│   ├── brand_metrics.csv     # Awareness, consideration
│   ├── competitor_activity.csv
│   ├── macroeconomic_indicators.csv
│   └── prepared_data.csv     # Merged and cleaned
│
├── scripts/                   # Core modeling modules
│   ├── mmm.py               # UCM-MMM implementation
│   ├── bvar.py              # BVAR implementation
│   └── utils.py             # Data utilities
│
├── notebooks/                 # Analysis workflow (run in order)
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Short_Term_Model.ipynb
│   ├── 03_Long_Term_Model.ipynb
│   ├── 04_Model_Validation.ipynb
│   └── 05_Insight_Generation.ipynb
│
├── docs/                      # Comprehensive documentation
│   ├── USER_GUIDE.md
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   └── README.md
│
└── reports/                   # Generated outputs
```

## Key Features

✅ **Two-Step Bayesian Approach**
- UCM-MMM for short-term activation
- BVAR for long-term brand effects

✅ **Advanced Transformations**
- Adstock modeling (carryover effects)
- Saturation curves (diminishing returns)
- Impulse Response Functions (long-term dynamics)

✅ **Complete ROI Decomposition**
- Short-term ROI by channel
- Long-term ROI via brand building
- Total ROI = Short + Long

✅ **Production Ready**
- Comprehensive docstrings
- Model validation diagnostics
- Error handling and troubleshooting

## Example Results

| Channel | Short-Term ROI | Long-Term ROI | **Total ROI** |
|---------|----------------|---------------|---------------|
| LinkedIn | 0.8x | 1.5x | **2.3x** ✅ |
| Google Ads | 2.5x | 0.3x | **2.8x** ✅ |
| Content Marketing | 0.2x | 2.0x | **2.2x** ✅ |

*Traditional MMM would cut LinkedIn and Content due to low short-term ROI!*

## Technology Stack

- **Python 3.13+** - Core language
- **PyMC** - Bayesian inference and MCMC
- **PyMC-Marketing** - Pre-built MMM components
- **pandas/numpy** - Data processing
- **matplotlib/seaborn** - Visualization
- **JupyterLab** - Interactive analysis

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/cybernexcorps/long-term-ad-effectiveness.git
cd long-term-ad-effectiveness

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

### Using the Python API

```python
from scripts.utils import load_data, merge_data, clean_data
from scripts.mmm import UCM_MMM
from scripts.bvar import BVAR

# 1. Prepare data
df = load_data('data/sales.csv')
# ... merge and clean

# 2. Short-term model
mmm = UCM_MMM(sales_data, marketing_data)
mmm.build_model()
mmm.fit(draws=3000, tune=1500)

# 3. Long-term model
bvar = BVAR(endog, exog, lags=4)
bvar.build_model()
bvar.fit(draws=3000, tune=1500)

# 4. Calculate total ROI
irf = bvar.calculate_irf(periods=52)
long_term_roi = bvar.calculate_long_term_roi(irf)
```

See [User Guide](docs/USER_GUIDE.md) for detailed tutorial.

## Data Requirements

Minimum data needed:
- **Duration:** 2+ years of weekly data (104+ weeks)
- **Sales:** Revenue or customer acquisition metrics
- **Marketing:** Spend by channel (weekly)
- **Brand Metrics:** Awareness, consideration (monthly surveys OK)
- **Control Variables:** Seasonality, competitor spend, macroeconomics

## Model Validation

The framework includes comprehensive validation:
- **Convergence diagnostics** (R-hat, ESS)
- **Posterior predictive checks**
- **Prior sensitivity analysis**
- **Holdout validation**

See [Model Validation notebook](notebooks/04_Model_Validation.ipynb).

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in research or publications:

```bibtex
@software{long_term_ad_effectiveness,
  title = {Long-Term Ad Effectiveness: Two-Step Bayesian MMM Framework},
  author = {CyberNex Corps},
  year = {2025},
  url = {https://github.com/cybernexcorps/long-term-ad-effectiveness}
}
```

## Support

- **Documentation:** [docs/README.md](docs/README.md)
- **Issues:** [GitHub Issues](https://github.com/cybernexcorps/long-term-ad-effectiveness/issues)
- **Discussions:** GitHub Discussions

## Acknowledgments

Built with:
- [PyMC](https://www.pymc.io) - Probabilistic programming
- [PyMC-Marketing](https://www.pymc-marketing.io) - Marketing analytics
- Inspired by Google's LightweightMMM and Meta's Robyn
