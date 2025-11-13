# Documentation

Comprehensive documentation for the Long-Term Ad Effectiveness MMM framework.

## Documentation Structure

```
docs/
├── README.md           # This file - documentation overview
├── ARCHITECTURE.md     # System architecture and design
├── API_REFERENCE.md    # Complete API documentation
└── USER_GUIDE.md       # Step-by-step tutorials
```

## Quick Links

### For New Users
Start here to understand the framework and run your first analysis:
- **[User Guide](USER_GUIDE.md)** - Practical tutorials with examples
  - Quick Start
  - Step-by-step walkthrough
  - Interpreting results
  - Troubleshooting

### For Developers
Technical documentation for contributing or extending the framework:
- **[Architecture Documentation](ARCHITECTURE.md)** - System design
  - Data flow diagrams
  - Component architecture
  - Model mathematics
  - Technology stack

- **[API Reference](API_REFERENCE.md)** - Complete function documentation
  - `scripts.mmm` - UCM-MMM model
  - `scripts.bvar` - BVAR model
  - `scripts.utils` - Data utilities
  - Usage examples and error handling

### For Data Scientists
Understanding the methodology:
- **[Long-Term Ad Effectiveness White Paper](../long-term-ad-effectiveness.md)** - Theoretical foundation
  - Two-step Bayesian approach
  - UCM-MMM methodology
  - BVAR and Impulse Response Functions
  - Industry best practices

## What is This Framework?

This framework implements a **two-step Bayesian approach** to Marketing Mix Modeling that measures both:

1. **Short-Term Activation Effects** (UCM-MMM)
   - Immediate sales response to marketing
   - Adstock transformation for carryover effects
   - Saturation curves for diminishing returns

2. **Long-Term Brand-Building Effects** (BVAR)
   - Sustained base sales lift from brand equity
   - Dynamic relationships through brand metrics
   - Impulse Response Functions over 12+ months

### Why This Matters

Traditional MMM only captures short-term effects, leading to:
- ❌ Undervaluation of brand marketing
- ❌ Over-optimization for performance channels
- ❌ Budget cuts to channels with long-term value

This framework provides:
- ✅ Complete ROI picture (short + long term)
- ✅ Fair evaluation of all channels
- ✅ Strategic budget allocation

### Example Results

| Channel | Short-Term ROI | Long-Term ROI | **Total ROI** |
|---------|----------------|---------------|---------------|
| LinkedIn | 0.8x | 1.5x | **2.3x** |
| Google Ads | 2.5x | 0.3x | **2.8x** |
| Content | 0.2x | 2.0x | **2.2x** |
| Events | 1.5x | 0.8x | **2.3x** |

*Without long-term measurement, LinkedIn and Content would be cut from the budget!*

## Getting Started

### Prerequisites

- Python 3.8+
- 2+ years of weekly marketing and sales data
- Basic understanding of statistics and marketing

### Installation

```bash
# Clone repository
git clone https://github.com/cybernexcorps/long-term-ad-effectiveness.git
cd long-term-ad-effectiveness

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

### First Analysis

Run the notebooks in order:

1. **[01_Data_Preparation.ipynb](../notebooks/01_Data_Preparation.ipynb)**
   - Load and merge data sources
   - Clean and validate data
   - Exploratory data analysis

2. **[02_Short_Term_Model.ipynb](../notebooks/02_Short_Term_Model.ipynb)**
   - Build UCM-MMM model
   - Fit using MCMC sampling
   - Extract short-term ROI

3. **[03_Long_Term_Model.ipynb](../notebooks/03_Long_Term_Model.ipynb)**
   - Build BVAR model
   - Calculate Impulse Response Functions
   - Measure long-term ROI

4. **[04_Model_Validation.ipynb](../notebooks/04_Model_Validation.ipynb)**
   - Check convergence diagnostics
   - Validate assumptions
   - Sensitivity analysis

5. **[05_Insight_Generation.ipynb](../notebooks/05_Insight_Generation.ipynb)**
   - Combine short and long-term results
   - Budget optimization recommendations
   - Scenario planning

## Documentation by Use Case

### "I want to run the analysis"
→ Start with [USER_GUIDE.md](USER_GUIDE.md)

### "I need to understand a specific function"
→ Check [API_REFERENCE.md](API_REFERENCE.md)

### "I want to modify or extend the code"
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) first

### "I need to explain this to stakeholders"
→ Use diagrams from [ARCHITECTURE.md](ARCHITECTURE.md) and results from [USER_GUIDE.md](USER_GUIDE.md)

### "I'm getting errors"
→ See Troubleshooting section in [USER_GUIDE.md](USER_GUIDE.md#troubleshooting)

## Key Concepts

### UCM-MMM (Unobserved Components Model)
Decomposes sales into:
- **Baseline trend** - Underlying growth
- **Seasonality** - Predictable patterns
- **Marketing effects** - Incremental lift from advertising

Formula:
```
Sales[t] = Baseline[t] + Σ(β_i × Adstock(Spend_i[t])) + Seasonality[t] + ε[t]
```

### BVAR (Bayesian Vector Autoregression)
Models dynamic relationships:
- Marketing Spend → Brand Awareness
- Brand Awareness → Consideration
- Consideration → Base Sales

Formula:
```
Y[t] = A₁·Y[t-1] + ... + Aₚ·Y[t-p] + B·X[t] + ε[t]
```

### Impulse Response Function (IRF)
Traces the effect of a one-time marketing investment over time:
- Week 0: Initial awareness spike
- Weeks 1-4: Consideration builds
- Weeks 5-52: Sustained base sales lift

## Technology Stack

| Component | Tool | Documentation |
|-----------|------|---------------|
| Core Language | Python 3.13+ | [python.org](https://python.org) |
| Bayesian Modeling | PyMC | [pymc.io](https://www.pymc.io) |
| Marketing Models | PyMC-Marketing | [pymc-marketing.io](https://www.pymc-marketing.io) |
| Data Processing | pandas, numpy | [pandas.pydata.org](https://pandas.pydata.org) |
| Visualization | matplotlib, seaborn | [matplotlib.org](https://matplotlib.org) |
| Notebooks | JupyterLab | [jupyter.org](https://jupyter.org) |

## Contributing to Documentation

### Adding New Documentation

1. Follow existing structure and tone
2. Include practical examples
3. Test all code snippets
4. Use Mermaid for diagrams
5. Link between related sections

### Documentation Standards

- **Code examples:** Always runnable
- **Mermaid diagrams:** Use for complex flows
- **Cross-references:** Link to related docs
- **Versioning:** Update when APIs change

### Building API Docs Automatically

To generate API documentation from docstrings:

```bash
# Install documentation tools
pip install sphinx sphinx-rtd-theme

# Generate HTML docs
cd docs
sphinx-build -b html . _build
```

## Common Questions

**Q: How long does model fitting take?**
A: 5-15 minutes for UCM-MMM, 10-20 minutes for BVAR with standard data sizes.

**Q: How much data do I need?**
A: Minimum 2 years (104 weeks) of weekly data. More is better for capturing seasonality.

**Q: Can I use daily data?**
A: Yes, but weekly is recommended to balance noise reduction with responsiveness.

**Q: What if I don't have brand metrics?**
A: You can still run UCM-MMM for short-term effects. Consider adding brand tracking surveys.

**Q: How do I know if results are valid?**
A: Check convergence diagnostics (R-hat < 1.1), validate against holdout data, and compare with business intuition.

**Q: Can this handle non-linear effects?**
A: Yes, via saturation curves (Hill function) for diminishing returns and adstock for carryover.

## Support

- **Issues:** [GitHub Issues](https://github.com/cybernexcorps/long-term-ad-effectiveness/issues)
- **Discussions:** Use GitHub Discussions for questions
- **Documentation Updates:** Submit PRs to improve docs

## Version History

- **v1.0** (2025-01) - Initial release with complete documentation
  - UCM-MMM implementation
  - BVAR implementation
  - Comprehensive user guide
  - Architecture documentation
  - API reference

## Additional Resources

### External Resources

- [PyMC Documentation](https://www.pymc.io/projects/docs/en/stable/)
- [Marketing Mix Modeling Guide](https://www.pymc-marketing.io/en/stable/guide/mmm/mmm_example.html)
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

### Academic Papers

- Hanssens et al. (2001) - *Market Response Models*
- Naik & Raman (2003) - *Understanding the Impact of Synergy in Multimedia Communications*
- Jin et al. (2017) - *Bayesian Methods for Media Mix Modeling*

### Industry Best Practices

- [Google's Media Mix Model Guide](https://github.com/google/lightweight_mmm)
- [Meta's Robyn MMM](https://github.com/facebookexperimental/Robyn)
- [PyMC-Marketing Examples](https://www.pymc-marketing.io/en/stable/notebooks/index.html)

---

**Last Updated:** 2025-01-05
**Maintained By:** Claude Code Documentation Generator
**License:** See repository LICENSE file
