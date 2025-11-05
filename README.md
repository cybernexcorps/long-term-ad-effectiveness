# Long-Term Ad Effectiveness Analysis

This project provides a framework for conducting a Marketing Mix Modeling (MMM) analysis to measure the long-term effectiveness of advertising.

## Project Structure

- `data/`: Contains the raw and prepared data for the analysis.
  - `marketing_spend.csv`: Marketing spend and impressions data.
  - `sales.csv`: Sales and new customer data.
  - `brand_metrics.csv`: Brand health metrics.
  - `competitor_activity.csv`: Competitor spend data.
  - `macroeconomic_indicators.csv`: Macroeconomic data.
- `notebooks/`: Contains Jupyter notebooks for each phase of the analysis.
  - `01_Data_Preparation.ipynb`: Loads, cleans, and prepares the data.
  - `02_Short_Term_Model.ipynb`: Fits the short-term UCM-MMM model.
  - `03_Long_Term_Model.ipynb`: Fits the long-term BVAR model.
  - `04_Model_Validation.ipynb`: Validates the models.
  - `05_Insight_Generation.ipynb`: Generates insights and strategic recommendations.
- `scripts/`: Contains Python scripts for the core modeling components.
  - `mmm.py`: The UCM-MMM model.
  - `bvar.py`: The BVAR model.
  - `utils.py`: Utility functions for data loading and cleaning.
- `reports/`: Will contain the final reports and visualizations.
- `requirements.txt`: A list of the Python packages required to run this project.

## Getting Started

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebooks:**

   Open the `notebooks/` directory and run the notebooks in order, starting with `01_Data_Preparation.ipynb`.

   ```bash
   jupyter lab
   ```
