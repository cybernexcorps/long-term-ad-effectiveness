
import pandas as pd
import numpy as np

# Load the datasets
marketing_spend = pd.read_csv('data/marketing_spend.csv', parse_dates=['Date'])
sales = pd.read_csv('data/sales.csv', parse_dates=['Date'])
brand_metrics = pd.read_csv('data/brand_metrics.csv', parse_dates=['Date'])
competitor_activity = pd.read_csv('data/competitor_activity.csv', parse_dates=['Date'])
macroeconomic_indicators = pd.read_csv('data/macroeconomic_indicators.csv', parse_dates=['Date'])

# Pivot marketing spend data to have channels as columns
marketing_spend_pivot = marketing_spend.pivot_table(index='Date', columns='Channel', values='Spend').reset_index()

# Merge the datasets into a single dataframe
df = pd.merge(sales, marketing_spend_pivot, on='Date', how='left')
df = pd.merge(df, brand_metrics, on='Date', how='left')
df = pd.merge(df, competitor_activity, on='Date', how='left')
df = pd.merge(df, macroeconomic_indicators, on='Date', how='left')

# Handle missing values (e.g., forward fill for brand and macro data)
df[['Awareness', 'Consideration']] = df[['Awareness', 'Consideration']].ffill()
df[['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']] = df[['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']].ffill()

# Fill remaining NaNs with 0 (for spend)
df.fillna(0, inplace=True)

# Save the prepared data
df.to_csv('data/prepared_data.csv', index=False)

print("Data preparation complete. The file 'prepared_data.csv' has been created in the 'data' directory.")
