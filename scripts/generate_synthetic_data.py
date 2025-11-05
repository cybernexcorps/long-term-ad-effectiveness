
import pandas as pd
import numpy as np

def generate_synthetic_data():
    # 1. Set up date range
    num_weeks = 4 * 52
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=num_weeks, freq='W'))

    # 2. Generate base data with trend and seasonality
    time = np.arange(num_weeks)
    revenue_trend = 1000000 + time * 10000
    revenue_seasonality = 100000 * np.sin(2 * np.pi * time / 52)
    base_revenue = revenue_trend + revenue_seasonality

    lead_quantity_trend = 500 + time * 5
    lead_quantity_seasonality = 50 * np.sin(2 * np.pi * time / 52)
    base_lead_quantity = lead_quantity_trend + lead_quantity_seasonality

    # 3. Generate marketing spend
    channels = ['LinkedIn', 'Google Ads', 'Content Marketing', 'Events']
    marketing_spend_data = []
    for date in dates:
        for channel in channels:
            if channel == 'LinkedIn':
                spend = np.random.uniform(10000, 20000)
            elif channel == 'Google Ads':
                spend = np.random.uniform(15000, 25000)
            elif channel == 'Content Marketing':
                spend = np.random.uniform(5000, 10000)
            else: # Events
                spend = np.random.choice([0, 50000], p=[0.9, 0.1]) # Sporadic events
            marketing_spend_data.append([date, channel, spend])
    
    marketing_spend_df = pd.DataFrame(marketing_spend_data, columns=['Date', 'Channel', 'Spend'])
    total_spend_per_week = marketing_spend_df.groupby('Date')['Spend'].sum().reset_index()

    # 4. Model the effect of marketing spend
    def adstock(spend, decay):
        return np.convolve(spend, decay ** np.arange(len(spend)))[:len(spend)]

    adstocked_spend = adstock(total_spend_per_week['Spend'], 0.5)
    revenue_from_marketing = adstocked_spend * 2.5 # Simple ROI
    leads_from_marketing = adstocked_spend * 0.05

    final_revenue = base_revenue + revenue_from_marketing + np.random.normal(0, 50000, num_weeks)
    final_lead_quantity = base_lead_quantity + leads_from_marketing + np.random.normal(0, 20, num_weeks)

    sales_df = pd.DataFrame({
        'Date': dates,
        'revenue': final_revenue.astype(int),
        'lead_quantity': final_lead_quantity.astype(int)
    })

    # 5. Generate other data
    brand_metrics_df = pd.DataFrame({
        'Date': dates,
        'Awareness': 0.5 + time * 0.001 + adstocked_spend / 1000000 + np.random.normal(0, 0.02, num_weeks),
        'Consideration': 0.3 + time * 0.0005 + adstocked_spend / 2000000 + np.random.normal(0, 0.01, num_weeks)
    })

    competitor_activity_df = pd.DataFrame({
        'Date': dates,
        'Competitor_A_Spend': np.random.uniform(20000, 40000, num_weeks),
        'Competitor_B_Spend': np.random.uniform(30000, 50000, num_weeks)
    })

    macroeconomic_indicators_df = pd.DataFrame({
        'Date': dates,
        'GDP_Growth': 0.02 + np.sin(2 * np.pi * time / (52*4)) * 0.01 + np.random.normal(0, 0.001, num_weeks),
        'Unemployment_Rate': 0.04 - np.sin(2 * np.pi * time / (52*4)) * 0.005 + np.random.normal(0, 0.001, num_weeks),
        'Consumer_Confidence': 100 + np.sin(2 * np.pi * time / 52) * 5 + np.random.normal(0, 2, num_weeks)
    })

    # 6. Save to CSV
    sales_df.to_csv('d:/Downloads/SynologyDrive/DDVB Analytics/long-term-ad-effectiveness/data/sales.csv', index=False)
    marketing_spend_df.to_csv('d:/Downloads/SynologyDrive/DDVB Analytics/long-term-ad-effectiveness/data/marketing_spend.csv', index=False)
    brand_metrics_df.to_csv('d:/Downloads/SynologyDrive/DDVB Analytics/long-term-ad-effectiveness/data/brand_metrics.csv', index=False)
    competitor_activity_df.to_csv('d:/Downloads/SynologyDrive/DDVB Analytics/long-term-ad-effectiveness/data/competitor_activity.csv', index=False)
    macroeconomic_indicators_df.to_csv('d:/Downloads/SynologyDrive/DDVB Analytics/long-term-ad-effectiveness/data/macroeconomic_indicators.csv', index=False)

if __name__ == '__main__':
    generate_synthetic_data()
    print("Synthetic data generated and saved to the 'data' directory.")
