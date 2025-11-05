
import pandas as pd

def load_data(path):
    return pd.read_csv(path, parse_dates=['Date'])

def merge_data(sales, marketing, brand, competitor, macro):
    df = pd.merge(sales, marketing, on='Date', how='left')
    df = pd.merge(df, brand, on='Date', how='left')
    df = pd.merge(df, competitor, on='Date', how='left')
    df = pd.merge(df, macro, on='Date', how='left')
    return df

def clean_data(df):
    df[['Awareness', 'Consideration', 'Purchase_Intent']] = df[['Awareness', 'Consideration', 'Purchase_Intent']].fillna(method='ffill')
    df[['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']] = df[['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']].fillna(method='ffill')
    df.fillna(0, inplace=True)
    return df
