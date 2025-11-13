"""
Data utility functions for Marketing Mix Modeling.

This module provides utilities for loading, merging, and cleaning data from
multiple sources required for MMM analysis.
"""

import pandas as pd


def load_data(path):
    """
    Load CSV data with date parsing.

    Args:
        path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataframe with 'Date' column parsed as datetime

    Example:
        >>> sales_df = load_data('data/sales.csv')
        >>> print(sales_df['Date'].dtype)
        datetime64[ns]
    """
    return pd.read_csv(path, parse_dates=['Date'])


def merge_data(sales, marketing, brand, competitor, macro):
    """
    Merge multiple data sources on Date column.

    Combines sales, marketing, brand metrics, competitor activity, and
    macroeconomic indicators into a single dataset for MMM analysis.

    Args:
        sales (pd.DataFrame): Sales and customer data with 'Date' column
        marketing (pd.DataFrame): Marketing spend by channel with 'Date' column
        brand (pd.DataFrame): Brand health metrics with 'Date' column
        competitor (pd.DataFrame): Competitor activity data with 'Date' column
        macro (pd.DataFrame): Macroeconomic indicators with 'Date' column

    Returns:
        pd.DataFrame: Merged dataset with all features aligned by date

    Example:
        >>> merged_df = merge_data(sales, marketing, brand, competitor, macro)
        >>> print(merged_df.columns)
        Index(['Date', 'revenue', 'lead_quantity', 'Channel', 'Spend',
               'Awareness', 'Consideration', 'Competitor_A_Spend', ...])

    Note:
        Uses left join with sales as base to preserve all sales periods.
        Missing values in other datasets will be NaN and need cleaning.
    """
    df = pd.merge(sales, marketing, on='Date', how='left')
    df = pd.merge(df, brand, on='Date', how='left')
    df = pd.merge(df, competitor, on='Date', how='left')
    df = pd.merge(df, macro, on='Date', how='left')
    return df


def clean_data(df):
    """
    Clean merged dataset by handling missing values.

    Applies forward-fill for survey and macroeconomic metrics (which are
    typically measured less frequently), and fills remaining missing values
    with zero (appropriate for spend data).

    Args:
        df (pd.DataFrame): Merged dataset from merge_data()

    Returns:
        pd.DataFrame: Cleaned dataset ready for modeling

    Data Cleaning Steps:
        1. Forward-fill brand metrics (Awareness, Consideration, Purchase_Intent)
           - Brand surveys are often monthly/quarterly
        2. Forward-fill macroeconomic indicators (GDP, Unemployment, Consumer_Confidence)
           - Economic data has lower frequency than sales
        3. Fill remaining NaN with 0
           - Missing marketing spend means no spend that week

    Example:
        >>> merged_df = merge_data(sales, marketing, brand, competitor, macro)
        >>> clean_df = clean_data(merged_df)
        >>> assert clean_df.isna().sum().sum() == 0  # No missing values
    """
    # Forward-fill brand metrics (surveys collected less frequently)
    df[['Awareness', 'Consideration', 'Purchase_Intent']] = \
        df[['Awareness', 'Consideration', 'Purchase_Intent']].fillna(method='ffill')

    # Forward-fill macroeconomic indicators (lower frequency data)
    df[['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']] = \
        df[['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']].fillna(method='ffill')

    # Fill remaining missing values with 0 (primarily for spend data)
    df.fillna(0, inplace=True)

    return df
