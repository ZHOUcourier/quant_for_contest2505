import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np
import calendar


# Function to read all TRD_Dalyr files and combine them
def read_and_combine_data(file_pattern='TRD_Dalyr*.csv'):
    """
    Read all TRD_Dalyr CSV files matching the pattern and combine them into one DataFrame.
    """
    all_files = glob.glob(file_pattern)

    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    print(f"Reading {len(all_files)} files: {all_files}")

    # Read and combine all files, ensuring Stkcd is read as string to preserve leading zeros
    dfs = []
    for file in all_files:
        df = pd.read_csv(file, dtype={'Stkcd': str})
        dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates if any
    combined_df = combined_df.drop_duplicates()

    # Ensure stock codes are formatted properly with leading zeros (6 digits)
    combined_df['Stkcd'] = combined_df['Stkcd'].str.zfill(6)

    return combined_df


# Function to create daily stock returns dataframe
def create_daily_stock_returns(df):
    """
    Create a dataframe with dates as index, stock codes as columns,
    and daily returns as values.
    """
    # Convert date to datetime format
    df['Trddt'] = pd.to_datetime(df['Trddt'])

    # Ensure stock codes are strings with proper formatting
    df['Stkcd'] = df['Stkcd'].astype(str).str.zfill(6)

    # Use Dretwd (returns with dividends reinvested) as the return metric
    # Create a pivot table with dates as index and stock codes as columns
    return_df = df.pivot(index='Trddt', columns='Stkcd', values='Dretwd')

    # Sort by date
    return_df = return_df.sort_index()

    return return_df


# Function to calculate market returns (equal-weighted average of all stocks)
def calculate_market_returns(daily_returns_df):
    """
    Calculate market returns as the equal-weighted average of all stock returns.
    """
    # Calculate the mean of all stock returns for each day
    market_returns = daily_returns_df.mean(axis=1)

    # Create a dataframe with market returns
    market_df = pd.DataFrame(market_returns, columns=['market_return'])

    return market_df


# Function to calculate monthly stock returns
def calculate_monthly_returns(daily_returns_df):
    """
    Calculate monthly returns from daily returns.
    Uses the last trading day of each month.
    """
    # Make a copy to avoid modifying the original dataframe
    daily_df = daily_returns_df.copy()

    # Get the stock code columns (all columns in the dataframe)
    stock_columns = daily_df.columns.tolist()

    # Ensure the index is sorted
    daily_df = daily_df.sort_index()

    # Extract year and month from the index
    daily_df['year'] = daily_df.index.year
    daily_df['month'] = daily_df.index.month

    # Group by year and month, and get the last trading day of each month
    # Use dictionary comprehension to compound returns for each stock code
    monthly_returns = daily_df.groupby(['year', 'month']).apply(
        lambda x: pd.Series({
            'date': x.index[-1],  # Last trading day of the month
            **{col: (1 + x[col]).prod() - 1 for col in stock_columns}  # Compound returns
        })
    )

    # Reset the index and set date as the index
    monthly_returns = monthly_returns.reset_index()
    monthly_returns = monthly_returns.set_index('date')

    # Drop the year and month columns
    monthly_returns = monthly_returns.drop(columns=['year', 'month'])

    return monthly_returns


# Main function to process all data
def process_stock_data(input_pattern='TRD_Dalyr*.csv',
                       output_dir='processed_data'):
    """
    Process TRD_Dalyr files to create the three required output files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read and combine all data
    print("Reading and combining data...")
    combined_data = read_and_combine_data(input_pattern)

    # Create daily stock returns
    print("Creating daily stock returns...")
    daily_returns = create_daily_stock_returns(combined_data)
    daily_returns.to_csv(os.path.join(output_dir, 'stock_returns.csv'))

    # Create market returns
    print("Calculating market returns...")
    market_returns = calculate_market_returns(daily_returns)
    market_returns.to_csv(os.path.join(output_dir, 'market_returns.csv'))

    # Create monthly stock returns
    print("Calculating monthly stock returns...")
    monthly_returns = calculate_monthly_returns(daily_returns)
    monthly_returns.to_csv(os.path.join(output_dir, 'stock_monthly_returns.csv'))

    print(f"All data processed and saved to {output_dir}/")
    print(f"Example of stock codes in output: {', '.join(daily_returns.columns[:5])}")


# Example usage:
if __name__ == "__main__":
    process_stock_data(input_pattern='TRD_Dalyr*.csv', output_dir='processed_data')