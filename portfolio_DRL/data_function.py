import yfinance as yf
import pandas as pd
import numpy as np
import os
from curl_cffi import requests

def load_tickers_from_file(file_path):
    """
    Load tickers from a text file, one ticker per line.

    Parameters:
    file_path (str): Path to the text file containing tickers.

    Returns:
    list: A list of tickers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]

    return tickers


def download_stock_prices(tickers, start_date, end_date, output_file):
    """
    Download historical stock prices for the given tickers and timeframe.

    Parameters:
    tickers (list): List of stock tickers.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    output_file (str): Path to save the resulting CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the daily price data with tickers as columns and dates as index.
    """
    if not tickers:
        raise ValueError("Ticker list is empty.")

    all_data = {}
    session = requests.Session(impersonate="chrome")
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            data = yf.Ticker(ticker, session=session).history(start=start_date, end=end_date,
                                                       interval='1d',
                                                       actions=False,
                                                       auto_adjust=False,
                                                       raise_errors=True)
            # data = yf.download(ticker, start=start_date, end=end_date, session)
            if 'Close' in data.columns:
                data = data[['Close']]
                data.columns = [ticker]  # Rename column to ticker
                all_data[ticker] = data
            else:
                print(f"No 'Close' data available for {ticker}. Skipping.")
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

    if 'AAPL' not in all_data:
        raise ValueError("AAPL data is required to check the reference length.")

    # Use IYR as the reference length
    reference_length = len(all_data['AAPL'])
    filtered_data = {ticker: data for ticker, data in all_data.items() if len(data) == reference_length}

    if len(filtered_data) < len(all_data):
        removed_tickers = set(all_data.keys()) - set(filtered_data.keys())
        print(f"Removed tickers with insufficient data: {', '.join(removed_tickers)}")

    combined_data = pd.concat(filtered_data.values(), axis=1)
    combined_data.to_csv(output_file)
    print(f"Data saved to {output_file}")
    return combined_data


def calculate_daily_returns(price_data, return_type='arithmetic'):
    """
    Calculate daily returns from price data.

    Parameters:
    price_data (pd.DataFrame): DataFrame containing daily price data for stocks with tickers as columns and dates as index.
    return_type (str): Type of return to calculate ('arithmetic' or 'geometric').

    Returns:
    pd.DataFrame: DataFrame containing daily returns with tickers as columns and dates as index.
    """
    if return_type == 'arithmetic':
        returns = price_data.pct_change()
    elif return_type == 'geometric':
        returns = np.log(price_data / price_data.shift(1))
    else:
        raise ValueError("Invalid return_type. Use 'arithmetic' or 'geometric'.")

    return returns


def read_factor_returns(factor_file, return_data):
    """
    Read factor daily returns and align them with the same dates as return data.

    Parameters:
    factor_file (str): Path to the CSV file containing factor returns.
    return_data (pd.DataFrame): DataFrame containing stock daily returns.

    Returns:
    pd.DataFrame: DataFrame containing factor returns aligned with the stock returns.
    """
    if not os.path.exists(factor_file):
        raise FileNotFoundError(f"The file {factor_file} does not exist.")

    factor_data = pd.read_csv(factor_file, index_col=0, parse_dates=True)
    aligned_factor_data = factor_data.reindex(return_data.index).dropna()
    return aligned_factor_data


