import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(path, index=False)
