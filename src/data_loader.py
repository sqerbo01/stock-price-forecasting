import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    return df

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(path, index=False)
