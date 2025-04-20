import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic features such as returns and moving averages.
    """
    df['Return'] = df['Adj Close'].pct_change()
    df['MA_5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA_10'] = df['Adj Close'].rolling(window=10).mean()
    df = df.dropna()
    return df
