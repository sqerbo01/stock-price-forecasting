from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def train_model(df: pd.DataFrame):
    """
    Train an XGBoost model to predict next-day closing price.
    """
    # Features
    X = df[['Return', 'MA_5', 'MA_10']]
    
    # Target: next-day closing price
    y = df['Close'].shift(-1)
    
    # Remove last row (NaN in y)
    X = X.iloc[:-1]
    y = y.iloc[:-1]

    # Train-test split (80% train, 20% test, no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, X_test, y_test, y_pred, rmse
