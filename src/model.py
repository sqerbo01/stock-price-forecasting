from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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



def train_model_with_tuning(df: pd.DataFrame):
    """
    Train an XGBoost model with hyperparameter tuning using GridSearchCV.
    """
    X = df[['Return', 'MA_5', 'MA_10']]
    y = df['Close'].shift(-1)
    df = df.dropna()
    X = X.iloc[:-1]
    y = y.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=XGBRegressor(),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return best_model, X_test, y_test, y_pred, rmse, grid_search.best_params_



def train_classification_model(df: pd.DataFrame):
    df = df.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    feature_cols = ['Return', 'MA_5', 'MA_10', 'RSI', 'MACD', 'MACD_Signal']
    X = df[feature_cols]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, X_test, y_test, y_pred, acc
