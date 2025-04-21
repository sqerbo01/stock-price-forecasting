# Stock Price Forecasting

This project explores multiple strategies for forecasting stock prices and simulating investment strategies using historical data. It combines regression and classification models with technical analysis indicators to assess how well machine learning can outperform the market.

## 📁 Project Structure

- `data/`: Raw CSV files (e.g., from Yahoo Finance).
- `notebooks/`: Jupyter notebooks for exploration, training, and evaluation.
- `src/`: Python modules for data loading, feature engineering, modeling, and technical indicators.
- `plots/`: Exported visualizations of model predictions and trading performance.
- `requirements.txt`: Project dependencies.

## ✅ What’s Implemented

### 🔹 Regression (XGBoost)
- Predicts next-day price using:
  - Daily return
  - Moving averages (5, 10 days)
- Evaluated with RMSE and actual vs predicted plots

### 🔹 Classification (Random Forest, XGBoost, Voting Ensemble)
- Predicts **direction** of price movement (up/down)
- Accuracy measured with:
  - Classification accuracy
  - Simulated trading strategy
  - Comparison to market performance

### 🔹 Technical Indicators
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)

### 📊 Strategy Evaluation
- Simulated strategy return vs market return
- Relative gain/loss in absolute and percentage terms
- Directional accuracy
- Visualization of cumulative returns

## 🚧 What's Next

- Sentiment-based features (news, Twitter)
- LSTM or GRU for temporal dependencies
- Streamlit dashboard for interactive visualization
- Hyperparameter tuning with Optuna

## 🛠 Getting Started

```bash
pip install -r requirements.txt