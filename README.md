# stock-price-forecasting

This project aims to predict future stock prices using historical data. It starts with a simple machine learning model based only on time series features and will be expanded later with sentiment analysis and macroeconomic indicators.

## Project Structure

- `data/`: Contains raw data (e.g., CSV downloaded from Yahoo Finance).
- `notebooks/`: Jupyter notebooks for data exploration and modeling.
- `src/`: Python modules for data loading, feature engineering, and modeling.
- `plots/`: Folder to save generated plots.
- `requirements.txt`: List of dependencies.

## Phase 1 – Historical Data Forecasting

The current phase uses only historical stock data to predict the future adjusted closing price. A simple XGBoost regression model is trained and evaluated.

## Future Plans

- Add technical indicators
- Incorporate sentiment data (e.g., from Twitter and financial news)
- Explore deep learning with LSTM
- Deploy a Streamlit dashboard

## Getting Started

```bash
pip install -r requirements.txt
