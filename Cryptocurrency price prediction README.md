# Bitcoin Price Prediction Analysis

## Overview

This Python script utilizes the yfinance API to download and analyze the latest Bitcoin prices from Yahoo Finance. The script visualizes the price changes through a candlestick chart, explores the correlation of features with the closing price, and employs the AutoTS library to predict Bitcoin prices for the next 30 days.

## Prerequisites

1. **Clone Repository:**
   - Clone this repository to your local machine:

     ```bash
     git clone https://github.com/your-username/bitcoin-price-prediction.git
     ```

2. **Install Dependencies:**
   - Ensure you have the required libraries installed:

     ```bash
     pip install pandas yfinance plotly autots
     ```

3. **Run the Script:**
   - Execute the Python script:

     ```bash
     python bitcoin_price_prediction.py
     ```

## Data Collection

- The script uses the yfinance API to download Bitcoin prices for analysis.
- The dataset covers the open, high, low, close, adjusted close prices, and volume.

## Visualizations

- **Candlestick Chart:** Visualizes the change in Bitcoin prices using a candlestick chart.

## Correlation Analysis

- Analyzes the correlation of all columns in the dataset concerning the close price.

## Bitcoin Price Prediction

- Utilizes the AutoTS library to predict Bitcoin prices for the next 30 days.

## Usage

Feel free to customize the script to analyze different timeframes or adapt it for other financial instruments.

## License

This project is licensed under the [MIT License](LICENSE), allowing for broad use, modification, and distribution.

