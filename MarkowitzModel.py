import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization


# stocks we are going to handle
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

start_date = '2020-01-01'
end_date = '2025-01-04'

def download_data():
    stock_data = {}

    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)
        ticker_price_history = ticker_data.history(start=start_date, end=end_date)['Close']
        stock_data[ticker] = ticker_price_history

    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10, 6))
    plt.show()

def calculate_return(data):
    # e.g. data = {'AAPL': [90, 91, 92]}
    # then data.shift(1) = {'AAPL': [NaN, 90, 91]}
    # and data/data.shift(1) = {'AAPL': [NaN, 1.0111, 1.0109]}
    log_return = np.log(data/data.shift(1))
    print(log_return)


if __name__ == '__main__':
    # dataset = download_data()
    # show_data(dataset)