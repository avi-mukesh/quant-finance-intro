import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

class StockData:
    def __init__(self, ticker, start_date, end_date):
        self.data = None
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
    def plot_prices(self):
        self.data.plot(ylabel='USD', xlabel='Date')
        plt.show()

    def download_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)['Adj Close']
    def calculate_log_daily_returns(self):
        self.returns = np.log(self.data / self.data.shift(1))
        self.returns = self.returns[1:]
        
    def calculate_daily_returns(self):
        self.returns = self.data.pct_change(periods=1)
        self.returns = self.returns[1:]*100

    def returns_histogram(self):
        plt.figure(figsize=(6,6))

        mean = self.returns.mean()
        variance = self.returns.var()
        sigma = np.sqrt(variance)

        x = np.linspace(mean - 3 * sigma, mean + 3 * sigma, 100)
        y = norm.pdf(x, mean, sigma)

        print('mean: ', mean)
        print('var: ', variance)

        plt.title('GOOGL Daily Returns from 2023-01-01 to 2026-01-01')
        plt.hist(self.returns, bins=55)
        plt.xlabel('Daily return %')
        plt.ylabel('Frequency')
        plt.xticks(np.linspace(-10,10,11))
        # plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    stockData = StockData(['GOOGL'], '2023-01-01', '2026-01-01')
    stockData.download_data()
    stockData.plot_prices()
    stockData.calculate_daily_returns()
    stockData.returns_histogram()