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

    def download_data(self):
        ticker_data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
        
        data = {}
        data[self.ticker] = ticker_data[['Adj Close']].squeeze()

        self.data = pd.DataFrame(data)
    
    def calculate_log_daily_returns(self):
        self.returns = np.log(self.data / self.data.shift(1))
        self.returns = self.returns[1:]
        
    def calculate_daily_returns(self):
        self.returns = (self.data - self.data.shift(1))/self.data.shift(1)
        self.returns = self.returns[1:]

    def returns_histogram(self):
        plt.figure(figsize=(10,6))

        mean = self.returns.mean()
        variance = self.returns.var()
        sigma = np.sqrt(variance)

        x = np.linspace(mean - 5 * sigma, mean + 5 * sigma, 100)
        y = norm.pdf(x, mean, sigma)

        print('mean: ', mean)
        print('var: ', variance)

        plt.hist(self.returns, bins=80)
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    stockData = StockData('MSFT', '2016-01-01', '2026-01-01')
    stockData.download_data()
    stockData.calculate_daily_returns()
    stockData.returns_histogram()