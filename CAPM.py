import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class CAPM:
    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, start=self.start_date, end=self.end_date, auto_adjust=False)
            # instead of using raw Close price, Adjusted Close takes into account any corporate actions such as dividends, stock splits etc
            data[stock] = ticker[['Adj Close']].squeeze()

        return pd.DataFrame(data)
    
    def initialize(self):
        stocks_data = self.download_data()
        # we want to look at monthly returns so resample just reduces the data to the last day of each month
        # .last() just makes it so we take the final day of the month's value to represent that month
        stocks_data = stocks_data.resample('ME').last()

        self.data = pd.DataFrame({
            'stock_adjclose': stocks_data[self.stocks[0]],
            'market_adjclose': stocks_data[self.stocks[1]]
        })

        self.data[['stock_logreturns', 'market_logreturns']] = np.log(self.data / self.data.shift(1))
        self.data = self.data[1:]

    def calculate_beta(self):
        cov_matrix = np.cov(self.data['stock_logreturns'], self.data['market_logreturns'])
        cov_stock_returns_and_market_returns = cov_matrix[0,1]
        var_market_returns = cov_matrix[1,1]
        return cov_stock_returns_and_market_returns / var_market_returns

if __name__ == '__main__':
    # ^GSPC is S&P500 and we use that as the market baseline
    capm = CAPM(['IBM', '^GSPC'], '2018-01-01', '2021-01-01')
    data = capm.initialize()
    beta = capm.calculate_beta()
    print('Beta: ', beta)