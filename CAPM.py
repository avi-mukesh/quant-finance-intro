import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimize

RISK_FREE_RATE = 0.05
MONTHS_IN_YEAR = 12

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

        plt.figure(figsize=(10,6))
        plt.scatter(x=self.data['market_logreturns'], y=self.data['stock_logreturns'], marker='.')
        plt.show()

        self.data = self.data[1:]

    def calculate_beta(self):
        cov_matrix = np.cov(self.data['stock_logreturns'], self.data['market_logreturns'])
        cov_stock_returns_and_market_returns = cov_matrix[0,1]
        var_market_returns = cov_matrix[1,1]
        return cov_stock_returns_and_market_returns / var_market_returns
    
    def regression(self):
        # E[asset_return] - risk_free_rate = alpha + beta * (E[market_return] - risk_free_rate)
        # linear regression to fit line
        beta, alpha = np.polyfit(self.data['market_logreturns'] - RISK_FREE_RATE, self.data['stock_logreturns'] - RISK_FREE_RATE, 1)
        print('Beta from linear regression: ', beta)
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20,10))
        axis.scatter(self.data['market_logreturns'], self.data['stock_logreturns'], label='Data points')
        axis.plot(self.data['market_logreturns'], RISK_FREE_RATE + alpha + beta * (self.data['market_logreturns']-RISK_FREE_RATE), color='red', label='Regression')
        plt.title('CAPM')
        plt.xlabel('Market return $R_m$')
        plt.xlabel('Stock return $R_a$')
        plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # ^GSPC is S&P500 and we use that as the market baseline
    capm = CAPM(['IBM', '^GSPC'], '2018-01-01', '2021-01-01')
    data = capm.initialize()
    beta = capm.calculate_beta()
    print('Beta from cov/var formula: ', beta)
    
    capm.regression()
