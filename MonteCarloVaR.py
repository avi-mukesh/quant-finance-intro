from math import log
import numpy as np
import datetime
import yfinance as yf
import pandas as pd

def download_data(ticker, start, end):
    ticker_data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    
    data = {}
    data[ticker] = ticker_data[['Adj Close']].squeeze()

    return pd.DataFrame(data)


class VaR:
    def __init__(self, S, c, mu, sigma, n, iterations):
        # S is value of investment at t=0
        self.S = S
        self.c = c
        self.mu = mu
        self.sigma = sigma
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(size=self.iterations)

        # geometric random walk simulation of stock price
        # recall, SDE is dS = mu * S * dt + sigma * S * dW
        # W is Wiener process i.e. dW ~ N(0, dt^2)
        stock_price_simulations = self.S * np.exp((self.mu - 0.5 * self.sigma ** 2)*self.n + self.sigma * np.sqrt(self.n) * rand)

        # 95% VaR means I am 95% confident my loses will not exceed this amount
        # i.e. 5% confident my losses will be less than this amount
        # so take the price such that 5% of prices are smaller than it i.e. the 5th percentile
        # worst case value of our given investment by the end date
        percentile = np.quantile(stock_price_simulations, 1-self.c)

        # maximum possible loss - VaR
        return self.S - percentile

if __name__ == '__main__':
    S=1e6
    iterations = 100000

    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2024, 1, 1)

    stock_data = download_data('AAPL', start, end)
    log_returns = np.log(stock_data['AAPL']).diff().dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    # perc_daily_returns = stock_data['AAPL'].pct_change()
    # mu = np.mean(perc_daily_returns)
    # sigma = np.std(perc_daily_returns, axis=0)

    print('mu', mu)
    print('sigma', sigma)

    
    print('1-day VaR at 95 using Monte Carlo simulations: £%.2f' % VaR(S, 0.95, mu, sigma, 1, iterations).simulation())
    print('1-day VaR at 99 using Monte Carlo simulations: £%.2f' % VaR(S, 0.99, mu, sigma, 1, iterations).simulation())
    print('1-year VaR at 95 using Monte Carlo simulations: £%.2f' % VaR(S, 0.95, mu, sigma, 252, iterations).simulation())
