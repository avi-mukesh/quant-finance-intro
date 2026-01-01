import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime

def download_data(stock, start_date, end_date):
    data = {}

    ticker = yf.download(stock, start=start_date, end=end_date, auto_adjust=False)
    data[stock] = ticker[['Adj Close']].squeeze()

    return pd.DataFrame(data)

# calculate 1-day VaR 
def calculate_var(position, c, mean, sigma):
    # c is the confidence level e.g. 0.95
    # VaR = position (mean - z * sigma)
    # z can be calculated using inv normal CD 
    # if c=0.95, then 1-c=0.05, and norm.ppf inverse normal CDF function i.e. P(Z<=z) = 0.05, what is z
    z = norm.ppf(1-c)
    var = position * (mean - z*sigma)
    return var

def calculate_var_n_days(position, c, mean, sigma, n):
    z = norm.ppf(1-c)
    var = position * (mean*n - z*sigma*np.sqrt(n))
    return var

if __name__ == '__main__':
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2024, 1, 1)

    stock_data = download_data('AAPL', start, end)

    # print(stock_data)
    log_daily_returns = np.log(stock_data/stock_data.shift(1))[1:]

    # this is the investment
    S = 1e6 
    # confidence level - this time at 95%
    c=0.95

    # we assume daily returns are normally distributed
    mu = np.mean(log_daily_returns)
    sigma = np.std(log_daily_returns, axis=0).iloc[0]

    varOneDay95 = calculate_var(S, 0.95, mu, sigma)
    varOneDay99 = calculate_var(S, 0.99, mu, sigma)
    varOneYear95 = calculate_var_n_days(S, c, mu, sigma, 252)

    print('mu', mu)
    print('sigma', sigma)
    print('')

    # if 1-day 95% VaR is x, it means with 95% confidence, losses cannot exceed x in one day
    print('1-day VaR at 95%', varOneDay95)
    # if 1-day 95% VaR is x, it means with 99% confidence, losses cannot exceed x in one day
    # var at 99% confidence is higher because we are more confident we are losing at most a certain amount of money, so that certain amount of money must be higher
    # e.g. think about what this sentence: "I am 100% confident my loss over the next day will not exceed ___"
    print('1-day VaR at 99%', varOneDay99)
    # if 1-year 95% VaR is x, it means with 95% confidence, losses cannot exceed x in one year
    print('1-year VaR at 95%', varOneYear95)
