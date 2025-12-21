import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# on average there are 252 trading days in a year
NUM_TRADING_DAYS = 252

# number of portfolios to simulate
NUM_PORTFOLIOS = 10000

# stocks we are going to handle
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']

start_date = '2020-01-01'
end_date = '2025-01-04'

def download_data():
    stock_data = {}

    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)
        ticker_price_history = ticker_data.history(start=start_date, end=end_date)['Close']
        stock_data[ticker] = ticker_price_history

    # structure is {'AAPL': [apple prices], 'MSFT': [microsoft prices], ...}
    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10, 6))
    plt.show()

def calculate_returns(data):
    # e.g. data is {'AAPL': [1,2,3]}
    # then data.shift(1) is {'AAPL': [NaN,1,2]}
    # so data/data.shift(1) is {'AAPL': [NaN,2,1.5]}
    log_returns = np.log(data/data.shift(1))
    return log_returns[1:]

def show_statistics(returns):
    # e.g. if returns are {'AAPL': [0.0027, 0.0023, ..., 0.0013]}
    # we work out the average return to be e.g. 0.0018
    # and then multiply by NUM_TRADING_DAYS to annualize it to e.g. 0.0018 * 252 = 0.4536
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance_of_portfolio(returns, weights):
    # term by term multiplication of returns.mean() and weights
    portfolio_expected_annual_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_annualised_variance = np.dot(weights.T, np.dot(returns.cov(), weights)) * NUM_TRADING_DAYS
    portfolio_annualised_volatility = np.sqrt(portfolio_annualised_variance)

    print("Expected portfolio mean (return): ", portfolio_expected_annual_return)
    print("Annualised portfolio standard deviation (volatility): ", portfolio_annualised_volatility)

def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []


    for _ in range(NUM_PORTFOLIOS):
        weights = np.random.rand(len(tickers)) # e.g. [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        weights /= np.sum(weights)
        portfolio_weights.append(weights)
        portfolio_means.append(np.sum(returns.mean() * weights) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)) * NUM_TRADING_DAYS))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def show_portfolios(returns_means, returns_volatilities):
    plt.figure(figsize = (10,6))
    plt.scatter(returns_volatilities, returns_means, c=returns_means/returns_volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

if __name__ == '__main__':
    dataset = download_data()
    log_daily_returns = calculate_returns(dataset)
    show_statistics(log_daily_returns)

    weights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)