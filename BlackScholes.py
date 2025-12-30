from scipy import stats
from numpy import log, exp, sqrt

def call_option_price(S, E, T, rf, sigma, t=0):
    d1 = (log(S/E) + (rf + 0.5 * sigma ** 2) * (T-t)) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T-t)

    return S*stats.norm.cdf(d1) - E*exp(-rf*(T-t))*stats.norm.cdf(d2)

def put_option_price(S, E, T, rf, sigma, t=0):
    d1 = (log(S/E) + (rf + 0.5 * sigma ** 2) * (T-t)) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T-t)

    return -S*stats.norm.cdf(-d1) + E*exp(-rf*(T-t))*stats.norm.cdf(-d2)

if __name__ == '__main__':
    # current stock price
    S = 100
    # strike price
    E = 100
    # expiry = 1 year
    T = 1
    # risk free rate
    rf = 0.05
    # volatility of stock
    sigma = 0.2

    print("Call option price according to Black-Scholes:", call_option_price(S, E, T, rf, sigma))
    print("Put option price according to Black-Scholes:", put_option_price(S, E, T, rf, sigma))
