import numpy as np

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
        rand = np.random.normal(0, 1, [1, self.iterations])

        # geometric random walk simulation of stock price
        # recall, SDE is dS = mu * S * dt + sigma * S * dW
        # W is Wiener process i.e. dW ~ N(0, t)
        stock_price_simulations = self.S * np.exp((self.mu - 0.5 * self.sigma ** 2)*self.n + self.sigma * np.sqrt(self.n) * rand)[0]


        # 95% VaR means I am 95% confident my loses will not exceed this amount
        # i.e. 5% confident my losses will be less than this amount
        # so take the price such that 5% of prices are smaller than it i.e. the 5th percentile
        stock_price_simulations = np.sort(stock_price_simulations)
        print(stock_price_simulations)

        # worst case value of our given investment by the end date
        percentile = np.quantile(stock_price_simulations, 1-self.c)

        # maximum possible loss - VaR
        return self.S - percentile

if __name__ == '__main__':
    mu = 0.02
    sigma = 0.03
    model = VaR(1e6, 0.95, mu, sigma, 1, 11)
    model.simulation()
