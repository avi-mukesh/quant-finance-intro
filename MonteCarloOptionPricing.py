import numpy as np

class OptionsPricing:
    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations
    
    def call_option_simulation(self):
        z = np.random.normal(0, 1, self.iterations)

        # simulate stock price at time T by simulating dS_t = \r_f*S_t*dt + \sigma * S_t * dW_t^Q
        # notice we are using risk neutral measure and using r_f instead of mu now
        # expected growth of asset = risk-free rate - the entire foundation of risk-free pricing
        stock_price_simulations = self.S0 * np.exp((self.rf - 0.5 * self.sigma ** 2)*self.T + self.sigma * np.sqrt(self.T) * z)
        
        payoffs = np.maximum(stock_price_simulations - self.E, 0)
        # work out the average payoff amongst all simulations
        avg_payoff = np.mean(payoffs)
        # discount to present time
        return avg_payoff * np.exp(-self.rf * self.T)
    
    def put_option_simulation(self):
        z = np.random.normal(0, 1, self.iterations)

        stock_price_simulations = self.S0 * np.exp((self.rf - 0.5 * self.sigma ** 2)*self.T + self.sigma * np.sqrt(self.T) * z)
        
        option_data = np.zeros([self.iterations, 2])
        
        option_data[:, 1] = self.E - stock_price_simulations

        payoffs = np.max(option_data, axis=1)
        avg_payoff = np.mean(payoffs)

        return avg_payoff * np.exp(-self.rf * self.T)

if __name__ == '__main__':
    op = OptionsPricing(100, 100, 1, 0.05, 0.2, 1000000)
    print('Value of call option £%.2f' % op.call_option_simulation())
    print('Value of put option £%.2f' % op.put_option_simulation())