import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_SIMULATIONS = 1000

def stock_monte_carlo(S0, mu, sigma, N=1000):
    result = []

    # possible S(t) realisations of the process
    for _ in range(NUM_SIMULATIONS):
        prices = [S0]
        for _ in range(N):
            stock_price = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * np.random.normal())
            prices.append(stock_price)

        result.append(prices)

    simulation_data = pd.DataFrame(result)
    simulation_data = simulation_data.T

    mean = simulation_data.mean(axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(simulation_data)
    plt.plot(mean, linewidth=3, color='black', label='Average')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    stock_monte_carlo(50, 0.0001, 0.01)