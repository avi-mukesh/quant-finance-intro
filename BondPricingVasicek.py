import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# simulating 1000 r(t) interest rate processes
NUM_SIMULATIONS = 1000
# 
NUM_POINTS = 200

# x = principal amount
# r0 = initial interest rate
def monte_carlo_simulation(F, r0, kappa, theta, sigma, T=1):
    dt = T / float(NUM_POINTS)
    result = []

    for _ in range(NUM_SIMULATIONS):
        rates = [r0]
        for _ in range(NUM_POINTS):
            rates.append(rates[-1] + kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal())

        result.append(rates)

    simulation_data = pd.DataFrame(result)
    return simulation_data.T

if __name__ == '__main__':
    interest_rate_paths = monte_carlo_simulation(1000, 0.1, 0.3, 0.12, 0.03)
    plt.plot(interest_rate_paths)
    plt.show()