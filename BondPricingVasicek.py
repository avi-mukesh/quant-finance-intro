import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# simulating 1000 r(t) interest rate processes
NUM_SIMULATIONS = 1000
NUM_POINTS = 200

# x = principal amount
# r0 = initial interest rate
def monte_carlo_simulation(F, r0, kappa, theta, sigma, T=1):
    dt = T / float(NUM_POINTS)
    result = []

    for _ in range(NUM_SIMULATIONS):
        # each rates array is one realization of the short-rate path: {r_0, r_dt, r_2dt, ..., r_T} (one possible future evolution of interest rates)
        rates = [r0]
        for _ in range(NUM_POINTS):
            rates.append(rates[-1] + kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal())

        result.append(rates)

    simulation_data = pd.DataFrame(result).T

    # price at t of zero-coupon bond maturing at T is E[exp(-integral of r_s from s=t to s=T)]
    # the expectation is taken under the risk-neutral measure
    # we are using Monte-Carlo simulation in order to calculate this expectation
    # using rectangles to numerically calculate integral
    integrals_for_each_simulation = simulation_data.sum() * dt
    discounted = np.exp(-integrals_for_each_simulation)
    bond_price = F * np.mean(discounted)

    print('Bond price based on Monte-Carlo simulation: %.2f' % bond_price)

    return simulation_data


if __name__ == '__main__':
    interest_rate_paths = monte_carlo_simulation(1000, 0.1, 0.3, 0.3, 0.03)
    plt.plot(interest_rate_paths)
    plt.show()