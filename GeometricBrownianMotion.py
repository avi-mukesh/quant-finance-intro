import matplotlib.pyplot as plt
import numpy as np

# S0 is start value of stock
def simulate_geometric_random_walk(S0, T=2, N=100, mu=0.1, sigma=0.5):
    dt = T/N
    t = np.linspace(0, T, N+1)

    # W(t+dt) - W(t) follows N(0, dt^2) distribution
    W = np.zeros(N+1)
    W[1:N+1] = np.cumsum(np.sqrt(dt) * np.random.standard_normal(size=N))

    # d(log(S(t))) = (mu - 0.5 * s^2) * dt + s * dW
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)

    return t, S

def plot_simulation(t, S):
    plt.plot(t, S)
    plt.xlabel('Time(t)')
    plt.ylabel('Stock price S(t)')
    plt.title('Geometric Brownian Motion')
    plt.show()

if __name__ == '__main__':
    t, S = simulate_geometric_random_walk(10)
    plot_simulation(t, S)