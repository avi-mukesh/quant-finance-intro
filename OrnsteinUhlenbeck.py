import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

def generate_process(x0=.5, dt=0.1, theta=1.2, mu=0.5, sigma=0.3, n=200):
    x = np.zeros(n+1)
    x[0] = x0

    for t in range(1, n+1):
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * normal(0, np.sqrt(dt))

    t = np.linspace(0, n*dt, n+1)
    
    plt.figure(figsize=(10,6))
    plt.plot(t, x)
    plt.show()

if __name__ == '__main__':
    generate_process()