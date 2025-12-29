import numpy.random as rand
import numpy as np
import matplotlib.pyplot as plt

def wiener_process(dt=0.1, x0=0, n=1000):
    W = np.zeros(n+1)

    t = np.linspace(x0, n, n+1)

    # draw n points from the normal distribution e.g. [-0.3, 0.4, 0.25, ...]
    # then cumulative sum gives [-0.3, 0.1, 0.35]
    W[1:n+1] = np.cumsum(rand.normal(0, np.sqrt(dt), n))

    return t, W

def plot_process(t, W):
    plt.plot(t, W)
    plt.xlabel('Time(t)')
    plt.ylabel('Wiener process W(t)')
    plt.show()

if __name__ == '__main__':
    time, data = wiener_process()
    plot_process(time, data)