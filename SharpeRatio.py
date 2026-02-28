import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use("seaborn-v0_8-darkgrid")


def brownian_bridge(T=1, N=252):
    dt = T/N
    t = np.linspace(0, T, N+1)
    
    # build standard Brownian motion
    W = np.zeros(N+1)
    for i in range(1, N+1):
        Z = np.random.normal()
        W[i] = W[i-1] + np.sqrt(dt) * Z

    # Force W_T = 0
    W_T = W[-1]
    for i in range(N+1):
        W[i] = W[i] - (t[i]/T) * W_T
    
    return t, W

def simulate_given_W(S0, t, W, sigma, g):
    # choose mu to guarantee same final value - cancels volatility drag
    N = len(t)-1
    S = np.zeros(N+1)
    
    mu = g + 0.5 * sigma**2
    for i in range(N+1):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i]
        S[i] = S0 * np.exp(drift + diffusion)
        
    return S

np.random.seed(35)

T = 1
g = 0.05   # desired final growth rate

t, W_A = brownian_bridge(T=T)
_, W_B = brownian_bridge(T=T)

S_A = simulate_given_W(100, t, W_A, sigma=0.02, g=g)
S_B = simulate_given_W(100, t, W_B, sigma=0.05, g=g)


data = {'A':S_A, 'B':S_B}
data = pd.DataFrame(data)

ret = data.pct_change(periods=1)

print(ret)
print(ret.mean()*252)
print(ret.std()*np.sqrt(252))

plt.plot(t, S_A, label='A')
plt.plot(t, S_B, label='B')
plt.xlabel('Years', fontsize=16)
plt.ylabel('Price', fontsize=16)
plt.legend(fontsize=15)
plt.title("Volatile and non-volatile strategies", fontsize=16)
plt.show()