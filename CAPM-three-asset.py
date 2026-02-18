import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
import numpy as np
from numpy.linalg import inv
import matplotlib.animation as animation
np.random.seed(42)

NUM_PORTFOLIOS = 10000

def generate_portfolios(m, C):
    portfolio_means = []
    portfolio_volatilities = []
    for _ in range(NUM_PORTFOLIOS):
        # use this if allowing only long positions w_i>=0
        # but then you won't get the whole feasible set
        # w = np.random.uniform(0, 1, len(m))
        
        # allow short selling as well by allowing negative weights
        w = np.random.normal(size=len(m))
        w /= np.sum(w)
        portfolio_means.append(w@m)
        portfolio_volatilities.append(np.sqrt(w@C@w.T))
    return portfolio_means, portfolio_volatilities
    
def minimum_variance_line(mu_vals, m, C):
    variance_values = []
    C_inv = inv(C)
    u = np.ones(len(m))
    M = np.array([[m@C_inv@m.T, u@C_inv@m.T],
                  [m@C_inv@u.T, u@C_inv@u.T]])
    
    for mu in mu_vals:
        l = inv(M) @ np.array([mu, 1])
        w = (l[0]*m@C_inv + l[1]*u@C_inv)
        variance_values.append(w@C@w.T)
    
    sigma_values = np.sqrt(variance_values)
    return sigma_values


def draw_line_from_point(sigma_vals, gradient, risk_free_rate):
    return sigma_vals * gradient + risk_free_rate

# asset means, volatilities, correlations, covariance matrix

m = np.array([0.1, 0.15, 0.2]) # example 3.29 in Mathematics for Finance: An Introduction to Financial Engineering
s1, s2, s3 = 0.28, 0.24, 0.25
p12, p13, p23 = -0.1, 0.25, 0.2
C = np.array([[s1**2, s1*s2*p12, s1*s3*p13], 
              [s1*s2*p12, s2**2, s2*s3*p23],
              [s1*s3*p13, s2*s3*p23, s3**2]])



# Scatter plot of many different portfolios, each with a different combination of the 3 assets
portfolio_means, portfolio_volatilities = generate_portfolios(m, C)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(portfolio_volatilities, portfolio_means, s=1, color='darksalmon')


# Minimum Variance Line
mu_vals = np.linspace(min(portfolio_means), max(portfolio_means), 100000)
sigma_vals = minimum_variance_line(mu_vals, m, C)
ax.plot(sigma_vals, mu_vals, color='maroon', linewidth=3, label='Minimum Variance Line')

# Efficient Frontier
u = np.ones(len(m))
w_mvp = (u@inv(C)) / (u@inv(C)@u.T)
mu_mvp = w_mvp@m
print('mu_mvp', mu_mvp)
mu_vals = np.linspace(mu_mvp, max(portfolio_means), 100000)
sigma_vals = minimum_variance_line(mu_vals, m, C)
ax.plot(sigma_vals, mu_vals, color='forestgreen', linewidth=3, label='Efficient Frontier')

# introducing a risk free asset with rate R=0.12
# animate lines going from (0, R) to the efficient frontier
R = 0.12
w_market = ((m-R*u) @ inv(C)) / ((m-R*u) @ inv(C) @ u.T)
mu_market = w_market@m
sigma_market = np.sqrt(w_market@C@w_market.T)
highest_sharpe = (mu_market - R) / sigma_market

sigma_vals_for_line = np.linspace(0, 1, 100)
line, = ax.plot([], [], color='black', linewidth=3)
def animate(gradient):
    mu_vals_for_line = draw_line_from_point(sigma_vals_for_line, gradient, R)
    line.set_data(sigma_vals_for_line, mu_vals_for_line)
    return line,
gradients = np.linspace(0.15, highest_sharpe, 100)
frames = np.concatenate((gradients, gradients[::-1])) # for reversing the animation
ax.scatter([0], [R], color='teal', marker='o', s=50, label='Risk-free asset')
anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=True, interval=20)

# Mark where the tangency portfolio is
ax.plot(sigma_market, mu_market, '*', markersize=20, color='gold', label='Tangency Portfolio')

# Capital Market Line - line with highest gradient/Sharpe ratio
mu_vals_for_cml_line = draw_line_from_point(sigma_vals_for_line, highest_sharpe, R)
ax.plot(sigma_vals_for_line, mu_vals_for_cml_line, color='black', linewidth=3, label='CML')

ax.set_xlim(0, 0.7)
ax.set_ylim(0, 0.3)
ax.set_xlabel('$\sigma$ (portfolio volatility)', fontsize=16)
ax.set_ylabel('$\mu$ (portfolio mean)', fontsize=16)
ax.set_title('Geometry of CAPM', fontsize=18, pad=10)
ax.legend(fontsize=14)

anim.save('lines.mp4', writer = animation.FFMpegWriter(fps=40))

plt.show()