import numpy as np
from random import random
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-darkgrid")
import matplotlib.animation as animation

num_steps = 10

fig, ax = plt.subplots()

fig.set_figheight(6)
fig.set_figwidth(6)

ax.grid(True, which='both', linestyle='--', alpha=0.6)

t = np.linspace(0,num_steps,num_steps+1)

y_t = [0]
for _ in range(num_steps):
    y_t.append(-1 if random()<0.5 else 1)
x_t = np.cumsum(y_t)

# initialize an empty Line2D artist
line = ax.plot([],[])[0]
ax.set(xlim=[0,num_steps], ylim=[-num_steps//2-1,num_steps//2+1], xlabel='t', ylabel='$X_t$', title='1D Simple Random Walk')
# set ticks at every integer step
ax.set_xticks(np.arange(0, num_steps+1, 1))
ax.set_yticks(np.arange(-num_steps//2-1, num_steps//2+2, 1))

def update(frame):
    # frame goes from 0..num_steps, include frame index
    T = t[:frame+1]
    Xt = x_t[:frame+1]

    line.set_data(T, Xt)

    return line,

ani = animation.FuncAnimation(fig, update, frames=num_steps+1, interval=1200)

FFwriter = animation.FFMpegWriter(fps=1)
ani.save('animation.mp4', writer = FFwriter)

plt.show()