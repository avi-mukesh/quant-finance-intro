import numpy as np
import math
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_sims = 1000
num_steps = 100

# set up figure and axes
fig, ax = plt.subplots(figsize=(8,5))
t = np.arange(0, num_steps+1)
ax.set_xlim(0, num_steps)

# precompute all walks
walks = []
for _ in range(num_sims):
    y_t = [0]
    for _ in range(num_steps):
        y_t.append(-1 if random()<0.5 else 1)
    x_t = np.cumsum(y_t)
    walks.append(x_t)
walks = np.array(walks)

# adjust y limits to data
max_abs = int(np.max(np.abs(walks))) + 1
ax.set_ylim(-max_abs, max_abs)

ax.set_xlabel('t')
ax.set_title('1D Simple Random Walk')

# integer ticks
ax.set_xticks(np.arange(0, num_steps+1, 10))
ax.set_yticks(np.arange(-max_abs, max_abs+1, 5))

cmap = plt.get_cmap('cool')
colors = cmap(np.linspace(0, 1, num_sims))

# create empty Line2D artists (one per simulation)
lines = [ax.plot([], [], color=colors[i])[0] for i in range(num_sims)]

def init():
    for line in lines:
        line.set_data([], [])
    return tuple(lines)

def update(frame_num):
    lines[frame_num].set_data(t, walks[frame_num])
    return lines[frame_num],

ani = animation.FuncAnimation(fig, update, frames=num_sims, init_func=init, interval=50, blit=True)

root_t = np.sqrt(np.linspace(0,num_steps,num_steps+1))
ax.plot(root_t, label='sqrt(t)', lw=2, c='black')
ax.plot(-root_t, label='-sqrt(t)', lw=2, c='black')

plt.show()