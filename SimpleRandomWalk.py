import numpy as np
from random import random
import matplotlib.pyplot as plt

y_t = [0]

num_steps = 100

for _ in range(num_steps):
    y_t.append(-1 if random()<0.5 else 1)
    
x_t = np.cumsum(y_t)

plt.figure(figsize=(7,7))
# plt.xticks(np.linspace(0,100,101))
# plt.yticks(np.linspace(-30,30,61))
plt.xlim(0,100)
plt.ylim(-20,20)
plt.grid()
plt.plot(x_t)
plt.xlabel('t')
plt.title('1D Simple Random Walk')

root_t = np.sqrt(np.linspace(0,100,101))
plt.plot(root_t, label='sqrt(t)')
plt.plot(-root_t, label='-sqrt(t)')
plt.legend()

plt.show()