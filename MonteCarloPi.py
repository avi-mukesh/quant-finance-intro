import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

radius = 1
inside = 0
total = 0

x_inside, y_inside = [], []
x_outside, y_outside = [], []

# Vertical video
fig, ax = plt.subplots(figsize=(6,9))

# Dark gray background
fig.patch.set_facecolor("#222222")

# Draw circle
circle = plt.Circle((0,0), radius, fill=False, linewidth=2, color="white")
ax.add_patch(circle)

# Draw square border (side length = 2*radius)
square = plt.Rectangle((-radius, -radius), 2*radius, 2*radius,
                       fill=False, linewidth=2, edgecolor="white")
ax.add_patch(square)

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_aspect("equal")

# Hide axes for cleaner look
ax.axis("off")

# --- Bolder, more visible π text ---
counter_text = ax.text(
    0, -1.1, "", 
    fontsize=18,        # bigger font
    color="white",    # gold color to pop on dark background
    horizontalalignment='center',
    verticalalignment='top'
)

pi_text = ax.text(
    0, -1.45, "", 
    fontsize=24,        # bigger font
    fontweight='bold',  # bold
    color="#FFD700",    # gold color to pop on dark background
    bbox=dict(facecolor="#222222", edgecolor="white", boxstyle="round,pad=0.3"),  # subtle background
    horizontalalignment='center',
    verticalalignment='top'
)

inside_scatter = ax.scatter([], [], s=5, color="#4cc9f0")
outside_scatter = ax.scatter([], [], s=5, color="#f72585")


def update(frame):
    global inside, total

    for _ in range(10):

        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)

        total += 1

        if x*x + y*y <= 1:
            inside += 1
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    inside_scatter.set_offsets(np.column_stack((x_inside, y_inside)))
    outside_scatter.set_offsets(np.column_stack((x_outside, y_outside)))

    pi_est = 4 * inside / total

    counter_text.set_text(f"Points inside circle: {inside}\nTotal points: {total}\n")
    pi_text.set_text(f"π ≈ {pi_est:.5f}")

    return inside_scatter, outside_scatter, pi_text


anim = FuncAnimation(fig, update, frames=500, interval=20)


anim.save('monte_carlo_pi.mp4', writer = animation.FFMpegWriter(fps=35))

plt.show()