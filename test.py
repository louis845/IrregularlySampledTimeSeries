import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1, 1)

x_offset = 0.0

# Define the animation function
def animate(frame):
    ax.clear()
    global x_offset
    x_offset += 0.1
    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x + x_offset)
    ax.plot(x, y)
    ax.set_title("Offset: {:.2f}".format(x_offset))
    return ax,

# Create the animation
anim = FuncAnimation(fig, animate, frames=100, interval=50)

# Set up the ffmpeg writer
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as an mp4 file
anim.save('sine_curve.mp4', writer=writer)

# Show the animation in a window
plt.show()