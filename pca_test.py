import time
import argparse

import matplotlib.figure
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.axes


def plot_interactive(titlename, update_callback, figsize=(10.8, 10.8), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)

    update_callback.set_renderers(fig, fig.add_subplot(gs[0, 0]))

    # Create the animation
    anim = animation.FuncAnimation(fig, update_callback, frames=frames, interval=1000/fps)
    plt.show()

class UpdateCallback:
    def __init__(self, plot_callback):
        self.axs = None
        self.fig = None
        self.plot_callback = plot_callback

    def set_renderers(self, fig, axs: np.ndarray):
        self.axs = axs
        self.fig = fig

    def __call__(self, i):
        self.plot_callback(i, self.fig, self.axs)

epoch = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
theta = torch.tensor(1.0, device=device)
matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=device)
matrix = torch.matmul(matrix, torch.matmul(torch.diag(torch.tensor([1.0, 0.5], device=device)), matrix.t()))
def odernn_run_plot(i, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes):
    global epoch
    epoch += 1
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    embeddings = torch.matmul(torch.randn(1000, 2, device=device), matrix)
    embeddings = embeddings - embeddings.mean(dim=0)
    U, S, V = np.linalg.svd(embeddings.cpu().numpy())
    print("---------------------------------------")
    print(V.T)
    print(np.transpose(V, [0, 1]))
    embeddings = torch.matmul(embeddings, torch.tensor(V.T, device=device)).cpu().numpy()
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    ax.scatter(x, y, s=15)

name="Title"

ctime = time.time()
plot_interactive("{}.mp4".format(name), UpdateCallback(odernn_run_plot), frames=2400, fps=1)
print("Time taken: {}".format(time.time() - ctime))