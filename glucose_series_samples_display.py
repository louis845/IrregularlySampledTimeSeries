import time
import argparse

import matplotlib.figure
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import encoder_ode_rnn
import decoder
import utils
import time_series_sampler

# parse arguments.
parser = argparse.ArgumentParser(description="Visualize the randomly generated glucose time series.")
args = parser.parse_args()

def plot_interactive(update_callback, figsize=(25.6, 14.4), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

    # Create the left subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Create the right subplot
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])
    fig.subplots_adjust(hspace=0.5)
    update_callback.set_renderers(fig, [ax1, ax2, ax3, ax4, ax5, ax6])

    # Create the animation
    anim = animation.FuncAnimation(fig, update_callback, frames=frames, interval=1000 / fps)
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

time_series_sampler.setup_glucose_sampling()
time_series_sampler.low = -160
time_series_sampler.high = 240
time_series_sampler.samples_width = 1.0 / 80
time_series_sampler.setup_glucose_sampling_with_fixed_decay(0.5)

device = torch.device("cuda")

n = 3

cbar = None
epoch = 1
ctime = time.time()

def odernn_run_plot(i, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes):
    global cbar, epoch, ctime
    low = -1
    high = 5

    input_time_points = np.empty((n,), dtype="object")
    output_time_points = np.empty((n,), dtype="object")
    ground_truth_input = np.empty((n,), dtype="object")
    ground_truth_output = np.empty((n,), dtype="object")

    input_time_points[0], output_time_points[0], ground_truth_input[0], ground_truth_output[
        0] = time_series_sampler.sample_time_series(1, device=device, min_samples=4, max_samples=10, after_samples=80,
                                           sampling_method="glucose")
    input_time_points[0] = input_time_points[0].squeeze(0)
    output_time_points[0] = output_time_points[0].squeeze(0)
    ground_truth_input[0] = ground_truth_input[0].squeeze(0)
    ground_truth_output[0] = ground_truth_output[0].squeeze(0)

    i = 1
    while i < n:
        input_time_points[i], output_time_points[i], ground_truth_input[i], ground_truth_output[
            i] = time_series_sampler.sample_time_series(1, device=device, min_samples=4, max_samples=10, after_samples=80,
                                           sampling_method="glucose")
        input_time_points[i] = input_time_points[i].squeeze(0)
        output_time_points[i] = output_time_points[i].squeeze(0)
        ground_truth_input[i] = ground_truth_input[i].squeeze(0)
        ground_truth_output[i] = ground_truth_output[i].squeeze(0)

        if torch.abs(ground_truth_output[i][0] - ground_truth_output[i - 1][0]) > 0.2:
            i += 1

    input_time_points_np = np.empty((n,), dtype="object")
    output_time_points_np = np.empty((n,), dtype="object")
    ground_truth_input_np = np.empty((n,), dtype="object")
    ground_truth_output_np = np.empty((n,), dtype="object")

    for i in range(n):
        input_time_points_np[i] = input_time_points[i].cpu().numpy()
        output_time_points_np[i] = output_time_points[i].cpu().numpy()
        ground_truth_input_np[i] = ground_truth_input[i].cpu().numpy()
        ground_truth_output_np[i] = ground_truth_output[i].cpu().numpy()

    plot_tmin = np.empty((n,), dtype="object")
    plot_tmax = np.empty((n,), dtype="object")

    # Compute the plot limits
    for i in range(n):
        plot_tmin[i] = np.min(input_time_points_np[i])
        plot_tmax[i] = np.max(output_time_points_np[i])

    for j in range(n):
        ax[j].clear()
        ax[j].set_xlim(plot_tmin[j], plot_tmax[j])
        ax[j].set_ylim(low, high)
        ax[j].set_xlabel("t")
        ax[j].set_ylabel("y")

        # Plot ground truth t:total_time_samples, y:total_function_samples. color="red". label="Ground Truth"
        ax[j].plot(output_time_points_np[j], ground_truth_output_np[j], color="red", label="Ground Truth")

        # Scatter the encoder predictions t:training_time, y:training_function_vals. color="blue". label="Irregular samples as input"
        ax[j].scatter(input_time_points_np[j], ground_truth_input_np[j], color="blue", label="Irregular samples as input", s=30)

    # Plot directly sampled (without sparse sampling or shifting) glucose time series.
    batch_size = n
    onset, weight, decay = time_series_sampler.generate_glucose_spikes(batch_size, device=device)

    time = torch.linspace(-3.0, 3.0, 1000, device=device)

    glucose_vals = time_series_sampler.sample_batch_glucose(
        time.unsqueeze(0).repeat_interleave(batch_size, dim=0), onset, weight, decay)
    time = time.cpu().numpy()

    for j in range(n):
        ax[n+j].clear()
        ax[n+j].set_xlim(time[0], time[-1])
        ax[n+j].set_ylim(low, high)
        ax[n+j].set_xlabel("t")
        ax[n+j].set_ylabel("y")
        ax[n+j].plot(time, glucose_vals[j, :].cpu().numpy(), color="red", label="Ground Truth")

    epoch += 1
    if epoch % 1000 == 0:
        ctime = time.time() - ctime
        print("Iter: {} Time taken: {}".format(epoch, ctime))
        ctime = time.time()



ctime = time.time()
plot_interactive(UpdateCallback(odernn_run_plot), frames=2400, fps=2)
print("Time taken: {}".format(time.time() - ctime))