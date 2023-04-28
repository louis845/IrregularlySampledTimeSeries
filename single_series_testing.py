import time

import matplotlib.figure
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import encoder_ode_rnn
import decoder
import utils
import time_series_sampler

def plot_to_mp4(filename, update_callback, figsize=(19.2, 10.8), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)
    update_callback.set_renderers(fig, axs)

    # Create the animation
    anim = animation.FuncAnimation(fig, update_callback, frames=frames)
    anim.save(filename, fps=fps, extra_args=extra_args)

def plot_interactive(update_callback, figsize=(19.2, 10.8), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)
    update_callback.set_renderers(fig, axs)

    # Create the animation
    anim = animation.FuncAnimation(fig, update_callback, frames=frames)
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

encoder = encoder_ode_rnn.OdeRNN(4,
                                 utils.ODEFuncWrapper(utils.feedforward_nn(4, 4, 64, 3, device=torch.device("cuda"))),
                                 torch.nn.GRUCell(input_size=1, hidden_size=4))
decoder = decoder.Decoder(utils.ODEFuncWrapper(utils.feedforward_nn(4, 4, 64, 3, device=torch.device("cuda"))), utils.feedforward_nn(4, 1, 64, 3, device=torch.device("cuda")))
latent_encoder = utils.feedforward_nn(4, 2, 64, 3, device=torch.device("cuda"))
latent_decoder = utils.feedforward_nn(2, 4, 64, 3, device=torch.device("cuda"))


device = torch.device("cuda")
encoder = encoder.to(device)
decoder = decoder.to(device)


input_time_points1, output_time_points1, ground_truth_input1, ground_truth_output1 = time_series_sampler.sample_time_series(1, device=device, min_samples=7)
input_time_points1 = input_time_points1.squeeze(0)
output_time_points1 = output_time_points1.squeeze(0)
ground_truth_input1 = ground_truth_input1.squeeze(0)
ground_truth_output1 = ground_truth_output1.squeeze(0)


while True:
    input_time_points2, output_time_points2, ground_truth_input2, ground_truth_output2 = time_series_sampler.sample_time_series(1, device=device, min_samples=7)
    input_time_points2 = input_time_points2.squeeze(0)
    output_time_points2 = output_time_points2.squeeze(0)
    ground_truth_input2 = ground_truth_input2.squeeze(0)
    ground_truth_output2 = ground_truth_output2.squeeze(0)

    if torch.abs(ground_truth_output2[0] - ground_truth_output1[0]) > 0.2:
        break


input_time_points1_np = input_time_points1.cpu().numpy()
output_time_points1_np = output_time_points1.cpu().numpy()
ground_truth_input1_np = ground_truth_input1.cpu().numpy()
ground_truth_output1_np = ground_truth_output1.cpu().numpy()
input_time_points2_np = input_time_points2.cpu().numpy()
output_time_points2_np = output_time_points2.cpu().numpy()
ground_truth_input2_np = ground_truth_input2.cpu().numpy()
ground_truth_output2_np = ground_truth_output2.cpu().numpy()

plot1_tmin = np.min(input_time_points1_np)
plot1_tmax = np.max(output_time_points1_np)
plot2_tmin = np.min(input_time_points2_np)
plot2_tmax = np.max(output_time_points2_np)


optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

def odernn_run_plot(i, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes):
    low = -1.5
    high = 1.5

    ax[0].clear()
    ax[0].set_xlim(plot1_tmin, plot1_tmax)
    ax[0].set_ylim(low, high)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("y")

    # Plot ground truth t:total_time_samples, y:total_function_samples. color="red". label="Ground Truth"
    ax[0].plot(output_time_points1_np, ground_truth_output1_np, color="red", label="Ground Truth")

    # Scatter the encoder predictions t:training_time, y:training_function_vals. color="blue". label="Irregular samples as input"
    ax[0].scatter(input_time_points1_np, ground_truth_input1_np, color="blue", label="Irregular samples as input", s=30)


    ax[1].clear()
    ax[1].set_xlim(plot2_tmin, plot2_tmax)
    ax[1].set_ylim(low, high)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("y")

    # Plot ground truth t:total_time_samples, y:total_function_samples. color="red". label="Ground Truth"
    ax[1].plot(output_time_points2_np, ground_truth_output2_np, color="red", label="Ground Truth")

    # Scatter the encoder predictions t:training_time, y:training_function_vals. color="blue". label="Irregular samples as input"
    ax[1].scatter(input_time_points2_np, ground_truth_input2_np, color="blue", label="Irregular samples as input", s=30)

    optimizer.zero_grad()
    batch_input_time_points, batch_output_time_points, batch_ground_truth_input, batch_ground_truth_output = time_series_sampler.sample_time_series(1000, device=device, min_samples=2)

    assert batch_input_time_points.shape[0] == batch_ground_truth_input.shape[1]
    assert batch_output_time_points.shape[0] == batch_ground_truth_output.shape[1]

    latent = encoder(batch_input_time_points, batch_ground_truth_input.unsqueeze(-1).permute(1, 0, 2))
    predictions = decoder(batch_output_time_points, latent).squeeze(dim=-1).permute(1, 0)

    loss = (torch.abs(predictions - batch_ground_truth_output) * torch.exp(-0.25 * batch_output_time_points).unsqueeze(0)).mean()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        latent = encoder(input_time_points1, ground_truth_input1.unsqueeze(-1))
        predictions = decoder(output_time_points1, latent).squeeze(dim=-1)

        latent2 = encoder(input_time_points2, ground_truth_input2.unsqueeze(-1))
        predictions2 = decoder(output_time_points2, latent2).squeeze(dim=-1)

    ax[0].plot(output_time_points1_np, predictions.detach().cpu().numpy(), color="green", label="Predictions")
    ax[1].plot(output_time_points2_np, predictions2.detach().cpu().numpy(), color="green", label="Predictions")

    ax[0].set_title("Function 1. Epoch: {} Start pred: {}".format(i, predictions[0].item()))
    ax[1].set_title("Function 2. Epoch: {} Start pred: {}".format(i, predictions2[0].item()))

    for k in range(2):
        ax[k].legend()

ctime = time.time()
plot_to_mp4("output.mp4", UpdateCallback(odernn_run_plot), frames=4900)
print("Time taken: {}".format(time.time() - ctime))

# Save the learned models encoder, decoder, latent_encoder, latent_decoder to files encoder.pt, decoder.pt, latent_encoder.pt, latent_decoder.pt
torch.save(encoder.state_dict(), "encoder.pt")
torch.save(decoder.state_dict(), "decoder.pt")
torch.save(latent_encoder.state_dict(), "latent_encoder.pt")
torch.save(latent_decoder.state_dict(), "latent_decoder.pt")