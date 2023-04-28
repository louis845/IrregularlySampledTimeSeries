import matplotlib.figure
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import encoder_ode_rnn
import decoder
import utils

def plot_to_mp4(filename, update_callback, figsize=(19.2, 10.8), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig, ax = plt.subplots(figsize=figsize)
    update_callback.set_renderers(fig, ax)

    # Create the animation
    anim = animation.FuncAnimation(fig, update_callback, frames=frames)
    anim.save(filename, fps=fps, extra_args=extra_args)

def plot_interactive(update_callback, figsize=(19.2, 10.8), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig, ax = plt.subplots(figsize=figsize)
    update_callback.set_renderers(fig, ax)

    # Create the animation
    anim = animation.FuncAnimation(fig, update_callback, frames=frames)
    plt.show()

class UpdateCallback:
    def __init__(self, plot_callback):
        self.ax = None
        self.fig = None
        self.plot_callback = plot_callback

    def set_renderers(self, fig, ax):
        self.ax = ax
        self.fig = fig

    def __call__(self, i):
        self.plot_callback(i, self.fig, self.ax)

encoder = encoder_ode_rnn.OdeRNN(4,
                                 utils.ODEFuncWrapper(utils.feedforward_nn(4, 4, 64, 3, device=torch.device("cuda"))),
                                 torch.nn.GRUCell(input_size=1, hidden_size=4))
decoder = decoder.Decoder(utils.ODEFuncWrapper(utils.feedforward_nn(4, 4, 64, 3, device=torch.device("cuda"))), utils.feedforward_nn(4, 1, 64, 3, device=torch.device("cuda")))
latent_encoder = utils.feedforward_nn(4, 2, 64, 3, device=torch.device("cuda"))
latent_decoder = utils.feedforward_nn(2, 4, 64, 3, device=torch.device("cuda"))

def function_to_approximate(t):
    return np.sin(t)

low = -8 * np.pi
high = 8 * np.pi

total_time_samples = np.linspace(low, high, 100, dtype=np.float32)
total_function_samples = function_to_approximate(total_time_samples)

# Choose randomly 30 numbers from 0 to 499 (inclusive), without replacement. The result should be sorted.
training_indices = np.concatenate([np.sort(np.random.choice(50, 10, replace=False)), np.array([50])])

training_time = total_time_samples[training_indices]
training_function_vals = total_function_samples[training_indices]

device = torch.device("cuda")
encoder = encoder.to(device)
decoder = decoder.to(device)

# Convert to torch
total_time_samples_torch = torch.from_numpy(total_time_samples).to(device)
total_function_samples_torch = torch.from_numpy(total_function_samples).to(device)
training_time_torch = torch.from_numpy(training_time).to(device)
training_function_vals_torch = torch.from_numpy(training_function_vals).to(device)
training_function_vals_torch_in = training_function_vals_torch.unsqueeze(dim=-1) # Add a dimension to the end, input_dim = 1
forecast_time_samples_torch = total_time_samples_torch[50:]
forecast_function_samples_torch = total_function_samples_torch[50:]


optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

def odernn_run_plot(i, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes):
    ax.clear()
    ax.set_xlim(low, high)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Encoder-Decoder. Epoch: {}".format(i))
    ax.set_xlabel("t")
    ax.set_ylabel("y")

    # Plot ground truth t:total_time_samples, y:total_function_samples. color="red". label="Ground Truth"
    ax.plot(total_time_samples, total_function_samples, color="red", label="Ground Truth")

    # Scatter the encoder predictions t:training_time, y:training_function_vals. color="blue". label="Training samples"
    ax.scatter(training_time, training_function_vals, color="blue", label="Training samples", s=30)

    optimizer.zero_grad()
    latent = encoder(training_time_torch, training_function_vals_torch_in)
    prediction = decoder(forecast_time_samples_torch, latent_decoder(latent_encoder(latent))).squeeze(dim=-1)
    assert prediction.shape == forecast_function_samples_torch.shape

    time_diff = forecast_time_samples_torch - torch.min(forecast_time_samples_torch)
    loss = (torch.abs(prediction - forecast_function_samples_torch) * torch.exp(-0.25 * time_diff)).mean()
    loss.backward()
    optimizer.step()

    # Plot the decoder predictions t:forecast_time_samples, y:prediction. color="green". label="Decoder predictions"
    ax.plot(forecast_time_samples_torch.cpu().detach().numpy(), prediction.cpu().detach().numpy(), color="green", label="Decoder predictions")
    ax.legend()

plot_interactive(UpdateCallback(odernn_run_plot))