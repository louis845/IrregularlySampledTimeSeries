import time
import argparse

import matplotlib.figure
import numpy as np
import torch
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import encoder_ode_rnn
import decoder
import utils
import time_series_sampler

# parse arguments.
parser = argparse.ArgumentParser(description="Train an ODE-RNN on a time series.")
# add boolean argument variational, with description "Whether to use variational encoder decoder, or just plain encoder decoder." Default value False
parser.add_argument("--variational", action="store_true", help="Whether to use variational encoder decoder, or just plain encoder decoder.")
# add string argument name, with description "Name (prefix) of output files." Default value "sine"
parser.add_argument("--name", type=str, default="glucose", help="Name (prefix) of output files.")
# add string argument reduce-dims, with description "Whether to reduce the dimensionality of the latent space to 2." Default value False
parser.add_argument("--reduce-dims", action="store_true", help="Whether to reduce the dimensionality of the latent space to 2.")
# add float argument kl-loss-weight, with description "Weight of the KL loss term. Only used when --variational argument is present." Default value 0.1
parser.add_argument("--kl-loss-weight", type=float, default=0.01, help="Weight of the KL loss term. Only used when --variational argument is present.")
args = parser.parse_args()

print("Using name:   {}".format(args.name))
print("Variational:  {}".format(args.variational))
print("Reduce dims:  {}".format(args.reduce_dims))
print("KL weight:    {}".format(args.kl_loss_weight))
print("-----------------------------------")


def plot_to_mp4(filename, update_callback, figsize=(25.6, 14.4), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

    # Create the left subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # Create the right subplot
    ax4 = fig.add_subplot(gs[:, 1])
    fig.subplots_adjust(hspace=0.5)
    update_callback.set_renderers(fig, [ax1, ax2, ax3, ax4])

    # Create the animation
    anim = animation.FuncAnimation(fig, update_callback, frames=frames)
    anim.save(filename, fps=fps, extra_args=extra_args)

def plot_interactive(update_callback, figsize=(25.6, 14.4), frames=240, fps=30, extra_args=['-vcodec', 'libx264']):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=5, ncols=3, figure=fig)

    # Create the left subplot
    col1 = []
    for k in range(5):
        col1.append(fig.add_subplot(gs[k, 0]))
    col2 = []
    for k in range(5):
        col2.append(fig.add_subplot(gs[k, 1]))

    # Create the right subplot
    ax_final = fig.add_subplot(gs[:, 2])
    fig.subplots_adjust(hspace=0.5)
    update_callback.set_renderers(fig, col1 + col2 + [ax_final])

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

time_series_sampler.setup_glucose_sampling()
time_series_sampler.low = -160
time_series_sampler.high = 240
time_series_sampler.samples_width = 1.0 / 80

latent_dims = 32
gru_hidden_dims = latent_dims * 2 if args.variational else latent_dims
encoder = encoder_ode_rnn.OdeRNN(latent_dims,
                                 utils.ODEFuncWrapper(utils.feedforward_nn(latent_dims, latent_dims, 64, 3, device=torch.device("cuda"))),
                                 torch.nn.GRUCell(input_size=1, hidden_size=gru_hidden_dims), compute_variance=args.variational)
decoder = decoder.Decoder(utils.ODEFuncWrapper(utils.feedforward_nn(latent_dims, latent_dims, 64, 3, device=torch.device("cuda"))), utils.feedforward_nn(latent_dims, 1, 64, 3, device=torch.device("cuda")))
latent_encoder = torch.nn.Linear(latent_dims, 2, device=torch.device("cuda"))
latent_decoder = torch.nn.Linear(2, latent_dims, device=torch.device("cuda"))
latent_variance_encoder = torch.nn.Linear(latent_dims, 2, device=torch.device("cuda"))


device = torch.device("cuda")
encoder = encoder.to(device)
decoder = decoder.to(device)


n = 10

input_time_points, output_time_points, ground_truth_input, ground_truth_output = None, None, None, None
input_time_points_np, output_time_points_np, ground_truth_input_np, ground_truth_output_np = None, None, None, None
plot_tmin, plot_tmax = None, None

def initialize_display_series():
    global input_time_points, output_time_points, ground_truth_input, ground_truth_output
    global input_time_points_np, output_time_points_np, ground_truth_input_np, ground_truth_output_np
    global plot_tmin, plot_tmax

    if input_time_points is not None:
        del input_time_points, output_time_points, ground_truth_input, ground_truth_output
    if input_time_points_np is not None:
        del input_time_points_np, output_time_points_np, ground_truth_input_np, ground_truth_output_np
    if plot_tmin is not None:
        del plot_tmin, plot_tmax

    input_time_points = np.empty((n,), dtype="object")
    output_time_points = np.empty((n,), dtype="object")
    ground_truth_input = np.empty((n,), dtype="object")
    ground_truth_output = np.empty((n,), dtype="object")

    input_time_points[0], output_time_points[0], ground_truth_input[0], ground_truth_output[0] = time_series_sampler.sample_time_series(1, device=device, min_samples=4, max_samples=10, after_samples=80, sampling_method="glucose")
    input_time_points[0] = input_time_points[0].squeeze(0)
    output_time_points[0] = output_time_points[0].squeeze(0)
    ground_truth_input[0] = ground_truth_input[0].squeeze(0)
    ground_truth_output[0] = ground_truth_output[0].squeeze(0)

    i = 1
    while i < n:
        input_time_points[i], output_time_points[i], ground_truth_input[i], ground_truth_output[i] = time_series_sampler.sample_time_series(1, device=device, min_samples=4, max_samples=10, after_samples=80, sampling_method="glucose")
        input_time_points[i] = input_time_points[i].squeeze(0)
        output_time_points[i] = output_time_points[i].squeeze(0)
        ground_truth_input[i] = ground_truth_input[i].squeeze(0)
        ground_truth_output[i] = ground_truth_output[i].squeeze(0)

        if torch.abs(ground_truth_output[i][0] - ground_truth_output[i-1][0]) > 0.2:
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


optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-5)

cbar = None
epoch = 1
ctime = time.time()

if args.variational:
    if args.reduce_dims:
        std_normal = torch.distributions.MultivariateNormal(loc=torch.zeros(2, device=device), covariance_matrix=torch.diag(torch.ones(2, device=device)))
    else:
        std_normal = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dims, device=device), covariance_matrix=torch.diag(torch.ones(latent_dims, device=device)))
    kl_loss_weight = torch.tensor(args.kl_loss_weight, device=device)

initialize_display_series()

def odernn_run_plot(i, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes):
    global cbar, epoch, ctime
    low = -1
    high = 5

    if epoch > 100:
        if epoch % 2 == 0:
            initialize_display_series()

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

    optimizer.zero_grad()
    # sample the time series from time_series_sampler
    batch_size = 5000
    batch_input_time_points, batch_output_time_points, batch_ground_truth_input, batch_ground_truth_output =\
        time_series_sampler.sample_time_series(batch_size, device=device, min_samples=4, max_samples=10, after_samples=80, sampling_method="glucose")

    assert batch_input_time_points.shape[0] == batch_ground_truth_input.shape[1]
    assert batch_output_time_points.shape[0] == batch_ground_truth_output.shape[1]

    # use the ODE-RNN encoder to encode the input time series
    latent = encoder(batch_input_time_points, batch_ground_truth_input.unsqueeze(-1).permute(1, 0, 2))
    # resample latent space if variational, and see whether further compression is needed
    if args.variational:
        if args.reduce_dims:
            latent = (latent_encoder(latent[0]), latent_variance_encoder(latent[1]))
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(loc=latent[0], covariance_matrix=torch.diag_embed(latent[1] ** 2))
        # compute kl_loss here
        kl_loss = torch.distributions.kl_divergence(distribution, std_normal).mean() * kl_loss_weight
        # use reparameterization trick to sample from latent space
        latent = std_normal.rsample((batch_size,)) * latent[1] + latent[0]

        if args.reduce_dims:
            latent = latent_decoder(latent)
    else:
        if args.reduce_dims:
            latent = latent_decoder(latent_encoder(latent))
    # do predictions here
    predictions = decoder(batch_output_time_points, latent).squeeze(dim=-1).permute(1, 0)

    loss = (torch.abs(predictions - batch_ground_truth_output) * torch.exp(-batch_output_time_points).unsqueeze(0)).mean()

    if args.variational:
        (loss + kl_loss).backward()
    else:
        loss.backward()
    optimizer.step()
    scheduler.step()

    # plot predicted values
    with torch.no_grad():
        for j in range(n):
            latent = encoder(input_time_points[j], ground_truth_input[j].unsqueeze(-1))
            if args.variational:
                if args.reduce_dims:
                    latent = (latent_encoder(latent[0]), latent_variance_encoder(latent[1]))

                latent = torch.cat([latent[0].unsqueeze(0), std_normal.rsample((5,)) * latent[1] + latent[0]], dim=0)

                if args.reduce_dims:
                    latent = latent_decoder(latent)
                predictions = decoder(output_time_points[j], latent).squeeze(dim=-1)
                predictions = predictions.squeeze(dim=-1).detach().cpu().numpy()

                ax[j].plot(output_time_points_np[j], predictions[:, 0], color="green", label="Predictions")
                ax[j].plot(output_time_points_np[j], predictions[:, 1:], color="yellow", label="VED Predictions")

                ax[j].set_title("Function {}. Epoch: {} Start pred: {}".format(j, epoch, predictions[0, 0].item()))
            else:
                if args.reduce_dims:
                    latent = latent_decoder(latent_encoder(latent))
                predictions = decoder(output_time_points[j], latent).squeeze(dim=-1)

                ax[j].plot(output_time_points_np[j], predictions.detach().cpu().numpy(), color="green", label="Predictions")

                ax[j].set_title("Function {}. Epoch: {} Start pred: {}".format(j, epoch, predictions[0].item()))

            ax[j].legend()

    # plot embeddings
    with torch.no_grad():
        if cbar is not None:
            cbar.remove()

        ax[n].clear()

        if args.variational:
            scatter_points_num = 205
        else:
            scatter_points_num = 4100
        mid_vals = torch.randint(low=time_series_sampler.low, high=time_series_sampler.high, size=(scatter_points_num,), device=device)

        batch_input_time_points, batch_output_time_points, batch_ground_truth_input, batch_ground_truth_output = \
            time_series_sampler.sample_time_series(scatter_points_num, device=device, mid_val=mid_vals, min_samples=4, max_samples=10, after_samples=80, sampling_method="glucose")
        color_values = batch_ground_truth_input[:, 0]

        embeddings = encoder(batch_input_time_points, batch_ground_truth_input.unsqueeze(-1).permute(1, 0, 2))
        if args.reduce_dims:
            if args.variational:
                embeddings = (latent_encoder(embeddings[0]), latent_variance_encoder(embeddings[1]))
                random_embeddings = std_normal.rsample((scatter_points_num * 20,)) * embeddings[1].repeat_interleave(20, dim=0) + embeddings[0].repeat_interleave(20, dim=0)
                embeddings = torch.cat([embeddings[0], random_embeddings], dim=0)
            else:
                embeddings = latent_encoder(embeddings)

            x = embeddings[:, 0].detach().cpu().numpy()
            y = embeddings[:, 1].detach().cpu().numpy()
        else:
            if args.variational:
                random_embeddings = std_normal.rsample((scatter_points_num * 20,)) * embeddings[1].repeat_interleave(20, dim=0) + embeddings[0].repeat_interleave(20, dim=0)
                embeddings = torch.cat([embeddings[0], random_embeddings], dim=0)
            # compute PCA to reduce the embeddings to 2 dimensions.
            embeddings = embeddings - embeddings.mean(dim=0)
            U, S, V = np.linalg.svd(embeddings.cpu().numpy())
            embeddings = np.matmul(embeddings.cpu().numpy(), V[:, :2])
            x = embeddings[:, 0]
            y = embeddings[:, 1]

        if args.variational:
            color = torch.cat([color_values, color_values.repeat_interleave(20)], dim=0).detach().cpu().numpy()
            sc = ax[n].scatter(x[scatter_points_num:], y[scatter_points_num:], c=color[scatter_points_num:],
                               cmap="viridis",
                               s=2)
            sc = ax[n].scatter(x[:scatter_points_num], y[:scatter_points_num], c=color[:scatter_points_num],
                               cmap="viridis",
                               s=15)
        else:
            color = color_values.detach().cpu().numpy()
            sc = ax[n].scatter(x, y, c=color, cmap="viridis")
        ax[n].set_xlim(np.min(x), np.max(x))
        ax[n].set_ylim(np.min(y), np.max(y))

        cbar = plt.colorbar(sc)
        cbar.set_label("Glucose level at prediction")
        if args.variational:
            if args.reduce_dims:
                ax[n].set_title("Embeddings. Loss: {}   KL_loss: {}   LR: {}".format(loss.item(), kl_loss.item(), scheduler.get_last_lr()))
            else:
                ax[n].set_title("Embeddings (PCA). Loss: {}   KL_loss: {}   LR: {}".format(loss.item(), kl_loss.item(), scheduler.get_last_lr()))
        else:
            if args.reduce_dims:
                ax[n].set_title("Embeddings. Loss: {}   LR: {}".format(loss.item(), scheduler.get_last_lr()))
            else:
                ax[n].set_title("Embeddings (PCA). Loss: {}   LR: {}".format(loss.item(), scheduler.get_last_lr()))

    epoch += 1
    if epoch % 1000 == 0:
        ctime = time.time() - ctime
        print("Epoch: {} Time taken: {}".format(epoch, ctime))
        ctime = time.time()



ctime = time.time()
plot_interactive(UpdateCallback(odernn_run_plot), frames=2400) # "{}.mp4".format(args.name)
print("Time taken: {}".format(time.time() - ctime))

# Save the learned models encoder, decoder, latent_encoder, latent_decoder to files encoder.pt, decoder.pt, latent_encoder.pt, latent_decoder.pt
torch.save(encoder.state_dict(), "encoder_{}.pt".format(args.name))
torch.save(decoder.state_dict(), "decoder_{}.pt".format(args.name))
torch.save(latent_encoder.state_dict(), "latent_encoder_{}.pt".format(args.name))
torch.save(latent_decoder.state_dict(), "latent_decoder_{}.pt".format(args.name))
torch.save(latent_variance_encoder.state_dict(), "latent_variance_encoder_{}.pt".format(args.name))