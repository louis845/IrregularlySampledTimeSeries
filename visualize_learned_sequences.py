import torch
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt

import encoder_ode_rnn
import decoder
import utils
import time_series_sampler

# load trained models here
encoder = encoder_ode_rnn.OdeRNN(4,
                                 utils.ODEFuncWrapper(utils.feedforward_nn(4, 4, 64, 3, device=torch.device("cuda"))),
                                 torch.nn.GRUCell(input_size=1, hidden_size=4))
decoder = decoder.Decoder(utils.ODEFuncWrapper(utils.feedforward_nn(4, 4, 64, 3, device=torch.device("cuda"))), utils.feedforward_nn(4, 1, 64, 3, device=torch.device("cuda")))
latent_encoder = torch.nn.Linear(4, 2, device=torch.device("cuda"))
latent_decoder = torch.nn.Linear(2, 4, device=torch.device("cuda"))


device = torch.device("cuda")
encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.load_state_dict(torch.load("encoder_sine.pt"))
decoder.load_state_dict(torch.load("decoder_sine.pt"))
latent_encoder.load_state_dict(torch.load("latent_encoder_sine.pt"))
latent_decoder.load_state_dict(torch.load("latent_decoder_sine.pt"))

# sample a single series
input_time_points1, output_time_points1, ground_truth_input1, ground_truth_output1 = time_series_sampler.sample_time_series(1, device=device)
input_time_points1 = input_time_points1
output_time_points1 = output_time_points1
ground_truth_input1 = ground_truth_input1[0, :]
ground_truth_output1 = ground_truth_output1[0, :]


while True:
    input_time_points2, output_time_points2, ground_truth_input2, ground_truth_output2 = time_series_sampler.sample_time_series(1, device=device)
    input_time_points2 = input_time_points2
    output_time_points2 = output_time_points2
    ground_truth_input2 = ground_truth_input2[0, :]
    ground_truth_output2 = ground_truth_output2[0, :]

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

# plot the graphs
fig, ax = plt.subplots(2, 1, figsize=(19.2, 10.8))

def plot_graphs():
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

    with torch.no_grad():
        latent = encoder(input_time_points1, ground_truth_input1.unsqueeze(-1))
        latent = latent_decoder(latent_encoder(latent))
        predictions = decoder(output_time_points1, latent).squeeze(dim=-1)
        print(latent)

        latent2 = encoder(input_time_points2, ground_truth_input2.unsqueeze(-1))
        latent2 = latent_decoder(latent_encoder(latent2))
        predictions2 = decoder(output_time_points2, latent2).squeeze(dim=-1)
        print(latent2)

    ax[0].plot(output_time_points1_np, predictions.detach().cpu().numpy(), color="green", label="Predictions")
    ax[1].plot(output_time_points2_np, predictions2.detach().cpu().numpy(), color="green", label="Predictions")

    ax[0].set_title("Function 1. Start pred: {}".format(predictions[0].item()))
    ax[1].set_title("Function 2. Start pred: {}".format(predictions2[0].item()))

    for k in range(2):
        ax[k].legend()

plot_graphs()

plt.show()