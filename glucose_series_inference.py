"""Demo of glucose series inference using gradio packages and the pretrained encoder decoder models."""

import torch
import encoder_ode_rnn
import decoder as decoder_ode_rnn
import utils
import time_series_sampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gradio as gr
import io
import os
import logging

import gradio as gr
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
width = 2
height = 3

input_time_points = np.empty((height, width), dtype=object)
output_time_points = np.empty((height, width), dtype=object)
ground_truth_input = np.empty((height, width), dtype=object)
ground_truth_output = np.empty((height, width), dtype=object)

ground_truth_input_noise = np.empty((height, width), dtype=object)

def render_single_prediction(ax, input_time_points, ground_truth_input, output_time_points, label_str="Predictions", color="blue"):
    with torch.no_grad():
        latent = encoder(input_time_points, ground_truth_input.unsqueeze(-1))
        if use_variational:
            if use_reduce_dims:
                latent = (latent_encoder(latent[0]), latent_variance_encoder(latent[1]))

            latent = latent[0]

            if use_reduce_dims:
                latent = latent_decoder(latent)
            predictions = decoder(output_time_points, latent).squeeze(dim=-1)

            ax.plot(output_time_points.cpu().numpy(), predictions.detach().cpu().numpy(), color=color,
                           label=label_str)
        else:
            if use_reduce_dims:
                latent = latent_decoder(latent_encoder(latent))
            predictions = decoder(output_time_points, latent).squeeze(dim=-1)

            ax.plot(output_time_points.cpu().numpy(), predictions.detach().cpu().numpy(), color=color,
                           label=label_str)

def render_predictions(axs):
    for i in range(height):
        for j in range(width):
            render_single_prediction(axs[i, j], input_time_points[i, j], ground_truth_input[i, j], output_time_points[i, j])
            render_single_prediction(axs[i, j], input_time_points[i, j], ground_truth_input_noise[i, j], output_time_points[i, j], label_str="Predictions (Noisy)", color="green")
def render():
    fig, axs = plt.subplots(height, width, figsize=(12.8, 7.2))
    for i in range(height):
        for j in range(width):
            axs[i, j].clear()
            axs[i, j].set_xlim(input_time_points[i, j].nan_to_num(nan=torch.inf).min().item(),
                               output_time_points[i, j].nan_to_num(nan=-torch.inf).max().item())
            axs[i, j].set_ylim(low, high)
            axs[i, j].set_xlabel("t")
            axs[i, j].set_ylabel("y")

            axs[i, j].plot(output_time_points[i, j].cpu().numpy(), ground_truth_output[i, j].cpu().numpy(), color="red",
                           label="Ground Truth")

            axs[i, j].scatter(input_time_points[i, j].cpu().numpy(), ground_truth_input[i, j].cpu().numpy(),
                              color="blue", label="Irregular samples as input", s=15)
            axs[i, j].scatter(input_time_points[i, j].cpu().numpy(), ground_truth_input_noise[i, j].cpu().numpy(),
                              color="green", label="Input with noise", s=15)

    if models_loaded():
        render_predictions(axs)

    for i in range(height):
        for j in range(width):
            axs[i, j].legend()

    fig.canvas.draw()
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

def callback_update_noise(noise, num_samples, relative_separation, decay_coefficient):
    if ground_truth_input_noise[0, 0] is None:
        return callback_update_samples(noise, num_samples, relative_separation, decay_coefficient)

    for i in range(height):
        for j in range(width):
            z = ground_truth_input_noise[i, j]
            ground_truth_input_noise[i, j] = ground_truth_input[i, j] + torch.normal(mean=0.0, std=noise, size=ground_truth_input[i, j].shape, device=device)
            del z
    return render()

low = -1
high = 5

def callback_update_samples(noise, num_samples, relative_separation, decay_coefficient, large_intake_generation_method):
    time_series_sampler.low = -int(160 / relative_separation)
    time_series_sampler.high = int(240 / relative_separation)
    time_series_sampler.samples_width = relative_separation * 1.0 / 80
    time_series_sampler.decay_values = torch.ones(size=(4,), device=device, dtype=torch.float32) * torch.tensor(decay_coefficient, device=device, dtype=torch.float32)
    if large_intake_generation_method:
        time_series_sampler.set_glucose_spikes_generation_method("other")
    else:
        time_series_sampler.set_glucose_spikes_generation_method("default")


    for i in range(height):
        for j in range(width):
            input_time_points[i, j], output_time_points[i, j], ground_truth_input[i, j], ground_truth_output[i, j]\
                = time_series_sampler.sample_time_series(1, device=device, min_samples=num_samples, max_samples=num_samples+1, after_samples=int(80 / relative_separation),
                                                            sampling_method="glucose")
            input_time_points[i, j] = input_time_points[i, j].squeeze(0)
            output_time_points[i, j] = output_time_points[i, j].squeeze(0)
            ground_truth_input[i, j] = ground_truth_input[i, j].squeeze(0)
            ground_truth_output[i, j] = ground_truth_output[i, j].squeeze(0)

            ground_truth_input_noise[i, j] = ground_truth_input[i, j] + torch.normal(mean=0.0, std=noise, size=ground_truth_input[i, j].shape, device=device)


    return render()

def callback_update_model_predictions(noise, num_samples, relative_separation, decay_coefficient, large_intake_generation_method):
    if input_time_points[0, 0] is None:
        return callback_update_samples(noise, num_samples, relative_separation, decay_coefficient, large_intake_generation_method)
    return render()


encoder, decoder = None, None
latent_encoder, latent_decoder, latent_variance_encoder = None, None, None
use_variational, use_reduce_dims = False, False

def models_loaded():
    return encoder is not None and decoder is not None and latent_encoder is not None and latent_decoder is not None and latent_variance_encoder is not None

def clean_models():
    global encoder, decoder, latent_encoder, latent_decoder, latent_variance_encoder
    if encoder is not None:
        del encoder
        encoder = None
    if decoder is not None:
        del decoder
        decoder = None
    if latent_encoder is not None:
        del latent_encoder
        latent_encoder = None
    if latent_decoder is not None:
        del latent_decoder
        latent_decoder = None
    if latent_variance_encoder is not None:
        del latent_variance_encoder
        latent_variance_encoder = None

def callback_update_model(model_name, variational, reduce_dims, epoch):
    global use_variational, use_reduce_dims
    use_variational, use_reduce_dims = variational, reduce_dims
    clean_models()
    global encoder, decoder, latent_encoder, latent_decoder, latent_variance_encoder
    model_folder_name = model_name[:-10] if "_finetuned" in model_name else model_name
    model_fmt_str = "models/{}/{}_finetune.pt" if "_finetuned" in model_name else "models/{}/{}.pt"
    model_fmt_str_epoch = "models/{}/{}_finetune_{}.pt" if "_finetuned" in model_name else "models/{}/{}_{}.pt"

    if not os.path.isdir("models/{}".format(model_folder_name)):
        return "Model {} does not exist".format(model_name)
    if not os.path.isfile(model_fmt_str_epoch.format(model_folder_name, "encoder", epoch)) and not os.path.isfile(model_fmt_str.format(model_folder_name, "encoder")):
        return "Model {} epoch {} does not exist".format(model_name, epoch)
    try:
        latent_dims = 32
        gru_hidden_dims = latent_dims * 2 if variational else latent_dims
        encoder = encoder_ode_rnn.OdeRNN(latent_dims,
                                         utils.ODEFuncWrapper(utils.feedforward_nn(latent_dims, latent_dims, 128, 3,
                                                                                   device=torch.device("cuda"))),
                                         torch.nn.GRUCell(input_size=1, hidden_size=gru_hidden_dims),
                                         compute_variance=variational)
        decoder = decoder_ode_rnn.Decoder(
            utils.ODEFuncWrapper(utils.feedforward_nn(latent_dims, latent_dims, 128, 3, device=torch.device("cuda"))),
            utils.feedforward_nn(latent_dims, 1, 128, 3, device=torch.device("cuda")))
        latent_encoder = torch.nn.Linear(latent_dims, 2, device=torch.device("cuda"))
        latent_decoder = torch.nn.Linear(2, latent_dims, device=torch.device("cuda"))
        latent_variance_encoder = torch.nn.Linear(latent_dims, 2, device=torch.device("cuda"))
        device = torch.device("cuda")
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        if os.path.isfile(model_fmt_str_epoch.format(model_folder_name, "encoder", epoch)):
            encoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "encoder", epoch)))
            decoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "decoder", epoch)))
            latent_encoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "latent_encoder", epoch)))
            latent_decoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "latent_decoder", epoch)))
            latent_variance_encoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "latent_variance_encoder", epoch)))
            msg = "Successfully loaded model {} at epoch {}".format(model_name, epoch)
        else:
            encoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "encoder")))
            decoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "decoder")))
            latent_encoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "latent_encoder")))
            latent_decoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "latent_decoder")))
            latent_variance_encoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "latent_variance_encoder")))
            msg = "Successfully loaded model {}".format(model_name)
        return msg
    except Exception:
        logging.exception("Error loading model.")
        return "Error loading model. Please try again."

# Define the slider limits and default values
slider_limits = [
    [0.0, 1.0],
    [2, 40],
    [0.5, 2.0],
    [0.3, 1.8]
]
slider_defaults = [0.0, 4, 1.0, 0.9]

demo = gr.Blocks()
with demo:
    gr.Markdown("# Irregularly sampled time series prediction Demo")
    gr.Markdown("This demo uses pretrained models to predict the future of a time series.")

    with gr.Tabs():
        with gr.TabItem("Inference"):
            image = gr.Image(value=None, type="numpy", interactive=False)
            # Create the sliders
            noise = gr.Slider(minimum=slider_limits[0][0], maximum=slider_limits[0][1], step=0.01, value=slider_defaults[0], label="Random noise", interactive=True)
            num_samples = gr.Slider(minimum=slider_limits[1][0], maximum=slider_limits[1][1], step=1, value=slider_defaults[1], label="Number of samples", interactive=True)
            relative_separation = gr.Slider(minimum=slider_limits[2][0], maximum=slider_limits[2][1], step=0.01, value=slider_defaults[2], label="Relative separation", interactive=True)
            decay_coefficient = gr.Slider(minimum=slider_limits[3][0], maximum=slider_limits[3][1], step=0.01, value=slider_defaults[3], label="Decay coefficient", interactive=True)
            large_intake_generation_method = gr.Checkbox(label="Use large glucose intake", interactive=True, value=False)

            # Create the buttons
            button_noise = gr.Button("Regenerate random noise")
            button_samples = gr.Button("Regenerate random samples")
            button_new_model = gr.Button("Predict with new model")
            button_noise.click(callback_update_noise, inputs=[noise, num_samples, relative_separation, decay_coefficient], outputs=[image])
            button_samples.click(callback_update_samples, inputs=[noise, num_samples, relative_separation, decay_coefficient, large_intake_generation_method], outputs=[image])
            button_new_model.click(callback_update_model_predictions, inputs=[noise, num_samples, relative_separation, decay_coefficient, large_intake_generation_method], outputs=[image])


        with gr.TabItem("Choose model"):
            subfolders = os.listdir("models/")
            model_names = []
            for subfolder in subfolders:
                model_names.append(subfolder)
                if os.path.exists("models/" + subfolder + "/encoder_finetune.pt"):
                    model_names.append(subfolder + "_finetuned")

            if len(model_names) == 0:
                print("No models found in models/ folder. Quitting.")
                quit()
            model_loading_status = gr.Textbox("No model loaded", label="Model loading status", interactive=False)


            model_dropdown = gr.Dropdown(model_names, label="Model", interactive=True, value=model_names[0])
            variational = gr.Checkbox(label="Variational", interactive=True, value=False)
            reduce_dims = gr.Checkbox(label="Reduce dims", interactive=True, value=False)
            epoch_select = gr.Slider(minimum=100, maximum=10000, step=100, value=100, label="Epoch", interactive=True)
            model_loading_button = gr.Button("Load model")
            model_loading_button.click(callback_update_model, inputs=[model_dropdown, variational, reduce_dims, epoch_select], outputs=[model_loading_status])

demo.launch()