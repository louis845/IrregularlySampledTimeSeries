import math
import time
import os

import numpy as np
import torch
import pandas as pd

import encoder_ode_rnn
import decoder as decoder_ode_rnn
import utils
import time_series_sampler

import argparse

parser = argparse.ArgumentParser(description="Compute the loss of the models.")
parser.add_argument("--eval_batch_size", type=int, default=20000, help="The batch size used for evaluating the metrics.")
parser.add_argument("--reduce_dims", action="store_true", help="If this argument is added, we reduce the dimensions of the latent space.")

args = parser.parse_args()

if args.reduce_dims:
    print("Reduce dim models are not supported.")
    quit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder, decoder = None, None
latent_encoder, latent_decoder, latent_variance_encoder = None, None, None
use_variational, use_reduce_dims = False, False

time_series_sampler.low = -160
time_series_sampler.high = 240
time_series_sampler.samples_width = 1.0 / 80
time_series_sampler.setup_glucose_sampling_with_fixed_decay(0.5)
time_series_sampler.set_glucose_spikes_generation_method("other")

def compute_loss(encoder, decoder, latent_encoder, latent_decoder):
    # sample
    batch_size = args.eval_batch_size
    batch_input_time_points, batch_output_time_points, batch_ground_truth_input, batch_ground_truth_output = \
        time_series_sampler.sample_time_series(batch_size, device=device, min_samples=4, max_samples=10,
                                               after_samples=80, sampling_method="glucose")

    assert batch_input_time_points.shape[0] == batch_ground_truth_input.shape[1]
    assert batch_output_time_points.shape[0] == batch_ground_truth_output.shape[1]

    # predict and compute loss
    latent = encoder(batch_input_time_points, batch_ground_truth_input.unsqueeze(-1).permute(1, 0, 2))
    if args.reduce_dims:
        latent = latent_decoder(latent_encoder(latent)) # basically this won't happen, see 'if args.reduce_dims' above
    predictions = decoder(batch_output_time_points, latent).squeeze(dim=-1).permute(1, 0)

    loss = (torch.abs(predictions - batch_ground_truth_output) * torch.exp(-2.0 * batch_output_time_points).unsqueeze(0)).mean()

    del batch_input_time_points, batch_output_time_points, batch_ground_truth_input, batch_ground_truth_output, latent, predictions

    return loss.item()

def list_models():
    return [f for f in os.listdir("models") if os.path.isdir(os.path.join("models", f))]

def load_model(model_name, epoch):
    model_folder_name = model_name[:-10] if "_finetuned" in model_name else model_name
    model_fmt_str = "models/{}/{}_finetune.pt" if "_finetuned" in model_name else "models/{}/{}.pt"
    model_fmt_str_epoch = "models/{}/{}_finetune_{}.pt" if "_finetuned" in model_name else "models/{}/{}_{}.pt"

    if not os.path.isdir("models/{}".format(model_folder_name)):
        raise "Model {} does not exist".format(model_name)
    if not os.path.isfile(model_fmt_str_epoch.format(model_folder_name, "encoder", epoch)) and not os.path.isfile(model_fmt_str.format(model_folder_name, "encoder")):
        raise "Model {} epoch {} does not exist".format(model_name, epoch)
    latent_dims = 32
    gru_hidden_dims = latent_dims
    encoder = encoder_ode_rnn.OdeRNN(latent_dims,
                                     utils.ODEFuncWrapper(utils.feedforward_nn(latent_dims, latent_dims, 128, 3,
                                                                               device=torch.device("cuda"))),
                                     torch.nn.GRUCell(input_size=1, hidden_size=gru_hidden_dims),
                                     compute_variance=False)
    decoder = decoder_ode_rnn.Decoder(
        utils.ODEFuncWrapper(utils.feedforward_nn(latent_dims, latent_dims, 128, 3, device=torch.device("cuda"))),
        utils.feedforward_nn(latent_dims, 1, 128, 3, device=torch.device("cuda")))
    latent_encoder = torch.nn.Linear(latent_dims, 2, device=torch.device("cuda"))
    latent_decoder = torch.nn.Linear(2, latent_dims, device=torch.device("cuda"))
    device = torch.device("cuda")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if os.path.isfile(model_fmt_str_epoch.format(model_folder_name, "encoder", epoch)):
        encoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "encoder", epoch)))
        decoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "decoder", epoch)))
        latent_encoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "latent_encoder", epoch)))
        latent_decoder.load_state_dict(torch.load(model_fmt_str_epoch.format(model_folder_name, "latent_decoder", epoch)))
    else:
        encoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "encoder")))
        decoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "decoder")))
        latent_encoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "latent_encoder")))
        latent_decoder.load_state_dict(torch.load(model_fmt_str.format(model_folder_name, "latent_decoder")))

    return encoder, decoder, latent_encoder, latent_decoder


loss_info = np.zeros((len(list_models()), 5))
for model in list_models():
    # We use the finetuned model whenever finetune is in the folder - that means it is trained with pretraining -> finetuning
    # Otherwise it is trained directly without pretraining
    if os.path.isfile(os.path.join("models", model, "encoder_finetune.pt")):
        model_name = "{}_finetuned".format(model)
    else:
        model_name = model

    print("Computing loss for model {}".format(model_name))
    ctime = time.time()

    for epoch in [100, 200, 300, 400, 500]:
        encoder, decoder, latent_encoder, latent_decoder = load_model(model_name, epoch)
        losses = np.zeros(shape=(10), dtype=np.float32)
        for i in range(10):
            losses[i] = compute_loss(encoder, decoder, latent_encoder, latent_decoder)
        loss = np.mean(losses)
        loss_info[list_models().index(model), list([100, 200, 300, 400, 500]).index(epoch)] = loss

        del encoder, decoder, latent_encoder, latent_decoder
        torch.cuda.empty_cache()

    print("Completed. Time taken: {}".format(time.time() - ctime))
loss_info = pd.DataFrame(loss_info, index=list_models(), columns=[100, 200, 300, 400, 500])
loss_info.to_csv("metrics.csv")