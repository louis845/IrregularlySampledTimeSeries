import torch
import numpy as np
import time

def sine_function(x):
    return torch.sin(x) + 0.1

def sample_glucose(x, onset = 0.0, weight = 1.0, decay = 1.0):
    xs = (x - 0.3) * decay / 0.4 - onset
    return 1.2 * weight * torch.exp(-(xs ** 2)) * torch.sigmoid(8 * xs)

def sample_batch_glucose(x, onset, weight, decay):
    """
    Samples a batch of toy glucose values.
    :param x: The x values. A torch tensor of (batch_size, time_vals)
    :param onset: The onsets of the glucose spikes. A torch tensor of (batch_size, num_spikes)
    :param weight: The heights of the glucose spikes. A torch tensor of (batch_size, num_spikes)
    :param decay: The decay of the glucose spikes. A torch tensor of (batch_size)
    :return: A torch tensor of (batch_size, time_vals)
    """
    assert len(x.shape) == 2
    assert len(onset.shape) == 2
    assert len(weight.shape) == 2
    assert len(decay.shape) == 1

    assert x.shape[0] == onset.shape[0] == weight.shape[0] == decay.shape[0]
    assert onset.shape[1] == weight.shape[1]

    return torch.sum(sample_glucose(x.unsqueeze(2), onset.unsqueeze(1), weight.unsqueeze(1), decay.unsqueeze(-1).unsqueeze(-1)), dim=2)

samples_width = np.pi / 20
low = -100
high = 100

def create_mask_from_tensor(batch_size: int, m: int, lengths: torch.Tensor, indices: torch.Tensor, device=torch.device("cuda")) -> torch.Tensor:
    """
    Create a mask from a tensor of indices and lengths.
    :param batch_size: The batch size
    :param m: The number of samples
    :param lengths: A (batch_size,) tensor of lengths
    :param indices: A (batch_size, m) tensor of indices
    :return: A (batch_size, m) tensor of 0s and 1s
    """
    # Increment the indices tensor by 1 elementwise
    indices += 1

    # Set all indices[i, j] with j>lengths[i]-1 to 0
    indices[torch.arange(indices.shape[1], device=device) >= lengths.unsqueeze(1)] = 0

    mask = torch.zeros((batch_size, m+1), device=device, dtype=torch.bool)
    mask[torch.arange(batch_size, device=device).unsqueeze(1), indices] = True
    mask = mask[:, 1:]

    return mask

decay_values = None
def setup_glucose_sampling(device=torch.device("cuda")):
    global decay_values
    #decay_values = (torch.rand(size=(4,), device=device) * 0.2) + 1.0
    decay_values = torch.tensor([1.0515, 1.1733, 1.0265, 1.0005], device=device)
    print("decay_values:   ", decay_values)

def generate_glucose_spikes(batch_size, device=torch.device("cuda")):
    onset = torch.rand(size=(batch_size, 4), device=device) * 2
    onset[:, 0] = -(onset[:, 0] + onset[:, 1] + 1.5)
    onset[:, 1] = -onset[:, 1] - 0.5
    onset[:, 2] = onset[:, 2] + 0.5
    onset[:, 3] = onset[:, 2] + onset[:, 3] + 1.5

    weight = torch.rand(size=(batch_size, 4), device=device) * 0.5 + 1.5
    decay = decay_values[torch.randint(low=0, high=4, size=(batch_size,), device=device)]
    return onset, weight, decay

def sample_time_series(batch_size, min_samples=2, max_samples=10, after_samples=40, device=torch.device("cuda"), mid_val=None, sampling_method="sine"):
    with torch.no_grad():
        msamples_width = torch.tensor(samples_width, device=device)

        if mid_val is None:
            mid_val = torch.randint(low=low, high=high, size=(batch_size,), device=device) # the (untransformed) time end of available data, i.e. the time at which the prediction should start
        else:
            assert mid_val.shape[0] == batch_size

        # randomly sample irregularly spaced time points before the mid_val
        # first generate increasing integer sequence
        before_samples = torch.randint(low=1, high=5, size=(batch_size, max_samples), device=device)
        before_samples = torch.cat([torch.zeros(size=(batch_size, 1), device=device, dtype=before_samples.dtype), torch.cumsum(before_samples, dim=-1)], dim=-1)
        # take the union of the increasing integer sequences, so that it can be represented as a boolean mask
        lengths = torch.randint(low=min_samples, high=max_samples, size=(batch_size,), device=device)
        before_samples_union = create_mask_from_tensor(batch_size, before_samples[:, -1].max().long().item() + 1,
                                lengths=lengths, indices=before_samples, device=device)
        corresponding_values = torch.arange(0, before_samples_union.shape[-1], device=device) # the corresponding values of the time points
        # we only need to keep the values that are actually used
        existing_values = torch.sum(before_samples_union, dim=0) > 0
        before_samples_union = before_samples_union[:, existing_values]
        corresponding_values = corresponding_values[existing_values]
        # flip the before samples
        before_samples_union = torch.flip(before_samples_union, dims=(-1,))
        corresponding_values = -torch.flip(corresponding_values, dims=(-1,))

        # fixed number of samples after the mid_val for the prediction
        after_samples = torch.arange(0, after_samples, device=device)

        # compute the ground truths for each batch
        if sampling_method == "sine":
            ground_truth_values_before = sine_function((corresponding_values + mid_val.unsqueeze(-1)) * msamples_width)
            ground_truth_values_after = sine_function((mid_val.unsqueeze(-1) + after_samples) * msamples_width)
        elif sampling_method == "glucose":
            onset, weight, decay = generate_glucose_spikes(batch_size, device=device)

            ground_truth_values_before = sample_batch_glucose((corresponding_values + mid_val.unsqueeze(-1)) * msamples_width, onset, weight, decay)
            ground_truth_values_after = sample_batch_glucose((mid_val.unsqueeze(-1) + after_samples) * msamples_width, onset, weight, decay)
        else:
            raise ValueError("Unknown sampling method {}".format(sampling_method))

        # obstruct the ground_truth_values_before with the random mask.
        ground_truth_values_before[torch.logical_not(before_samples_union)] = torch.nan

    return corresponding_values * msamples_width, after_samples * msamples_width, ground_truth_values_before, ground_truth_values_after

if __name__ == "__main__":
    print(sample_time_series(10))