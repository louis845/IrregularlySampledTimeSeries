import torch
import numpy as np
import time

def ground_truth_function(x):
    return torch.sin(x) + 0.1

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

def sample_time_series(batch_size, min_samples=2, max_samples=10, after_samples=40, device=torch.device("cuda")):
    with torch.no_grad():
        msamples_width = torch.tensor(samples_width, device=device)

        mid_val = torch.randint(low=low, high=high, size=(batch_size,), device=device) # the (untransformed) time end of available data, i.e. the time at which the prediction should start

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
        ground_truth_values_before = ground_truth_function((corresponding_values + mid_val.unsqueeze(-1)) * msamples_width)
        ground_truth_values_after = ground_truth_function((mid_val.unsqueeze(-1) + after_samples) * msamples_width)

        # obstruct the ground_truth_values_before with the random mask.
        ground_truth_values_before[torch.logical_not(before_samples_union)] = torch.nan

    return corresponding_values * msamples_width, after_samples * msamples_width, ground_truth_values_before, ground_truth_values_after

if __name__ == "__main__":
    print(sample_time_series(10))