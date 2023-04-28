import torch
import torchdiffeq

def constant_dummy(x):
    return x

class OdeRNN(torch.nn.Module):
    def __init__(self, latent_dims, ode_func: torch.nn.Module, gru_cell: torch.nn.GRUCell, compute_variance = False, projection = constant_dummy):
        """
        :param latent_dims: The dimensions of the latent variables.
        :param ode_func: The ODE function to use
        :param compute_variance: Whether to compute the variance of the predictions. If True, the predictions will be returned as a 2 * latent_dims vector, to include the variance (confidence) of the predictions.
        If False, the predictions will be returned as a latent_dims vector.
        """
        super(OdeRNN, self).__init__()
        self.ode_func = ode_func
        self.gru_cell = gru_cell
        self.compute_variance = compute_variance
        self.final_projection = constant_dummy
        self.latent_dims = latent_dims

    def compute_gru(self, x_time_slice, hidden_state_mean, hidden_state_variance=None):
        if self.compute_variance:
            hidden_state = torch.cat((hidden_state_mean, hidden_state_variance), dim=-1)
        else:
            hidden_state = hidden_state_mean

        shape = hidden_state.shape

        # reshape the hidden state and x time slice to be of shape (B, hidden_state.shape[-1]) and (B, x_time_slice.shape[-1]) respectively.
        hidden_state = hidden_state.reshape(-1, shape[-1])
        x_time_slice = x_time_slice.reshape(-1, x_time_slice.shape[-1])

        # compute the new hidden state
        new_hidden_state = self.gru_cell(x_time_slice, hidden_state)

        # reshape the new hidden state to be of shape (..., hidden_state.shape[-1])
        new_hidden_state = new_hidden_state.reshape(*shape[:-1], shape[-1])

        if self.compute_variance:
            return new_hidden_state[..., :self.latent_dims], new_hidden_state[..., self.latent_dims:]
        else:
            return new_hidden_state


    def forward(self, time_steps, obs_x):
        """
        :param time_steps: The time steps to predict the time series on.
        :param obs_x: The observed values of the time series. Should be a tensor of shape (len(time_steps), ..., input_dims).
        :return: A tensor containing the predicted values (states). This would be a tensor of shape (..., total_latent_dims).
        Recall that total_latent_dims =  2 * latent_dims if compute_variance = True, and total_latent_dims = latent_dims if compute_variance = False.

        The mask is inferred from the nan values in the obs_x tensor. If there is a nan value at a certain time step, the corresponding latent state is not updated.
        """

        # Compute the mask from the nan values in the obs_x tensor.
        with torch.no_grad():
            mask = torch.sum(torch.isnan(obs_x), dim=-1) == 0
            obs_x_imputed = torch.where(torch.isnan(obs_x), torch.zeros_like(obs_x), obs_x)

        assert obs_x.shape[0] == time_steps.shape[0], "obs_x and time_steps should have the same length"
        assert len(obs_x.shape) > 1, "obs_x should have at least 2 dimensions. The shape should be (len(time_steps)," \
                                     "..., input_dims), where ... are optional batch dimensions"
        assert len(time_steps.shape) == 1, "time_steps should be a 1D tensor"
        assert (time_steps.shape[0] > 0), "time_steps should have at least one element"
        assert(torch.sum(torch.isnan(time_steps)) == 0), "time_steps should not have any nan values"
        assert mask[0, ...].all(), "The first time step should have no nan values"
        assert mask[-1, ...].all(), "The last time step should have no nan values"

        # compute the first state using the obs_x, which are zeros fed into the GRU directly
        if self.compute_variance:
            latent_state_mean = torch.zeros(obs_x.shape[1:-1] + (self.latent_dims,), device=obs_x.device, dtype=torch.float32)
            latent_state_variance = torch.zeros(obs_x.shape[1:-1] + (self.latent_dims,), device=obs_x.device, dtype=torch.float32)
            latent_state_mean, latent_state_variance = self.compute_gru(obs_x[0, ...], latent_state_mean, latent_state_variance)
        else:
            latent_state_mean = torch.zeros(obs_x.shape[1:-1] + (self.latent_dims,), device=obs_x.device, dtype=torch.float32)
            latent_state_mean = self.compute_gru(obs_x[0, ...], latent_state_mean)


        # for each iteration, the latent_state is updated first with the ODE, then with the GRU. If the values of x is unknown, we do not use the GRU, which means that the ODE is continuously integrated.
        for iter in range(time_steps.shape[0] - 1):
            # use the ODE to forward the current latent states. only the mean should be updated, not the variance
            latent_state_mean = torchdiffeq.odeint_adjoint(self.ode_func, latent_state_mean, time_steps[iter:(iter+2)], rtol=1e-2, atol=1e-4)[1, ...]

            # plug into GRU only if the mask is 1.0
            # first compute the result of the GRU cell with obs_x_imputed[iter, ...] as input
            if self.compute_variance:
                new_latent_state_mean, new_latent_state_variance = self.compute_gru(obs_x_imputed[iter+1, ...], latent_state_mean, latent_state_variance)
                latent_state_variance = torch.where(mask[iter+1, ...].unsqueeze(-1), new_latent_state_variance, latent_state_variance)
            else:
                new_latent_state_mean = self.compute_gru(obs_x_imputed[iter+1, ...], latent_state_mean)

            latent_state_mean = torch.where(mask[iter+1, ...].unsqueeze(-1), new_latent_state_mean, latent_state_mean)

        if self.compute_variance:
            return self.final_projection((latent_state_mean, latent_state_variance))
        else:
            return self.final_projection(latent_state_mean)

