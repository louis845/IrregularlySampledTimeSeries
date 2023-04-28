import torch
import torchdiffeq

class Decoder(torch.nn.Module):
    def __init__(self, ode_func: torch.nn.Module, projector: torch.nn.Module):
        """
        :param ode_func: The ODE function to use.
        :param projector: The projector to use. This projects the high-dim ODE predictions to the low-dimensional time series.
        :param device: The device to use for the model.
        """
        super(Decoder, self).__init__()
        self.ode_func = ode_func
        self.projector = projector

    def forward(self, time_steps_prediction, latent_variables):
        """
        :param time_steps_prediction: The time steps to predict the time series on.
        :param latent_variables: The latent variables to use for the prediction. Should be a tensor of shape (..., latent_dims).
        :return: A tensor containing the predicted values (states). This would be a tensor of shape (len(time_steps_prediction), ..., time_series_dims).
        """
        return self.projector(torchdiffeq.odeint_adjoint(self.ode_func, latent_variables, time_steps_prediction, rtol=1e-2, atol=1e-4))